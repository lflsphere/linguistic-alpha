"""Audit and build the reusable local Capital IQ FinBERT13 feature store."""

from __future__ import annotations

import argparse
from collections import deque
from contextlib import closing, contextmanager
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
from time import perf_counter
import zlib

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from finbert_runtime import MODEL_FILES, REVISION

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STORE = ROOT / "data/derived/finbert"
DEFAULT_MODEL_TABLE = ROOT / "outputs/sp500_financial_extension/sp500_model_extended.csv"
REFERENCE = "9cf2d0e6f410006b20ba9db183698ef251ac7f36:it_finbert_features.py"
SECTIONS = ("prepared", "answers")
FEATURES = (
    "fb_sent_prepared", "fb_sent_answers", "fb_pos_pct_prepared", "fb_neg_pct_prepared",
    "fb_pos_pct_answers", "fb_neg_pct_answers", "fb_sent_std_prepared", "fb_sent_std_answers",
    "fb_sent_gap", "fb_sent_prepared_chg", "fb_sent_answers_chg",
    "fb_n_sent_prepared", "fb_n_sent_answers",
)
KEYS = ("transcriptid", "permno", "clock", "year", "sector")
SPLIT = re.compile(r"(?<=[.!?])\s+")
ENGINE = "it-top-label-strict-prior-v1"
SCORE_SCHEMA = pa.schema([
    ("transcriptid", pa.int64()), ("section", pa.string()), ("sentence_index", pa.int32()),
    ("sentence_sha256", pa.string()), ("top_label", pa.string()), ("confidence", pa.float32()),
    ("token_count_before_truncation", pa.int32()), ("character_truncated", pa.bool_()),
])


@dataclass(frozen=True)
class Call:
    transcriptid: int
    source: str
    source_sha: str
    sections: tuple
    legacy_answer_paragraphs_omitted: int


@dataclass(frozen=True)
class Coverage:
    source_sha: str
    lengths: tuple
    character_truncations: int
    omitted: int

    @classmethod
    def from_call(cls, call):
        return cls(call.source_sha, tuple(map(len, call.sections)),
                   sum(flag for section in call.sections for _, flag in section),
                   call.legacy_answer_paragraphs_omitted)


def stable_digest(path):
    h = sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_write(path, payload):
    temp = path.with_suffix(path.suffix + ".part")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")
    temp.replace(path)


def parquet_write(path, frame):
    temp = path.with_suffix(path.suffix + ".part")
    frame.to_parquet(temp, index=False)
    temp.replace(path)


def local_path(path):
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(STORE) or resolved == STORE:
        raise ValueError(f"Expected a location inside {STORE}")
    return resolved


@contextmanager
def writer_lock(archive):
    lock = archive / "writer.lock"
    with lock.open("a+b") as stream:
        if not stream.tell():
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def source_name(path):
    return path.resolve().relative_to(ROOT).as_posix()


def events_from(path, sectors):
    header = pd.read_csv(path, nrows=0)
    if "clock" in header:
        fields = ["transcriptid", "permno", "clock", "sector"]
        data = pd.read_csv(path, usecols=fields)
        data["clock"] = pd.to_datetime(data.clock, utc=True, errors="raise", format="mixed")
    else:
        fields = ["transcriptid", "permno", "call_date", "call_time_utc", "sector"]
        data = pd.read_csv(path, usecols=fields)
        data["clock"] = pd.to_datetime(data.call_date + " " + data.call_time_utc,
                                      utc=True, errors="coerce", format="mixed")
        data["year"] = pd.to_datetime(data.call_date).dt.year.astype(int)
    data = data.loc[data.sector.isin(sectors)].copy()
    if "year" not in data:
        data["year"] = data.clock.dt.year
    if data.empty or not data.transcriptid.is_unique or data.transcriptid.isna().any():
        raise ValueError("No unique transcript identifiers for the requested sector(s)")
    return data[list(KEYS)].sort_values(["year", "clock", "transcriptid"])


def sentences(text):
    pieces = [part.strip() for part in SPLIT.split(text.strip())]
    return tuple((part[:512], len(part) > 512) for part in pieces if len(part) >= 10)


def sections(group):
    prepared = "\n\n".join(group.loc[group.transcriptcomponenttypeid.eq(2), "componenttext"])
    dialogue = group.loc[group.transcriptcomponenttypeid.isin([3, 4])]
    turns = [("ANSWER: " if code == 4 else "QUESTION: ") + text
             for code, text in zip(dialogue.transcriptcomponenttypeid, dialogue.componenttext)]
    blocks = "\n\n".join(turns).split("\n\n")
    answers = "\n".join(block[7:].strip() for block in blocks if block.startswith("ANSWER:"))
    omitted = int(group.loc[group.transcriptcomponenttypeid.eq(4), "componenttext"]
                  .str.count("\n\n").sum())
    return (sentences(prepared), sentences(answers)), omitted


def source_calls(events):
    """Read canonical raw shards first, then team shards for residual IDs."""
    desired = set(events.transcriptid.astype(int))
    columns = ["transcriptid", "componentorder", "transcriptcomponenttypeid", "componenttext"]
    roots = [(ROOT / "data/raw", "ciq_transcript_component_*.parquet"),
             (ROOT / "data/team_pit/sp500_pit", "ciq_transcript_component_all_*.parquet")]
    for folder, pattern in roots:
        for file in sorted(folder.glob(pattern)):
            if not desired:
                return
            reader = pq.ParquetFile(file)
            ids = pa.array(sorted(desired), type=reader.schema_arrow.field("transcriptid").type)
            selected = []
            for batch in reader.iter_batches(batch_size=8192, columns=columns, use_threads=False):
                matching = batch.filter(pc.is_in(batch.column(0), value_set=ids))
                if matching.num_rows:
                    selected.append(matching)
            if not selected:
                continue
            raw = pa.Table.from_batches(selected).to_pandas()
            raw = raw.drop_duplicates().sort_values(["transcriptid", "componentorder"])
            if raw.duplicated(["transcriptid", "componentorder"]).any():
                raise ValueError(f"Conflicting component order in {source_name(file)}")
            raw["componenttext"] = raw.componenttext.fillna("")
            for tid, group in raw.groupby("transcriptid", sort=False):
                tid = int(tid)
                if not group.componenttext.str.strip().ne("").any():
                    continue
                desired.remove(tid)
                content = group[columns[1:]].to_json(orient="values", force_ascii=False)
                payload, omitted = sections(group)
                yield Call(tid, source_name(file), sha256(content.encode("utf-8")).hexdigest(),
                           payload, omitted)


def setup(db):
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=FULL")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS calls (
            transcriptid INTEGER PRIMARY KEY, permno INTEGER, clock TEXT, year INTEGER NOT NULL,
            sector TEXT NOT NULL, source TEXT NOT NULL, source_sha TEXT NOT NULL,
            sentence_ids BLOB NOT NULL, omitted INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS scores (
            sentence_sha TEXT PRIMARY KEY, label INTEGER NOT NULL, confidence REAL NOT NULL,
            token_count INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
    """)


def hashes_for_model(archive):
    return {name: stable_digest(archive / "model" / REVISION / name) for name in MODEL_FILES}


def bind_cache(db, archive):
    settings = json.dumps({"engine": ENGINE, "revision": REVISION,
                           "model_sha256": hashes_for_model(archive)}, sort_keys=True)
    row = db.execute("SELECT value FROM settings WHERE key='fingerprint'").fetchone()
    if row and settings != row[0]:
        raise ValueError("Cache identity changed. Preserve this cache and use a new --store-name.")
    with db:
        db.execute("INSERT OR IGNORE INTO settings VALUES ('fingerprint', ?)", (settings,))
    return json.loads(settings)


def pack_call(call, event):
    links = [[(sha256(text.encode("utf-8")).hexdigest(), truncated) for text, truncated in section]
             for section in call.sections]
    fields = (call.transcriptid, None if pd.isna(event.permno) else int(event.permno),
              None if pd.isna(event.clock) else event.clock.isoformat(), int(event.year), event.sector,
              call.source, call.source_sha, zlib.compress(json.dumps(links).encode("utf-8")),
              call.legacy_answer_paragraphs_omitted)
    return fields, {key: text for link, section in zip(links, call.sections)
                    for (key, _), (text, _) in zip(link, section)}


def lookup(db, keys):
    found = {}
    keys = list(keys)
    for offset in range(0, len(keys), 800):
        group = keys[offset:offset + 800]
        params = ",".join("?" for _ in group)
        rows = db.execute(f"SELECT * FROM scores WHERE sentence_sha IN ({params})", group)
        found.update((key, (label, confidence, length)) for key, label, confidence, length in rows)
    return found


def process(db, group, make_scorer, batch_size):
    wanted = {}
    rows = []
    for call, event in group:
        row, keyed = pack_call(call, event)
        rows.append(row)
        wanted.update(keyed)
    present = lookup(db, wanted)
    inferences = [key for key in wanted if key not in present]
    emitted = []
    if inferences:
        model = make_scorer()
        for start in range(0, len(inferences), 1024):
            new = inferences[start:start + 1024]
            probabilities, lengths = model.score([wanted[key] for key in new], batch_size)
            labels = probabilities.argmax(axis=1)
            emitted.extend((key, int(label), float(row[label]), int(length))
                           for key, label, row, length in zip(new, labels, probabilities, lengths))
    for new in emitted:
        if not 0 <= new[1] <= 2 or not 1 / 3 - 1e-6 <= new[2] <= 1 or new[3] < 2:
            raise ValueError("Invalid classifier output")
    with db:
        db.executemany("INSERT OR IGNORE INTO scores VALUES (?, ?, ?, ?)", emitted)
        db.executemany("""INSERT OR REPLACE INTO calls VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""", rows)
    return len(inferences)


def aggregate(scored, section):
    nets = np.array([confidence if label == 0 else -confidence if label == 1 else 0.
                     for label, confidence, _ in scored], dtype=float)
    labels = np.array([row[0] for row in scored])
    return {f"fb_sent_{section}": float(nets.mean()) if len(nets) else np.nan,
            f"fb_pos_pct_{section}": float((labels == 0).mean()) if len(nets) else np.nan,
            f"fb_neg_pct_{section}": float((labels == 1).mean()) if len(nets) else np.nan,
            f"fb_sent_std_{section}": float(nets.std(ddof=0)) if len(nets) else np.nan,
            f"fb_n_sent_{section}": len(nets)}


def prior_changes(features):
    for name in SECTIONS:
        features[f"fb_sent_{name}_chg"] = np.nan
    observed = features.dropna(subset=["permno", "clock"])
    for _, history in observed.groupby("permno", sort=False):
        prior = deque(maxlen=4)
        for _, current in history.sort_values(["clock", "transcriptid"]).groupby("clock", sort=False):
            if prior:
                values = np.asarray(prior, dtype=float)
                for column, name in enumerate(SECTIONS):
                    past = values[:, column]
                    past = past[np.isfinite(past)]
                    if len(past):
                        features.loc[current.index, f"fb_sent_{name}_chg"] = current[f"fb_sent_{name}"] - past.mean()
            prior.extend(current[[f"fb_sent_{name}" for name in SECTIONS]].itertuples(index=False, name=None))
    return features


def publish(db, archive):
    sentence_path = archive / "sentence_scores.parquet"
    temp = sentence_path.with_suffix(".parquet.part")
    call_rows, score_rows = [], []
    with pq.ParquetWriter(temp, SCORE_SCHEMA, compression="zstd") as writer:
        for tid, permno, clock, year, sector, source, digest, packed, omitted in db.execute(
                "SELECT * FROM calls ORDER BY transcriptid"):
            links = json.loads(zlib.decompress(packed))
            scores = lookup(db, {key for section in links for key, _ in section})
            row = dict(transcriptid=tid, permno=permno, clock=clock, year=year, sector=sector,
                       source_path=source, component_sha256=digest,
                       legacy_answer_paragraphs_omitted=omitted)
            for section, occurrences in zip(SECTIONS, links):
                row.update(aggregate([scores[key] for key, _ in occurrences], section))
                for idx, (key, truncated) in enumerate(occurrences):
                    label, confidence, length = scores[key]
                    score_rows.append(dict(transcriptid=tid, section=section, sentence_index=idx + 1,
                                           sentence_sha256=key, top_label=("positive", "negative", "neutral")[label],
                                           confidence=confidence, token_count_before_truncation=length,
                                           character_truncated=truncated))
            row["fb_sent_gap"] = row["fb_sent_prepared"] - row["fb_sent_answers"]
            call_rows.append(row)
            if len(score_rows) >= 16384:
                writer.write_table(pa.Table.from_pylist(score_rows, schema=SCORE_SCHEMA))
                score_rows.clear()
        if score_rows:
            writer.write_table(pa.Table.from_pylist(score_rows, schema=SCORE_SCHEMA))
    if not call_rows:
        temp.unlink()
        raise ValueError("No validated sentence-cache calls to publish")
    features = prior_changes(pd.DataFrame(call_rows))
    features["clock"] = pd.to_datetime(features.clock, utc=True, format="mixed")
    features["permno"] = features.permno.astype("Int64")
    parquet_write(archive / "call_features.parquet", features)
    temp.replace(sentence_path)
    return features


def audit(events, calls, cached):
    audit_rows = []
    for (sector, year), group in events.groupby(["sector", "year"]):
        found = [(tid, calls[tid]) for tid in group.transcriptid if tid in calls]
        audit_rows.append(dict(sector=sector, year=int(year), eligible_universe=len(group),
                               text_accessible=len(found), text_missing=len(group) - len(found),
                               prepared_available=sum(bool(info.lengths[0]) for _, info in found),
                               answers_available=sum(bool(info.lengths[1]) for _, info in found),
                               prepared_sentences=sum(info.lengths[0] for _, info in found),
                               answer_sentences=sum(info.lengths[1] for _, info in found),
                               character_truncations=sum(info.character_truncations for _, info in found),
                               legacy_answer_paragraphs_omitted=sum(info.omitted for _, info in found),
                               validated_cached=sum(cached.get(tid) == info.source_sha for tid, info in found)))
    return pd.DataFrame(audit_rows)


def store_coverage(db, archive, events, source_sha, current):
    """Keep separate coverage contracts for each requested input universe."""
    retained = {}
    for sector, subset in current.groupby("sector"):
        membership = events.loc[events.sector.eq(sector), "transcriptid"].astype(int).sort_values()
        key = f"{source_sha}:{sector}"
        retained[key] = {
            "event_universe_sha256": source_sha, "sector": sector, "events": len(membership),
            "transcript_list_sha256": sha256(json.dumps(membership.tolist()).encode("utf-8")).hexdigest(),
            "complete": bool(subset.validated_cached.equals(subset.text_accessible)),
            "annual": subset.to_dict(orient="records"),
        }
    with db:
        db.executemany("INSERT OR REPLACE INTO settings VALUES (?, ?)",
                       [("coverage:" + key, json.dumps(value, allow_nan=False)) for key, value in retained.items()])
    contracts = {key[9:]: json.loads(payload) for key, payload in db.execute(
        "SELECT * FROM settings WHERE key LIKE 'coverage:%' ORDER BY key")}
    rows = [dict(universe_sha256=record["event_universe_sha256"], **row)
            for record in contracts.values() for row in record["annual"]]
    pd.DataFrame(rows).to_csv(archive / "coverage_summary.csv", index=False)
    return contracts


def feature_dictionary():
    values = {}
    for section in SECTIONS:
        values[f"fb_sent_{section}"] = "Mean top-label signed confidence. Neutral contributes zero."
        values[f"fb_pos_pct_{section}"] = "Share of sentences classified positive."
        values[f"fb_neg_pct_{section}"] = "Share of sentences classified negative."
        values[f"fb_sent_std_{section}"] = "Population SD of signed confidence. One sentence gives zero."
        values[f"fb_n_sent_{section}"] = "Count of retained sentences. Empty observed section gives zero."
        values[f"fb_sent_{section}_chg"] = (
            "Current mean minus the observed mean among up to four strictly earlier calls at the same PERMNO."
        )
    values["fb_sent_gap"] = "Prepared mean minus answer mean."
    return {feature: values[feature] for feature in FEATURES}


def verify(archive, require_complete=True):
    manifest = json.loads((archive / "feature_manifest.json").read_text(encoding="utf-8"))
    if manifest["revision"] != REVISION or manifest["engine"] != ENGINE:
        raise ValueError("Unexpected feature store identity")
    if require_complete and not manifest["completed_requested_extraction"]:
        raise ValueError("Requested FinBERT corpus has unscored calls")
    for name, value in manifest["parquet_sha256"].items():
        if stable_digest(archive / name) != value:
            raise ValueError(f"Feature archive hash mismatch: {name}")
    features = pd.read_parquet(archive / "call_features.parquet")
    if not features.transcriptid.is_unique or not set(FEATURES).issubset(features.columns):
        raise ValueError("Invalid call-feature schema")
    expected = features[[f"fb_n_sent_{name}" for name in SECTIONS]].sum().sum()
    if pq.ParquetFile(archive / "sentence_scores.parquet").metadata.num_rows != expected:
        raise ValueError("Sentence and call counts disagree")
    return features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=DEFAULT_MODEL_TABLE)
    parser.add_argument("--sector", action="append")
    parser.add_argument("--store-name", help="Alternate subdirectory under the ignored FinBERT directory")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--checkpoint-calls", type=int, default=8)
    parser.add_argument("--max-calls", type=int, help="Limit new calls. Zero only republishes saved checkpoints.")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    archive = local_path(STORE / args.store_name) if args.store_name else STORE
    if args.verify_only:
        features = verify(archive, require_complete=False)
        print(f"Verified {len(features)} calls and {len(FEATURES)} features without inference.")
        return
    if min(args.threads, args.batch_size, args.checkpoint_calls) < 1 or (
            args.max_calls is not None and args.max_calls < 0):
        parser.error("Batch and thread limits must be positive, and max-calls must be nonnegative")
    archive.mkdir(parents=True, exist_ok=True)
    for flag in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY", "DO_NOT_TRACK"):
        os.environ[flag] = "1"
    events = events_from(args.events, args.sector or ["Health Care"])
    metadata = {int(event.transcriptid): event for event in events.itertuples(index=False)}
    db_path = archive / "inference.sqlite3"
    with writer_lock(archive), closing(sqlite3.connect(db_path, timeout=45)) as db:
        setup(db)
        cached = dict(db.execute("SELECT transcriptid, source_sha FROM calls"))
        if args.audit_only:
            calls = {call.transcriptid: Coverage.from_call(call) for call in source_calls(events)}
            coverage = audit(events, calls, cached)
            print(coverage.to_string(index=False), flush=True)
            (archive / "coverage_audit.csv").write_text(coverage.to_csv(index=False), encoding="utf-8")
            return
        identity = bind_cache(db, STORE)
        engine = None

        def make_scorer():
            nonlocal engine
            if engine is None:
                sys.path.append(str(STORE / "runtime"))
                from finbert_runtime import OfflineFinBERT
                engine = OfflineFinBERT(STORE, args.device, args.threads)
                print(f"Offline inference: OpenVINO {engine.version}, {engine.device}, "
                      f"available devices {engine.available_devices}", flush=True)
                settings = {"device": engine.device, "available_devices": engine.available_devices,
                            "engine_version": engine.version, "threads": args.threads,
                            "batch_size": args.batch_size, "precision": "f32"}
                with db:
                    db.execute("INSERT OR IGNORE INTO settings VALUES (?, ?)",
                               ("run:" + json.dumps(settings, sort_keys=True), ""))
            return engine

        clock = perf_counter()
        total, processed, skipped, pending_count = 0, 0, 0, 0
        calls, group = {}, []
        print(f"Checking {len(events)} requested calls against {len(cached)} cached calls.", flush=True)

        def checkpoint():
            nonlocal total, processed
            total += process(db, group, make_scorer, args.batch_size)
            processed += len(group)
            group.clear()
            elapsed = perf_counter() - clock
            print(f"Checkpoint {processed} calls; {skipped} reused; {total} new sentences; "
                  f"{elapsed:.0f}s; {total / max(elapsed, 1):.1f} unique sentences/s.", flush=True)
        for call in source_calls(events):
            tid = call.transcriptid
            calls[tid] = Coverage.from_call(call)
            event = metadata[tid]
            if cached.get(tid) == call.source_sha:
                skipped += 1
                with db:
                    db.execute("""UPDATE calls SET permno=?,clock=?,year=?,sector=?,source=?
                                  WHERE transcriptid=?""",
                               (int(event.permno) if pd.notna(event.permno) else None,
                                event.clock.isoformat() if pd.notna(event.clock) else None,
                                int(event.year), event.sector, call.source, tid))
                continue
            pending_count += 1
            if args.max_calls is not None and processed + len(group) >= args.max_calls:
                continue
            group.append((call, event))
            if len(group) == args.checkpoint_calls:
                checkpoint()
        if group:
            checkpoint()
        features = publish(db, archive)
        coverage = audit(events, calls, dict(db.execute("SELECT transcriptid,source_sha FROM calls")))
        universe_sha = stable_digest(args.events)
        contracts = store_coverage(db, archive, events, universe_sha, coverage)
        print(coverage.to_string(index=False), flush=True)
        runs = [json.loads(key[4:]) for key, in db.execute(
            "SELECT key FROM settings WHERE key LIKE 'run:%' ORDER BY key")]
        source_hashes = {name: stable_digest(ROOT / name) for name in sorted(features.source_path.unique())}
        names = ("call_features.parquet", "sentence_scores.parquet")
        manifest = {
            **identity, "model_id": "ProsusAI/finbert", "source_reference": REFERENCE,
            "parquet_sha256": {name: stable_digest(archive / name) for name in names},
            "source_shard_sha256": source_hashes, "event_universe_sha256": universe_sha,
            "validated_cohorts": contracts,
            "event_universe": source_name(args.events) if args.events.resolve().is_relative_to(ROOT)
            else "user-supplied local event manifest",
            "identifiers": list(KEYS), "definitions": feature_dictionary(), "execution_settings": runs,
            "tokenizer": "BERT WordPiece uncased, original vocabulary, CLS/SEP, right truncation to 512 tokens",
            "segmentation": "Split at . ! ? followed by whitespace, minimum 10 characters, cap at 512 characters",
            "section_mapping": "CIQ 2 prepared, 3 analyst question (excluded), 4 answer, original ANSWER block parser",
            "missing_rules": "Missing section has zero sentences and null sentiment. No earlier valid mean gives null change.",
            "completed_requested_extraction": processed == pending_count,
            "archived_calls": len(features),
            "sentence_rows": pq.ParquetFile(archive / "sentence_scores.parquet").metadata.num_rows,
        }
        json_write(archive / "feature_manifest.json", manifest)
        verify(archive, require_complete=False)
        print(f"Verified local archive: {manifest['archived_calls']} calls, "
              f"{manifest['sentence_rows']} sentence rows. New inference: {total}.", flush=True)


if __name__ == "__main__":
    main()
