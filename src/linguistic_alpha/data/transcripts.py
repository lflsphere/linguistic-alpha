"""Capital IQ transcripts: version selection, canonical components, Q&A pairing.

Canonical input is the long CIQ component table (one row per speaker turn), i.e.
``ciq.ciqtranscriptcomponent`` joined to ``ciq.wrds_transcript_person``. The legacy
``panel.xlsx`` (flattened prepared_text / qa_text) is supported through
:func:`components_from_legacy_panel` so the downstream code runs before the full pull.
"""

from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd

from linguistic_alpha.schemas import COMPONENTS, QA_PAIRS, validate

# ciq.ciqtranscriptcomponenttype. 1-4 are confirmed in our data; check the remaining ids
# against the type table before the full run (anything unmapped becomes "other").
SECTION_BY_TYPE = {1: "operator", 2: "prepared", 3: "question", 4: "answer"}

ROLE_PATTERNS = [("operator", "operator"), ("analyst", "analyst"), ("executive", "executive")]


def select_latest_version(detail: pd.DataFrame) -> pd.DataFrame:
    """Keep one transcript per call (keydevid): the most recently created version.

    The legacy notebook took the max ``transcriptid``, which is not a version order
    (re-collected transcripts get new, larger ids). Ties fall back to ``transcriptid``.
    TODO: rank by transcriptpresentationtypeid (final > edited > preliminary) once the
    type table is pulled, and decide whether to mix redacted/unredacted collections.
    """
    d = detail.copy()
    created = pd.to_datetime(d.get("transcriptcreationdate_utc"), errors="coerce")
    if "transcriptcreationtime_utc" in d:
        created = created + pd.to_timedelta(d["transcriptcreationtime_utc"].astype(str),
                                            errors="coerce").fillna(pd.Timedelta(0))
    d["_created"] = created
    d = d.sort_values(["keydevid", "_created", "transcriptid"], na_position="first")
    return d.groupby("keydevid").tail(1).drop(columns="_created").reset_index(drop=True)


def speaker_role(speakertypename: pd.Series) -> pd.Series:
    name = speakertypename.fillna("").str.lower()
    role = pd.Series("unknown", index=name.index, dtype="string")
    for pattern, label in ROLE_PATTERNS:
        role = role.mask(name.str.contains(pattern), label)
    return role


def build_components(raw: pd.DataFrame, persons: pd.DataFrame | None = None) -> pd.DataFrame:
    """Raw CIQ components (+ optional speaker table) -> canonical ``COMPONENTS`` table."""
    out = pd.DataFrame({
        "transcriptid": raw["transcriptid"].astype("int64"),
        "componentorder": raw["componentorder"].astype("int64"),
        "section": raw["transcriptcomponenttypeid"].map(SECTION_BY_TYPE)
                                                   .fillna("other").astype("string"),
        "speaker_id": pd.to_numeric(raw["transcriptpersonid"], errors="coerce").astype("Int64"),
        "text": raw["componenttext"].fillna("").astype("string"),
    })
    if persons is not None:
        p = persons[["transcriptid", "transcriptpersonid", "speakertypename"]].copy()
        p["speaker_id"] = p.pop("transcriptpersonid").astype("Int64")
        out = out.merge(p, on=["transcriptid", "speaker_id"], how="left")
        out["speaker_role"] = speaker_role(out.pop("speakertypename"))
    else:
        out["speaker_role"] = pd.Series("unknown", index=out.index, dtype="string")
    out.loc[out["section"] == "operator", "speaker_role"] = "operator"

    out = out.sort_values(["transcriptid", "componentorder"]).reset_index(drop=True)
    validate(out, COMPONENTS, "components")
    return out[list(COMPONENTS)]


def pair_qa(components: pd.DataFrame) -> pd.DataFrame:
    """Group each analyst question with the management answer turns that follow it.

    A pair opens at a ``question`` turn and collects consecutive ``answer`` turns; any other
    section closes it. Answers before the first question (e.g. IR hand-off to the operator,
    sometimes tagged as Answer by CIQ) are dropped.
    """
    c = components[components["section"].isin(["question", "answer", "operator", "other"])]
    c = c.sort_values(["transcriptid", "componentorder"])
    is_q = c["section"].eq("question")
    pair_id = is_q.groupby(c["transcriptid"]).cumsum()
    # break the chain at operator/other so a stray later answer does not attach to an old Q
    broken = (~c["section"].isin(["question", "answer"])).groupby(
        [c["transcriptid"], pair_id]).cumsum() > 0
    keep = (pair_id > 0) & ~broken & c["section"].isin(["question", "answer"])
    c = c[keep].assign(pair_id=pair_id[keep].astype("int64"))

    q = c[c["section"] == "question"].groupby(["transcriptid", "pair_id"]).agg(
        question_speaker_id=("speaker_id", "first"), question=("text", "first"))
    a = c[c["section"] == "answer"].groupby(["transcriptid", "pair_id"]).agg(
        answer=("text", lambda s: "\n\n".join(s)), n_answer_turns=("text", "size"))
    pairs = q.join(a, how="left").reset_index()
    pairs["answer"] = pairs["answer"].fillna("").astype("string")
    pairs["n_answer_turns"] = pairs["n_answer_turns"].fillna(0).astype("int64")
    validate(pairs, QA_PAIRS, "qa_pairs")
    return pairs[list(QA_PAIRS)]


def call_sections(components: pd.DataFrame) -> pd.DataFrame:
    """One row per transcript with section texts and basic length counts."""
    def joined(section: str) -> pd.Series:
        s = components[components["section"] == section]
        return s.groupby("transcriptid")["text"].agg(lambda t: "\n\n".join(t))

    def n_words(text: pd.Series) -> pd.Series:
        return text.str.count(r"\S+").astype("int64")

    out = pd.DataFrame(index=pd.Index(components["transcriptid"].unique(), name="transcriptid"))
    for section in ("prepared", "question", "answer"):
        out[f"{section}_text"] = joined(section)
        out[f"{section}_text"] = out[f"{section}_text"].fillna("").astype("string")
        out[f"words_{section}"] = n_words(out[f"{section}_text"])
    counts = components.groupby(["transcriptid", "section"]).size().unstack(fill_value=0)
    out["n_questions"] = counts.get("question", 0)
    out["n_answers"] = counts.get("answer", 0)
    out["avg_answer_words"] = out["words_answer"] / out["n_answers"].replace(0, np.nan)
    return out.reset_index()


_QA_SPLIT = re.compile(r"(?:^|\n\n)(QUESTION|ANSWER): ")
EXCEL_CELL_LIMIT = 32_767


def components_from_legacy_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Rebuild approximate components from the flattened legacy ``panel.xlsx``.

    Lossy: speaker ids/roles and operator turns are gone, prepared remarks are split on
    blank lines (not the original component boundaries), and Excel cut every text cell at
    32,767 characters, so long calls lose their final Q&A turns.
    """
    truncated = (panel[["prepared_text", "qa_text"]].astype(str)
                 .apply(lambda s: s.str.len()).ge(EXCEL_CELL_LIMIT).any(axis=1))
    if truncated.any():
        warnings.warn(f"{int(truncated.sum())} of {len(panel)} legacy calls hit the Excel "
                      f"cell limit ({EXCEL_CELL_LIMIT} chars); their text is truncated.",
                      stacklevel=2)
    rows = []
    for tid, prepared, qa in panel[["transcriptid", "prepared_text", "qa_text"]].itertuples(
            index=False):
        order = 0
        for chunk in str(prepared or "").split("\n\n"):
            if chunk.strip():
                order += 1
                rows.append((tid, order, 2, None, chunk.strip()))
        parts = _QA_SPLIT.split(str(qa or ""))
        # split() yields [preamble, TAG, text, TAG, text, ...]
        for tag, text in zip(parts[1::2], parts[2::2]):
            order += 1
            rows.append((tid, order, 3 if tag == "QUESTION" else 4, None, text.strip()))
    raw = pd.DataFrame(rows, columns=["transcriptid", "componentorder",
                                      "transcriptcomponenttypeid", "transcriptpersonid",
                                      "componenttext"])
    return build_components(raw)
