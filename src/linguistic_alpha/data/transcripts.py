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


def select_earliest_version(detail: pd.DataFrame) -> pd.DataFrame:
    """Keep one transcript per call (keydevid): the earliest created version.

    Robustness only (``transcripts.version_rule: earliest``); the main rule is
    ``select_latest_before``. Creation time only orders versions. Never order by max
    ``transcriptid`` (re-collected transcripts get new, larger ids). Ties fall back to the
    lowest ``transcriptid``; versions without a creation time come last.
    """
    d = detail.copy()
    created = pd.to_datetime(d.get("transcriptcreationdate_utc"), errors="coerce")
    if "transcriptcreationtime_utc" in d:
        created = created + pd.to_timedelta(d["transcriptcreationtime_utc"].astype(str),
                                            errors="coerce").fillna(pd.Timedelta(0))
    d["_created"] = created
    d = d.sort_values(["keydevid", "_created", "transcriptid"], na_position="last")
    return d.groupby("keydevid").head(1).drop(columns="_created").reset_index(drop=True)


def _created_utc(d: pd.DataFrame) -> pd.Series:
    date = pd.to_datetime(d.get("transcriptcreationdate_utc"), errors="coerce")
    if "transcriptcreationtime_utc" in d:
        date = date + pd.to_timedelta(d["transcriptcreationtime_utc"].astype(str),
                                      errors="coerce").fillna(pd.Timedelta(0))
    return date.dt.tz_localize("UTC")


def day0_close_utc(detail: pd.DataFrame, trading_days: pd.DatetimeIndex,
                   market_close_et: str = "16:00",
                   placeholder_unknown: bool = False) -> pd.Series:
    """Close of the call's own session (UTC) per call, i.e. Day 0 taken from the CALL time.

    This is not the release's Day 0. With the release as anchor it equals:
      - the release's Day 0 close when the call falls on Day 0 (call lag 0);
      - the release's Day +1 close, the trade entry, when the call falls on Day +1
        (call lag 1, e.g. release before the open, call after the close or next morning).
    So calls on Day +1 keep a transcript. The bound "never after entry" holds only after the
    call-lag filter (``events.max_call_lag_sessions`` = 1); ``select_latest_before`` enforces
    it when given ``entry_close_utc``. Indexed by ``keydevid``.

    ``placeholder_unknown`` (``events.placeholder_time_last``): CIQ's 00:00:00 UTC placeholder
    is an unknown time (the call date, ``next_session``), as in ``la events``.
    """
    from linguistic_alpha.events.timing import NY, assign_day0, call_timestamp_utc
    calls = detail.drop_duplicates("keydevid").set_index("keydevid")
    time = calls["mostimportanttimeutc"].astype("string")
    if placeholder_unknown:
        time = time.mask(time.str.strip().eq("00:00:00"))
    ts = call_timestamp_utc(calls["mostimportantdateutc"], time)
    d0 = assign_day0(ts, trading_days, call_date_fallback=calls["mostimportantdateutc"],
                     market_close_et=market_close_et)["day0"]
    return (d0 + pd.Timedelta(market_close_et + ":00")).dt.tz_localize(NY).dt.tz_convert("UTC")


def entry_close_utc(day0: pd.Series, trading_days: pd.DatetimeIndex,
                    market_close_et: str = "16:00") -> pd.Series:
    """Trade entry (UTC) per event: the close of the session after the release's Day 0."""
    from linguistic_alpha.events.timing import NY
    days = pd.DatetimeIndex(trading_days).normalize().sort_values().unique()
    pos = days.searchsorted(pd.to_datetime(day0).dt.normalize(), side="left") + 1
    nxt = pd.to_datetime(pd.Series([days[p] if p < len(days) else pd.NaT for p in pos],
                                   index=day0.index))
    return (nxt + pd.Timedelta(market_close_et + ":00")).dt.tz_localize(NY).dt.tz_convert("UTC")


def select_latest_before(detail: pd.DataFrame, cutoff_utc: pd.Series,
                         no_version: str = "drop",
                         entry_close_utc: pd.Series | None = None) -> pd.DataFrame:
    """Keep one transcript per call: the LATEST version created at or before the cutoff.

    Spec (docs/pipeline.md, "Transcript version and timing"): the cutoff is the close of the
    call's own session (``day0_close_utc``): the release's Day 0 close for calls on Day 0,
    the Day +1 close (entry) for calls on Day +1. ``entry_close_utc`` (release Day +1 close,
    per ``keydevid``, from :func:`entry_close_utc`) caps the cutoff, so no text created after
    the position opens is ever used, even if a call time or the lag filter is wrong; capped
    calls are counted in a warning. Versions are ordered by creation time, never by
    ``transcriptid``; ties take the highest ``transcriptid`` among equal times.
    ``no_version``: "drop" leaves out calls with no version by the cutoff (main sample);
    "earliest" keeps their earliest version, flagged ``version_before_cutoff`` = False.
    Adds ``version_created_utc``, ``version_type`` (presentation / collection),
    ``version_before_cutoff`` and ``version_final``.

    ``version_final`` is the single text-quality control: 1 when the chosen version has
    reached the Final stage (Edited or later: Edited, Proofed, Audited and their SA / EV
    variants), 0 when it is still Preliminary (Spellchecked). Capital IQ ties the two fields
    together: Preliminary versions are always Spellchecked and Final versions never are.
    """
    if entry_close_utc is not None:
        entry = entry_close_utc.reindex(cutoff_utc.index)
        late = cutoff_utc.gt(entry)
        if late.any():
            warnings.warn(f"{int(late.sum())} calls had a transcript cutoff after the trade "
                          "entry (call after Day +1?); capped at the Day +1 close", stacklevel=2)
        cutoff_utc = cutoff_utc.where(~late, entry)
    d = detail.copy()
    d["version_created_utc"] = _created_utc(d)
    d["_cut"] = d["keydevid"].map(cutoff_utc)
    d["version_before_cutoff"] = d["version_created_utc"].le(d["_cut"])
    if {"transcriptpresentationtypename", "transcriptcollectiontypename"} <= set(d.columns):
        d["version_type"] = (d["transcriptpresentationtypename"].astype(str) + " / "
                             + d["transcriptcollectiontypename"].astype(str))
        d["version_final"] = d["transcriptpresentationtypename"].eq("Final").astype(int)
    ok = d[d["version_before_cutoff"]].sort_values(["keydevid", "version_created_utc", "transcriptid"])
    out = ok.groupby("keydevid").tail(1)
    if no_version == "earliest":
        rest = d[~d["keydevid"].isin(out["keydevid"])]
        rest = rest.sort_values(["keydevid", "version_created_utc", "transcriptid"],
                                na_position="last").groupby("keydevid").head(1)
        out = pd.concat([out, rest])
    elif no_version != "drop":
        raise ValueError(no_version)
    return out.drop(columns="_cut").sort_values("keydevid").reset_index(drop=True)


def annotate_versions(detail: pd.DataFrame) -> pd.DataFrame:
    """All versions + ``version_created_utc``, ``version_type`` and ``version_final`` (parse pass:
    ``interim/transcript_versions``)."""
    d = detail.copy()
    d["version_created_utc"] = _created_utc(d)
    if {"transcriptpresentationtypename", "transcriptcollectiontypename"} <= set(d.columns):
        d["version_type"] = (d["transcriptpresentationtypename"].astype(str) + " / "
                             + d["transcriptcollectiontypename"].astype(str))
        d["version_final"] = d["transcriptpresentationtypename"].eq("Final").astype(int)
    return d


def select_versions(versions: pd.DataFrame, trading_days: pd.DatetimeIndex, cfg: dict,
                    release_day0: pd.Series | None = None) -> pd.DataFrame:
    """One version per keydevid under ``transcripts.version_rule`` (shared by ``la events`` and
    the select pass, so both pick the same transcript).

    ``release_day0``: the release's Day 0 per keydevid where ``la events`` found one; its Day +1
    close (``entry_close_utc``) caps the cutoff. Calls without one are not capped (the cap binds
    only when the call falls 2+ sessions after the release, and such events are excluded).
    """
    tcfg, ecfg = cfg["transcripts"], cfg["events"]
    close = ecfg["market_close_et"]
    if tcfg["version_rule"] == "earliest":
        return select_earliest_version(versions)
    if tcfg["version_rule"] != "latest_before_day0_close":
        raise ValueError(tcfg["version_rule"])
    cutoff = day0_close_utc(versions, trading_days, close,
                            placeholder_unknown=bool(ecfg.get("placeholder_time_last", False)))
    entry = None
    if release_day0 is not None:
        r = pd.to_datetime(release_day0).dropna()
        entry = entry_close_utc(r, trading_days, close)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = select_latest_before(versions, cutoff, tcfg["no_version_before_cutoff"],
                                   entry_close_utc=entry)
    for w in caught:
        print(f"select_versions: {w.message}")
    return out


def speaker_role(speakertypename: pd.Series) -> pd.Series:
    name = speakertypename.fillna("").str.lower()
    role = pd.Series("unknown", index=name.index, dtype="string")
    for pattern, label in ROLE_PATTERNS:
        role = role.mask(name.str.contains(pattern), label)
    return role


def build_components(raw: pd.DataFrame, persons: pd.DataFrame | None = None) -> pd.DataFrame:
    """Raw CIQ components (+ optional speaker table) -> canonical ``COMPONENTS`` table.

    One output row per raw component. The WRDS speaker table (``ciq_transcript_person``) has
    one row per component, not per speaker, so it is reduced to one row per
    ``(transcriptid, transcriptpersonid)`` before the merge; otherwise every component is
    repeated once per turn its speaker has.
    """
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
        p["transcriptid"] = p["transcriptid"].astype("int64")
        p["speaker_id"] = pd.to_numeric(p.pop("transcriptpersonid"),
                                        errors="coerce").astype("Int64")
        p = p.dropna(subset=["speaker_id"])
        n_types = p.groupby(["transcriptid", "speaker_id"])["speakertypename"].nunique()
        if (n_types > 1).any():
            warnings.warn(f"{int((n_types > 1).sum())} speakers have more than one "
                          "speakertypename; keeping the first", stacklevel=2)
        p = p.drop_duplicates(["transcriptid", "speaker_id"])
        out = out.merge(p, on=["transcriptid", "speaker_id"], how="left",
                        validate="many_to_one")
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
