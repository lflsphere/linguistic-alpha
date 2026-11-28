"""Candidate 1: richer pair features (rule-based, nothing fitted on labels).

Question type (yes/no, number, guidance, why/how, multi-part), number of sub-questions, answer
specificity (numbers, units, money, dates/periods, capitalised names), refusal and deferral cues
(wider than NONANS_RULE), generic talking-point cues, and how much of the question the answer covers.
`pair_features(q, a)` takes one anonymised question and answer; `frame(df)` applies it to a DataFrame.
"""
from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd

WORD = re.compile(r"[a-z][a-z'\-]*")
STOP = None


def _stop():
    global STOP
    if STOP is None:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        STOP = set(ENGLISH_STOP_WORDS) | {"just", "guess", "maybe", "kind", "sort", "lot", "thing", "things",
                                         "think", "really", "know", "question", "questions", "thanks",
                                         "thank", "great", "good", "morning", "afternoon", "quarter",
                                         "year", "company", "yes", "okay", "sure", "talk", "little", "bit"}
    return STOP


def _rx(*parts):
    return re.compile("|".join(parts), re.I)


YESNO_START = _rx(r"(?:^|[.?!]\s+|\b(?:so|and|but|just|i guess|i'm curious|curious)\s*,?\s+)"
                  r"(?:is|are|was|were|do|does|did|can|could|will|would|should|have|has|had|any|is there|"
                  r"are there)\s+(?:you|there|it|that|this|the|we|your|any)\b")
NUMBER_Q = _rx(r"\bhow (?:much|many|big|large)\b", r"\bwhat(?:'s| is| was| are| were)? (?:the )?"
               r"(?:number|percent|percentage|magnitude|size|amount|rate|level|range|split|mix)\b",
               r"\bquantify\b", r"\bsize (?:the|that|it)\b", r"\bballpark\b", r"\bbasis points?\b", r"\bbps\b",
               r"\bin terms of (?:dollars|percent)\b", r"\bany (?:numbers|color on the number)\b")
GUIDE_Q = _rx(r"\bguid(?:e|ance|ing)\b", r"\boutlook\b", r"\bforecast\b", r"\bexpect(?:ation)?s?\b",
              r"\bnext (?:year|quarter)\b", r"\bgoing forward\b", r"\blong[- ]term (?:target|goal)s?\b",
              r"\btarget(?:s|ing)?\b", r"\bfull[- ]year\b", r"\bYEAR\b")
WHY_Q = _rx(r"\bwhy\b", r"\bhow (?:do|did|should|are|will|would|could|can) (?:you|we)\b",
            r"\bwhat(?:'s| is) driving\b", r"\bdrivers?\b", r"\bexplain\b", r"\bunpack\b")
COLOR_Q = _rx(r"\bcolor\b", r"\bthoughts?\b", r"\bcomment on\b", r"\bperspective\b", r"\bhelp us understand\b")
MULTI_Q = _rx(r"\b(?:two|three|a couple(?: of)?|couple|a few|few) (?:quick )?(?:questions|things|parts|"
              r"follow[- ]ups?|items)\b", r"\b(?:first|firstly|second|secondly|third|lastly|finally),",
              r"\b(?:my )?(?:second|follow[- ]up|other|last|next) (?:question|one)\b", r"\band then\b",
              r"\balso,? (?:on|if|could|can)\b", r"\bseparately\b", r"\bunrelated\b")

REFUSE = _rx(
    r"\b(?:we|i)\s+(?:don't|do not|won't|will not|wouldn't|would not|can't|cannot|can not|aren't|are not|"
    r"haven't|have not)\s+(?:\w+\s+){0,3}(?:disclose|comment|provide|give|break|guide|forecast|quantify|"
    r"speculate|share|split|detail|parse|get into|go into|talk about|discuss|update|specify|answer)",
    r"\bnot (?:going|gonna|able|prepared|in a position|ready|willing) to\b",
    r"\b(?:too early|premature|early days)\b", r"\bno comment\b", r"\bprefer not to\b",
    r"\bnot (?:something|a number|a metric|an area) (?:that )?we\b",
    r"\bwe(?:'re| are) not (?:providing|giving|disclosing|guiding|breaking|going)\b",
    r"\b(?:don't|do not) (?:have|know) (?:that|the|those|a|an|exact|specific|precise)\b",
    r"\b(?:as a (?:matter of )?(?:policy|practice))\b", r"\bwe(?:'ve| have) never (?:disclosed|broken|provided)\b",
    r"\bhard to (?:say|tell|predict|quantify|know)\b", r"\bdifficult to (?:say|tell|predict|quantify|know)\b",
    r"\bcan't (?:say|tell|predict|quantify)\b", r"\bwouldn't want to\b", r"\bnot going to get into\b",
    r"\bwon't get into\b", r"\bnot (?:really )?(?:in a position|prepared)\b", r"\bcompetitive reasons\b")
DEFER = _rx(r"\boffline\b", r"\bfollow up\b", r"\bget back to you\b", r"\bcircle back\b",
            r"\binvestor day\b", r"\banalyst day\b", r"\bnext (?:call|quarter's call|earnings call)\b",
            r"\bmore (?:detail|details|color) (?:later|in|at|when|on)\b", r"\bstay tuned\b",
            r"\b(?:we'll|we will) (?:update|give|provide|talk|share|have more)\b",
            r"\bin (?:due|good) (?:course|time)\b", r"\bwhen we (?:report|give|provide|get to)\b",
            r"\b(?:february|january|march) (?:call|guidance)\b", r"\bat the appropriate time\b")
GENERIC = _rx(r"\bas (?:i|we|\w+) (?:said|mentioned|noted|indicated|discussed)\b", r"\bwe(?:'re| are) (?:very )?"
              r"(?:pleased|excited|confident|comfortable|focused|committed)\b", r"\bwe continue to\b",
              r"\bover the long[- ]term\b", r"\blong[- ]term\b", r"\bour strategy\b", r"\bwe remain\b",
              r"\bwe feel good\b", r"\bat the end of the day\b", r"\boverall\b", r"\bin general\b",
              r"\bbroadly\b", r"\bpositioned\b", r"\bopportunit(?:y|ies)\b")
HEDGE = _rx(r"\bi think\b", r"\bi guess\b", r"\bkind of\b", r"\bsort of\b", r"\bprobably\b", r"\bmaybe\b",
            r"\bperhaps\b", r"\bi would say\b", r"\bit depends\b", r"\bwe'll see\b")
DIRECT = _rx(r"^\s*(?:yes|no|yeah|correct|that's (?:right|correct)|absolutely|exactly|sure)\b")
COURTESY = _rx(r"^\s*(?:(?:yes|yeah|sure|so|well|okay|ok),?\s+)?(?:that's a |it's a )?(?:good|great|fair|"
               r"excellent) question\b", r"^\s*thanks?(?: you)?\b")
HANDOFF = _rx(r"\b(?:i'll|let me|i will|i'm going to) (?:let|ask|have|turn (?:it|that) over to|hand)\b")
NUM = re.compile(r"(?<![\w.])\$?\d[\d,]*(?:\.\d+)?%?")
UNIT = _rx(r"\b(?:percent|percentage points?|basis points?|bps|bips)\b", r"%")
MONEY = _rx(r"\$\s?\d", r"\b(?:million|billion|thousand|dollars?|cents?)\b")
PERIOD = _rx(r"\b(?:q[1-4]|first|second|third|fourth) (?:quarter|half)\b", r"\bq[1-4]\b", r"\bh[12]\b",
             r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b",
             r"\bYEAR\b", r"\b(?:fiscal|calendar) (?:year|\d)")
SPELLED = _rx(r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|twenty|thirty|fifty|hundred|half|"
              r"double|triple)\b")
CAP = re.compile(r"(?<=[a-z,;] )[A-Z][a-zA-Z]+")


def _content(t):
    S = _stop()
    return [w for w in WORD.findall(t.lower()) if w not in S and len(w) > 2]


def _n(rx, t):
    return len(rx.findall(t))


def pair_features(q: str, a: str) -> dict:
    q, a = q or "", a or ""
    ql, al = q.lower(), a.lower()
    aw = max(len(a.split()), 1)
    qw = max(len(q.split()), 1)
    per100 = 100.0 / aw
    head = " ".join(a.split()[:40])
    qc, ac = _content(q), _content(a)
    qs, as_ = set(qc), set(ac)
    a_head = set(_content(head))
    qbi = {tuple(qc[i:i + 2]) for i in range(len(qc) - 1)}
    abi = {tuple(ac[i:i + 2]) for i in range(len(ac) - 1)}
    q_num = int(bool(NUMBER_Q.search(q)))
    a_numbers = len(NUM.findall(a))
    a_spec = a_numbers + _n(UNIT, a) + _n(MONEY, a) + _n(PERIOD, a)
    f = dict(
        q_qmarks=q.count("?"),
        q_multi=_n(MULTI_Q, q),
        q_subq=max(q.count("?"), 1) + _n(MULTI_Q, q),
        q_yesno=int(bool(YESNO_START.search(q))),
        q_number=q_num,
        q_guidance=int(bool(GUIDE_Q.search(q))),
        q_why=int(bool(WHY_Q.search(q))),
        q_color=int(bool(COLOR_Q.search(q))),
        a_num_p100=a_numbers * per100,
        a_unit_p100=_n(UNIT, a) * per100,
        a_money_p100=_n(MONEY, a) * per100,
        a_period_p100=_n(PERIOD, a) * per100,
        a_spelled_p100=_n(SPELLED, al) * per100,
        a_cap_p100=len(CAP.findall(a)) * per100,
        a_spec_p100=a_spec * per100,
        a_any_number=int(a_numbers > 0),
        num_q_no_num=int(q_num == 1 and a_numbers == 0),
        a_refuse=_n(REFUSE, al),
        a_refuse_head=int(bool(REFUSE.search(head.lower()))),
        a_defer=_n(DEFER, al),
        a_generic_p100=_n(GENERIC, al) * per100,
        a_hedge_p100=_n(HEDGE, al) * per100,
        a_direct=int(bool(DIRECT.search(al))),
        a_courtesy=int(bool(COURTESY.search(al))),
        a_handoff=int(bool(HANDOFF.search(al))),
        a_we_share=len(re.findall(r"\b(?:we|our|us)\b", al)) / aw,
        a_i_share=len(re.findall(r"\b(?:i|my|me)\b", al)) / aw,
        q_cover=len(qs & as_) / len(qs) if qs else np.nan,
        q_cover_head=len(qs & a_head) / len(qs) if qs else np.nan,
        q_bigram_cover=len(qbi & abi) / len(qbi) if qbi else np.nan,
        log_ratio=math.log((aw + 1) / (qw + 1)),
        a_short=int(aw < 40),
        a_qmarks=a.count("?"),
    )
    return f


def frame(df: pd.DataFrame, q="q_anon", a="a_anon") -> pd.DataFrame:
    return pd.DataFrame([pair_features(x, y) for x, y in zip(df[q], df[a])], index=df.index)


RICH = list(pair_features("How much? And then the second one?", "We don't disclose that. Revenue rose 5%.").keys())
