import pandas as pd

from linguistic_alpha.data import transcripts


def raw_components():
    # IR hand-off tagged as Answer before any question, a 2-turn answer, an operator break
    rows = [
        (1, 1, 1, 1, "Good day and welcome."),
        (1, 2, 2, 10, "Thanks. Revenue grew."),
        (1, 3, 4, 11, "Thank you, Tim. [Operator Instructions]"),
        (1, 4, 3, 20, "How is margin?"),
        (1, 5, 4, 10, "Margin is fine."),
        (1, 6, 4, 12, "And costs fell."),
        (1, 7, 1, 1, "Next question."),
        (1, 8, 3, 21, "What about China?"),
        (1, 9, 4, 10, "China is strong."),
        (1, 10, 3, 22, "Unanswered last question?"),
    ]
    return pd.DataFrame(rows, columns=["transcriptid", "componentorder",
                                       "transcriptcomponenttypeid", "transcriptpersonid",
                                       "componenttext"])


def test_build_components_sections_and_roles():
    persons = pd.DataFrame({"transcriptid": [1, 1, 1], "transcriptpersonid": [10, 20, 1],
                            "speakertypename": ["Executives", "Analysts", "Operator"]})
    c = transcripts.build_components(raw_components(), persons)
    assert list(c["section"][:4]) == ["operator", "prepared", "answer", "question"]
    assert c.loc[c["speaker_id"] == 10, "speaker_role"].eq("executive").all()
    assert c.loc[c["speaker_id"] == 20, "speaker_role"].eq("analyst").all()
    assert c.loc[c["speaker_id"] == 12, "speaker_role"].eq("unknown").all()


def test_pair_qa_groups_answer_turns():
    pairs = transcripts.pair_qa(transcripts.build_components(raw_components()))
    assert len(pairs) == 3
    first = pairs.iloc[0]
    assert first["question"] == "How is margin?"
    assert first["n_answer_turns"] == 2
    assert "costs fell" in first["answer"]
    assert pairs.iloc[1]["answer"] == "China is strong."
    assert pairs.iloc[2]["n_answer_turns"] == 0
    assert "Operator Instructions" not in " ".join(pairs["answer"])


def test_call_sections_counts():
    s = transcripts.call_sections(transcripts.build_components(raw_components())).iloc[0]
    assert s["n_questions"] == 3
    assert s["n_answers"] == 4
    assert s["words_prepared"] == 3


def test_select_latest_version_uses_creation_time_not_id():
    detail = pd.DataFrame({
        "keydevid": [7, 7], "transcriptid": [900, 100],
        "transcriptcreationdate_utc": ["2010-01-01", "2010-02-01"],
    })
    assert transcripts.select_latest_version(detail)["transcriptid"].tolist() == [100]


def test_legacy_panel_round_trip():
    panel = pd.DataFrame({
        "transcriptid": [5],
        "prepared_text": ["Intro remarks.\n\nCFO remarks."],
        "qa_text": [("ANSWER: stray hand-off\n\nQUESTION: Q one?\n\nANSWER: A one."
                    "\n\nANSWER: A two.\n\nQUESTION: Q two?\n\nANSWER: A three.")],
    })
    c = transcripts.components_from_legacy_panel(panel)
    assert (c["section"] == "prepared").sum() == 2
    pairs = transcripts.pair_qa(c)
    assert pairs["n_answer_turns"].tolist() == [2, 1]
