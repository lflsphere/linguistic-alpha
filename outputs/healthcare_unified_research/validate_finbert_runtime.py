"""Compare the offline OpenVINO scorer against the original Transformers architecture."""

import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np

from extract_finbert import HERE, STORE, json_write, sentences
from finbert_runtime import REVISION, OfflineFinBERT


def cases():
    base = [
        "The company beat forecasts and raised its guidance.",
        "The company posted disappointing results and further losses.",
        "The annual meeting is scheduled for May.",
        "Third-quarter margins improved, although unit sales fell year over year.",
        "Diversification and strong free cash flow offset a provision for impairment.",
        "Cash, net of taxes, is approximately $2.9 billion. Quarterly EPS fell 2%.",
        "The U.S. business posted a 5.2% increase in adjusted income.",
        "FDA approved the new therapy this morning.",
        "The trials did not meet the primary endpoint.",
        "Revenue increased modestly despite material adverse FX and softer demand.",
        "The 2026 balance-sheet figures reflect a change in reporting policy.",
        "There's no change in our forecast. Next question, please.",
        "NAÏVE projections: earnings lift? Réserves were unchanged!",
        "中" * 520,
    ]
    return [value for record in base for value, _ in sentences(record)]


def reference_file(path):
    sys.path.append(str(STORE / "reference_runtime"))
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch
    import transformers

    texts = cases()
    torch.set_num_threads(2)
    point = STORE / "model" / REVISION
    tokenizer = AutoTokenizer.from_pretrained(point, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(point, local_files_only=True,
                                                                use_safetensors=True,
                                                                low_cpu_mem_usage=True).eval()
    rows = []
    with torch.inference_mode():
        for text in texts:
            model_inputs = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
            rows.append(torch.softmax(model(**model_inputs).logits, dim=1).cpu().numpy()[0])
    reference = np.stack(rows)
    json_write(path, {"scores": reference.tolist(), "ids": tokenizer(texts, truncation=False)["input_ids"],
                      "torch": torch.__version__, "transformers": transformers.__version__})


def compare(path):
    expected = json.loads(path.read_text(encoding="utf-8"))
    sys.path.append(str(STORE / "runtime"))
    texts = cases()
    reference = np.asarray(expected["scores"])
    device = OfflineFinBERT(STORE, "auto", 4)
    actual, lengths = device.score(texts, 8)
    encoded = device.tokenizer.encode_batch(texts)
    if any(encoded_row.ids != original for encoded_row, original in zip(encoded, expected["ids"])):
        raise ValueError("Token IDs differ from the pinned Transformers tokenizer")
    np.testing.assert_allclose(actual, reference, atol=1e-5, rtol=1e-5)
    np.testing.assert_array_equal(actual.argmax(axis=1), reference.argmax(axis=1))
    # A second batch partition checks padding masks and order restoration.
    np.testing.assert_allclose(device.score(texts, 1)[0], actual, atol=1e-5, rtol=1e-5)
    summary = {
        "model": "ProsusAI/finbert", "revision": REVISION, "examples": len(texts),
        "token_limit_examples": int((lengths > 512).sum()), "matching_top_labels": len(texts),
        "max_abs_softmax_difference": float(np.abs(actual - reference).max()),
        "device": device.device, "openvino": device.version,
        "torch": expected["torch"], "transformers": expected["transformers"],
    }
    json_write(HERE / "runtime_validation.json", summary)
    print(summary, flush=True)


def check():
    with TemporaryDirectory(prefix="finbert-runtime-test-") as temp:
        path = Path(temp) / "reference.json"
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--reference", str(path)], check=True)
        compare(path)


if __name__ == "__main__":
    for flag in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY", "DO_NOT_TRACK"):
        os.environ[flag] = "1"
    if len(sys.argv) == 3 and sys.argv[1] == "--reference":
        reference_file(Path(sys.argv[2]))
    elif len(sys.argv) == 3 and sys.argv[1] == "--compare":
        compare(Path(sys.argv[2]))
    else:
        check()
