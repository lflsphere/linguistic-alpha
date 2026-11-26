"""Offline FinBERT inference from the publisher's pinned safetensors weights."""

import os
from pathlib import Path

import numpy as np

REVISION = "db38d3727cbaed87c9aed72df7b3519e2ba5cca1"
MODEL_FILES = ("config.json", "tokenizer_config.json", "special_tokens_map.json",
               "vocab.txt", "model.safetensors")


def build_model(config, weights_path):
    import openvino as ov
    from openvino import opset13 as op
    from safetensors.numpy import load_file

    weights = load_file(str(weights_path))
    ids = op.parameter([-1, -1], np.int32, name="input_ids")
    mask = op.parameter([-1, -1], np.int32, name="attention_mask")
    hidden = config["hidden_size"]
    heads = config["num_attention_heads"]

    def constant(name):
        return op.constant(weights[name])

    def linear(x, name):
        return op.add(op.matmul(x, constant(name + ".weight"), False, True),
                      constant(name + ".bias"))

    def normalization(x, name):
        scaled = op.mvn(x, [-1], True, config["layer_norm_eps"], "inside_sqrt")
        return op.add(op.multiply(scaled, constant(name + ".weight")), constant(name + ".bias"))

    def split_heads(x):
        shaped = op.reshape(x, [0, 0, heads, hidden // heads], True)
        return op.transpose(shaped, [0, 2, 1, 3])

    length = op.gather(op.shape_of(ids, "i32"), 1, 0)
    positions = op.range(np.int32(0), length, np.int32(1), "i32")
    prefix = "bert.embeddings."
    word = op.gather(constant(prefix + "word_embeddings.weight"), ids, 0)
    position = op.gather(constant(prefix + "position_embeddings.weight"), positions, 0)
    type_zero = op.gather(constant(prefix + "token_type_embeddings.weight"), 0, 0)
    x = normalization(op.add(op.add(word, type_zero), position), prefix + "LayerNorm")
    allowed = op.unsqueeze(op.convert(mask, "boolean"), [1, 2])

    for index in range(config["num_hidden_layers"]):
        prefix = f"bert.encoder.layer.{index}."
        q = split_heads(linear(x, prefix + "attention.self.query"))
        k = split_heads(linear(x, prefix + "attention.self.key"))
        v = split_heads(linear(x, prefix + "attention.self.value"))
        context = op.scaled_dot_product_attention(q, k, v, allowed, causal=False)
        joined = op.reshape(op.transpose(context, [0, 2, 1, 3]), [0, 0, hidden], True)
        attention = linear(joined, prefix + "attention.output.dense")
        x = normalization(op.add(x, attention), prefix + "attention.output.LayerNorm")
        middle = op.gelu(linear(x, prefix + "intermediate.dense"), "erf")
        result = linear(middle, prefix + "output.dense")
        x = normalization(op.add(x, result), prefix + "output.LayerNorm")

    pooled = op.tanh(linear(op.gather(x, 0, 1), "bert.pooler.dense"))
    probabilities = op.softmax(linear(pooled, "classifier"), 1)
    return ov.Model([probabilities], [ids, mask], "ProsusAI FinBERT")


class OfflineFinBERT:
    def __init__(self, archive: Path, device, threads):
        import json

        if os.name == "nt":
            local_app = archive / "runtime_settings"
            consent = local_app / "Intel Corporation" / "openvino_telemetry"
            consent.parent.mkdir(parents=True, exist_ok=True)
            consent.write_text("0\n", encoding="ascii")
            previous = os.environ.get("LOCALAPPDATA")
            os.environ["LOCALAPPDATA"] = str(local_app)
            try:
                import openvino as ov
            finally:
                if previous is None:
                    os.environ.pop("LOCALAPPDATA", None)
                else:
                    os.environ["LOCALAPPDATA"] = previous
        else:
            import openvino as ov
        from tokenizers import BertWordPieceTokenizer

        model_dir = archive / "model" / REVISION
        config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        if config["id2label"] != {"0": "positive", "1": "negative", "2": "neutral"}:
            raise ValueError("Unexpected FinBERT label map")
        options = json.loads((model_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
        self.tokenizer = BertWordPieceTokenizer(str(model_dir / "vocab.txt"),
                                                lowercase=options["do_lower_case"])
        core = ov.Core()
        self.available_devices = core.available_devices
        requested = device if device != "auto" else next(
            (name for name in self.available_devices if name.startswith("GPU")), "CPU")
        properties = {"INFERENCE_PRECISION_HINT": "f32"}
        if requested == "CPU":
            properties["INFERENCE_NUM_THREADS"] = threads
        self.compiled = core.compile_model(build_model(config, model_dir / "model.safetensors"),
                                           requested, properties)
        self.device = requested
        self.version = ov.__version__

    def score(self, texts, batch_size):
        """Bucket by length, then restore the original sentence order."""
        encoded = self.tokenizer.encode_batch(texts)
        lengths = np.array([len(value.ids) for value in encoded])
        order = np.argsort(lengths, kind="stable")
        scores = np.empty((len(texts), 3), dtype=np.float32)
        for start in range(0, len(order), batch_size):
            rows = order[start:start + batch_size]
            width = min(int(lengths[rows].max()), 512)
            inputs = np.zeros((len(rows), width), dtype=np.int32)
            mask = np.zeros_like(inputs)
            for index, row in enumerate(rows):
                ids = encoded[row].ids
                if len(ids) > 512:
                    ids = [*ids[:511], ids[-1]]
                inputs[index, :len(ids)] = ids
                mask[index, :len(ids)] = 1
            scores[rows] = self.compiled([inputs, mask])[0]
        if not np.isfinite(scores).all() or not np.allclose(scores.sum(axis=1), 1, atol=2e-6):
            raise ValueError("Invalid FinBERT softmax scores")
        return scores, lengths
