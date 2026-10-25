"""Generate reference fixtures for the lldx-engine gates with Hugging Face Transformers.

Everything runs on CPU in float32 with deterministic algorithms, so the same pinned
library versions (uv.lock) produce byte-identical files.

    uv run --project fixtures --locked python fixtures/generate.py --tiny
    uv run --project fixtures --locked python fixtures/generate.py --model <dir>

--tiny        Random-weight SmolLM3 (4 layers, one of them NoPE), written to fixtures/tiny/.
--model DIR   Real SmolLM3-3B at the pinned revision, written to fixtures/out/smollm3-3b/.

Both write reference.safetensors (input ids and activations captured with forward hooks)
and record sha256 hashes plus library versions in fixtures/manifest.json.
"""

import argparse
import hashlib
import json
import platform
import shutil
import sys
from pathlib import Path

import safetensors
import safetensors.torch
import torch
import transformers
from transformers import AutoTokenizer, SmolLM3Config, SmolLM3ForCausalLM

FIXTURES = Path(__file__).resolve().parent
MANIFEST = FIXTURES / "manifest.json"

SEED = 0
BATCH = 2
TINY_SEQ_LEN = 8

TINY_CONFIG = {
    "vocab_size": 512,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "max_position_embeddings": 128,
    "tie_word_embeddings": True,
    # SmolLM3-3B's rope_theta. no_rope_layers is derived (interval 4), so layer 3 is NoPE.
    "rope_parameters": {"rope_type": "default", "rope_theta": 5_000_000.0},
    # The default special-token ids (128000+) are outside a 512-token vocab, and the tiny model
    # has no tokenizer.
    "bos_token_id": None,
    "eos_token_id": None,
    "pad_token_id": None,
}

REAL_REPO = "HuggingFaceTB/SmolLM3-3B"
REAL_REVISION = "a07cc9a04f16550a088caea529712d1d335b0ac1"
# Hashes of the files at REAL_REVISION, from the Hugging Face tree API: sha256 for LFS files,
# git blob sha1 for the rest.
REAL_FILES = {
    "config.json": ("git-sha1", "89ad848b8bbebe0ae085847b7b2e010a4aa1b408"),
    "model.safetensors.index.json": ("git-sha1", "ee6c028c012814d385be51c1d537c345c9fa9457"),
    "model-00001-of-00002.safetensors": (
        "sha256",
        "e5eed4113a925264c33e3cdf76cd9f547c06a207ef2270914a7274a2bf685ffd",
    ),
    "model-00002-of-00002.safetensors": (
        "sha256",
        "6c3ad90646457295723e4da5ee8afac47006e20407b122df5174693dd8d68a43",
    ),
    "tokenizer.json": ("sha256", "7b6a500b662a34eb3f0374db856ba4ad7de4c81040571d78dc0d357238930005"),
    "tokenizer_config.json": ("git-sha1", "61910c2db5cbdc9e6a6f37e14aaf00584cc6ad47"),
    "special_tokens_map.json": ("git-sha1", "190d5624dbbc1ad56f2f34c9d58e03fef7e5328b"),
    "chat_template.jinja": ("git-sha1", "e01e3a1bca00ae47bca8326b38cc397729f87481"),
}


def configure_determinism() -> None:
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    # One thread fixes the reduction order inside CPU kernels.
    torch.set_num_threads(1)


def file_digest(path: Path, kind: str) -> str:
    if kind == "sha256":
        digest = hashlib.sha256()
    else:
        digest = hashlib.sha1()
        digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def capture(model: SmolLM3ForCausalLM, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
    """Runs one forward pass and returns the input ids plus every captured activation."""
    captured: dict[str, torch.Tensor] = {}

    def keep(name: str, take_first: bool = False):
        def hook(_module, _inputs, output):
            captured[name] = (output[0] if take_first else output).detach().clone()

        return hook

    inner = model.model
    handles = [inner.embed_tokens.register_forward_hook(keep("embeddings"))]
    for i, layer in enumerate(inner.layers):
        # Attention returns (output, weights). The layer and MLP return a tensor.
        handles.append(layer.self_attn.register_forward_hook(keep(f"layers.{i}.attention", True)))
        handles.append(layer.mlp.register_forward_hook(keep(f"layers.{i}.mlp")))
        handles.append(layer.register_forward_hook(keep(f"layers.{i}.hidden")))

    with torch.no_grad():
        logits = model(input_ids=input_ids, use_cache=False).logits
    for handle in handles:
        handle.remove()

    expected = 1 + 3 * len(inner.layers)
    if len(captured) != expected:
        sys.exit(f"captured {len(captured)} activations, expected {expected}")
    return {"input_ids": input_ids, **captured, "logits": logits.detach().clone()}


def save_reference(tensors: dict[str, torch.Tensor], path: Path) -> None:
    contiguous = {name: t.contiguous() for name, t in tensors.items()}
    safetensors.torch.save_file(contiguous, path, metadata={"format": "pt"})


def generate_tiny() -> tuple[str, dict, list[Path]]:
    configure_determinism()
    config = SmolLM3Config(**TINY_CONFIG)
    config._attn_implementation = "eager"
    if 0 not in config.no_rope_layers:
        sys.exit(f"tiny config has no NoPE layer: no_rope_layers={config.no_rope_layers}")

    model = SmolLM3ForCausalLM(config).to(torch.float32).eval()
    generator = torch.Generator().manual_seed(SEED)
    input_ids = torch.randint(0, config.vocab_size, (BATCH, TINY_SEQ_LEN), generator=generator)

    out = FIXTURES / "tiny"
    out.mkdir(exist_ok=True)
    # Same fields save_pretrained() writes, without the generation_config.json it adds.
    config.architectures = ["SmolLM3ForCausalLM"]
    config.dtype = torch.float32
    config_path = out / "config.json"
    config_path.write_text(config.to_json_string())

    # Tied embeddings: like the real checkpoint, lm_head.weight is not stored.
    state = {k: v.contiguous() for k, v in model.state_dict().items() if k != "lm_head.weight"}
    weights_path = out / "model.safetensors"
    safetensors.torch.save_file(state, weights_path, metadata={"format": "pt"})

    reference_path = out / "reference.safetensors"
    save_reference(capture(model, input_ids), reference_path)

    meta = {
        "description": "Random-weight SmolLM3, seeded, for tier-2 gates",
        "model_repo": None,
        "revision": None,
        "seed": SEED,
        "no_rope_layers": config.no_rope_layers,
    }
    return "tiny", meta, [config_path, weights_path, reference_path]


def generate_real(model_dir: Path) -> tuple[str, dict, list[Path]]:
    for name, (kind, want) in REAL_FILES.items():
        path = model_dir / name
        if not path.is_file():
            sys.exit(f"{path} is missing; download {REAL_REPO} at revision {REAL_REVISION}")
        if (got := file_digest(path, kind)) != want:
            sys.exit(f"{path}: {kind} {got} does not match revision {REAL_REVISION} ({want})")

    configure_determinism()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = SmolLM3ForCausalLM.from_pretrained(
        model_dir, dtype=torch.float32, attn_implementation="eager"
    ).eval()

    prompts = json.loads((FIXTURES / "prompts.json").read_text(encoding="utf-8"))
    by_id = {p["id"]: p["text"] for p in prompts["prompts"]}
    texts = [by_id[i] for i in prompts["reference_batch"]]
    ids = [tokenizer(text)["input_ids"] for text in texts]
    # Equal-length batch, no padding: truncate to the shortest prompt (ADR-04).
    seq_len = min(len(x) for x in ids)
    input_ids = torch.tensor([x[:seq_len] for x in ids], dtype=torch.long)

    out = FIXTURES / "out" / "smollm3-3b"
    out.mkdir(parents=True, exist_ok=True)
    config_path = out / "config.json"
    shutil.copyfile(model_dir / "config.json", config_path)
    reference_path = out / "reference.safetensors"
    save_reference(capture(model, input_ids), reference_path)

    meta = {
        "description": "SmolLM3-3B reference activations for tier-3 gates (weights stay in LLDX_MODEL_DIR)",
        "model_repo": REAL_REPO,
        "revision": REAL_REVISION,
        "seed": SEED,
        "reference_batch": prompts["reference_batch"],
    }
    return "smollm3-3b", meta, [config_path, reference_path]


def update_manifest(key: str, meta: dict, files: list[Path]) -> None:
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    manifest[key] = {
        **meta,
        "libraries": {
            "python": platform.python_version(),
            "safetensors": safetensors.__version__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "files": {
            path.relative_to(FIXTURES).as_posix(): file_digest(path, "sha256") for path in files
        },
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--tiny", action="store_true", help="random-weight SmolLM3 to fixtures/tiny/")
    target.add_argument("--model", type=Path, metavar="DIR", help="SmolLM3-3B directory at the pinned revision")
    args = parser.parse_args()

    key, meta, files = generate_tiny() if args.tiny else generate_real(args.model)
    update_manifest(key, meta, files)
    for path in files:
        print(f"wrote {path.relative_to(FIXTURES)} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
