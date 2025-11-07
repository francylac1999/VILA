"""Convert model checkpoint and optionally split into layer-wise shards.

This script can:
 - load a .safetensors or .pt checkpoint,
 - normalize keys and load into the model skeleton,
 - save a combined .pt file (optional),
 - split the state_dict into per-key .pt shard files for layerwise loading.

It's intended to be used when running on memory-constrained devices (e.g. Jetson)
that require the `mem_efficient_load` folder layout used by the loader.
"""

import sys
import argparse
import os
from pathlib import Path

# Prevent a local `logging.py` from shadowing Python's stdlib `logging`
# when importing third-party packages (torch, loguru, etc.).
# We temporarily remove the script directory (and empty cwd entry) from
# sys.path, import the external libs, then restore sys.path so local
# imports continue to work.
_script_dir = os.path.dirname(os.path.abspath(__file__))
_orig_sys_path = list(sys.path)
sys.path = [p for p in sys.path if p and os.path.abspath(p) != _script_dir]
try:
    import torch
    from safetensors.torch import load_file as safe_load
    from transformers import AutoConfig
    from tqdm import tqdm
finally:
    sys.path = _orig_sys_path

from tinychat.models.nvila_qwen2 import NVILAQwen2


def normalize_state_dict_keys(state_dict: dict) -> dict:
    """Normalize common prefixes from safetensors keys.
    Strips leading 'module.' if present which often appears
    when checkpoints were saved from DataParallel or wrapped models.
    """
    new_state = {}
    for k, v in state_dict.items():
        new_k = k
        if new_k.startswith("module."):
            new_k = new_k[len("module."):]
        new_state[new_k] = v
    return new_state


def load_checkpoint_file(path: Path) -> dict:
    """Load a checkpoint file (.safetensors or .pt) and return a state_dict mapping."""
    p = str(path)
    if p.endswith(".safetensors"):
        sd = safe_load(p)
        return dict(sd)
    else:
        data = torch.load(p, map_location="cpu")
        if isinstance(data, dict):
            for candidate in ("state_dict", "model_state_dict", "model", "state"):
                if candidate in data and isinstance(data[candidate], dict):
                    return data[candidate]
            # Assume it's already a state dict
            return data
        raise RuntimeError("Unsupported checkpoint format: expected dict-like checkpoint")


def split_state_dict_to_shards(state_dict: dict, outdir: Path, overwrite: bool = False):
    # This simple splitter writes files named exactly as the desired key + '.pt'.
    outdir.mkdir(parents=True, exist_ok=True)
    keys = list(state_dict.keys())
    print(f"Saving {len(keys)} shard files (from raw keys) to {outdir}")
    for k in tqdm(keys, desc="Saving shards"):
        v = state_dict[k]
        try:
            if hasattr(v, "cpu"):
                v = v.cpu()
        except Exception:
            pass
        filename = k + ".pt"
        outpath = outdir / filename
        if outpath.exists() and not overwrite:
            tqdm.write(f"Skipping existing: {outpath}")
            continue
        torch.save(v, str(outpath))


def convert_and_split(input_path: Path, output_pt: Path, config_path: Path, split_outdir: Path = None, overwrite: bool = False):
    input_path = input_path.expanduser().resolve()
    output_pt = output_pt.expanduser().resolve() if output_pt else None
    config_path = config_path.expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input checkpoint not found: {input_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config path not found: {config_path}")

    # Load raw state dict from checkpoint
    print(f"Loading checkpoint from {input_path}...")
    raw_state = load_checkpoint_file(input_path)
    raw_state = {k: (v.cpu() if hasattr(v, "cpu") else v) for k, v in raw_state.items()}
    raw_state = normalize_state_dict_keys(raw_state)

    # Instantiate model skeleton and load weights permissively
    config = AutoConfig.from_pretrained(str(config_path))
    model = NVILAQwen2(config, True)
    print("Loading state dict into model (strict=False)...")
    load_result = model.load_state_dict(raw_state, strict=False)

    # Save combined .pt if requested
    if output_pt:
        output_pt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), str(output_pt))
        print(f"Saved combined .pt to {output_pt}")

    # Split into shards if requested
    if split_outdir:
        # We want the shard filenames to match the exact keys expected by the
        # LLM loader (often keys include a 'model.' prefix or 'lm_head.weight').
        # Build desired keys from the model's LLM part when available.
        desired_keys = None
        if hasattr(model, "llm"):
            desired_keys = list(model.llm.state_dict().keys())
        else:
            desired_keys = list(model.state_dict().keys())

        # For each desired key, find the best matching key in raw_state and save
        # under the desired key name (so loader finds exact filenames).
        outdir = Path(split_outdir).expanduser().resolve()
        outdir.mkdir(parents=True, exist_ok=True)
        raw_keys = set(raw_state.keys())
        created = 0
        skipped = 0
        for dk in tqdm(desired_keys, desc="Preparing shard files to match model keys"):
            # candidate matching order
            candidates = [dk, dk.replace("model.", ""), dk.replace("llm.", ""), dk.replace("llm.model.", "")]
            found = None
            for c in candidates:
                if c in raw_state:
                    found = c
                    break
            # special case: map lm_head.weight -> embed_tokens.weight
            if found is None and dk == "lm_head.weight":
                if "embed_tokens.weight" in raw_state:
                    found = "embed_tokens.weight"
            if found is None:
                # not found: try by suffix match (last part after dot)
                suffix = dk.split('.', 1)[-1]
                for rk in raw_keys:
                    if rk.endswith(suffix):
                        found = rk
                        break
            if found is None:
                tqdm.write(f"WARNING: no source key found for desired key '{dk}'")
                continue

            outpath = outdir / (dk + ".pt")
            if outpath.exists() and not overwrite:
                skipped += 1
                continue
            torch.save(raw_state[found], str(outpath))
            created += 1
        print(f"Shard files created: {created}; skipped (existing): {skipped}")

    # Print load summary
    missing = len(load_result.missing_keys) if hasattr(load_result, "missing_keys") else None
    unexpected = len(load_result.unexpected_keys) if hasattr(load_result, "unexpected_keys") else None
    if missing is not None or unexpected is not None:
        print("Load summary:")
        if missing:
            print(f"  Missing keys: {missing} (first 10): {load_result.missing_keys[:10]}")
        else:
            print("  Missing keys: 0")
        if unexpected:
            print(f"  Unexpected keys: {unexpected} (first 10): {load_result.unexpected_keys[:10]}")
        else:
            print("  Unexpected keys: 0")


def parse_args():
    p = argparse.ArgumentParser(description="Convert checkpoint and optionally split into per-key shard files")
    p.add_argument("--input", "-i", required=True, help="Path to input checkpoint (.safetensors or .pt)")
    p.add_argument("--output-pt", "-o", required=False, default="", help="Optional combined output .pt path")
    p.add_argument("--config", "-c", required=False, default="/home/workspace/NVILA-Lite-2B/", help="Path to model config for AutoConfig.from_pretrained")
    p.add_argument("--split-outdir", "-s", required=False, default="", help="If set, split state_dict into per-key .pt files in this folder")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing shard files when splitting")
    return p.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_pt = Path(args.output_pt) if args.output_pt else None
    split_outdir = Path(args.split_outdir) if args.split_outdir else None
    try:
        convert_and_split(input_path, output_pt, Path(args.config), split_outdir, overwrite=args.overwrite)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()