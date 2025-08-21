#!/usr/bin/env python3
import argparse, json, os, re, sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Optional PEFT import (only needed for LoRA checkpoints)
try:
    from peft import PeftModel
    HAS_PEFT = True
except Exception:
    HAS_PEFT = False

CKPT_RE = re.compile(r"checkpoint-(\d+)$")

def find_latest_checkpoint(dirpath: Path) -> Path | None:
    if not dirpath.exists():
        return None
    cks = []
    for p in dirpath.iterdir():
        if p.is_dir():
            m = CKPT_RE.search(p.name)
            if m:
                cks.append((int(m.group(1)), p))
    if not cks:
        return None
    return sorted(cks, key=lambda t: t[0])[-1][1]

def looks_like_peft_checkpoint(ckpt: Path) -> bool:
    # Common PEFT/LoRA marker files
    return (ckpt / "adapter_config.json").exists() or (ckpt / "adapter_model.safetensors").exists()

def main():
    parser = argparse.ArgumentParser(description="Load a GRPO checkpoint and run inference.")
    parser.add_argument("--base_model", required=True, help="Base HF model id/path used for GRPO (e.g., Qwen/Qwen2.5-7B-Instruct).")
    parser.add_argument("--output_dir", required=True, help="GRPOTrainer output_dir that contains checkpoint-*/")
    parser.add_argument("--prompt", default="Write a short haiku about GPUs and kernels.")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--merge_lora", action="store_true", help="If using LoRA, merge adapters into the base weights for faster inference.")
    parser.add_argument("--use_bf16", action="store_true", help="Force bf16 if available.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    latest = find_latest_checkpoint(out_dir)
    if latest is None:
        print(f"[ERROR] No checkpoint-* found under {out_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Using checkpoint: {latest}")

    # Tokenizer: prefer checkpoint tokenizer if present, else base.
    tok_src = latest if (latest / "tokenizer_config.json").exists() else args.base_model
    tokenizer = AutoTokenizer.from_pretrained(tok_src, use_fast=True)
    if tokenizer.pad_token is None:
        # Make a safe default: pad to eos (common for causal LMs)
        tokenizer.pad_token = tokenizer.eos_token

    # Choose dtype/device map
    dtype = torch.bfloat16 if (args.use_bf16 and torch.cuda.is_available()) else torch.float16 if torch.cuda.is_available() else torch.float32
    device_map = "auto" if torch.cuda.is_available() else None

    # Load model
    if looks_like_peft_checkpoint(latest):
        if not HAS_PEFT:
            print("[ERROR] This looks like a LoRA/PEFT checkpoint but 'peft' is not installed. Try `pip install peft`.", file=sys.stderr)
            sys.exit(1)

        print("[INFO] Detected PEFT (LoRA) checkpoint.")
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=dtype,
            device_map=device_map,
        )
        model = PeftModel.from_pretrained(
            base,
            latest,
            torch_dtype=dtype,
        )
        if args.merge_lora:
            print("[INFO] Merging LoRA adapters into the base model (merge_and_unload).")
            model = model.merge_and_unload()
    else:
        print("[INFO] Loading as a full-model checkpoint.")
        # If GRPO saved a full model, we can load directly
        model = AutoModelForCausalLM.from_pretrained(
            latest,
            torch_dtype=dtype,
            device_map=device_map,
        )

    model.eval()

    # Prepare input. If the base model is chat-instruct, try to use its chat template when available.
    prompt = args.prompt
    try:
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            messages = [
                {"role": "user", "content": prompt}
            ]
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            input_text = prompt
    except Exception:
        input_text = prompt

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        padding=True,
    )
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    with torch.inference_mode():
        outputs = model.generate(**inputs, **gen_kwargs)

    # Decode only the generated continuation
    generated = tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)[0]
    print("\n=== PROMPT ===")
    print(prompt)
    print("\n=== COMPLETION ===")
    print(generated.strip())

if __name__ == "__main__":
    main()
