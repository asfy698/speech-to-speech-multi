from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import time


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load only the Gemma 4 E4B LLM on the GPU and keep it alive for VRAM inspection."
    )
    parser.add_argument(
        "--model_id",
        default=os.getenv("GEMMA_MODEL_ID", "ggml-org/gemma-4-E4B-it-GGUF"),
        help="Gemma model id to load. Default matches the repo's GGUF example.",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "llama-server", "transformers"),
        default=os.getenv("GEMMA_BACKEND", "auto"),
        help="How to load the model. Auto picks llama-server for GGUF models.",
    )
    parser.add_argument(
        "--revision",
        default=os.getenv("GEMMA_MODEL_REVISION"),
        help="Optional model revision for the Transformers backend.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default=os.getenv("GEMMA_DTYPE", "float16"),
        help="Torch dtype for the Transformers backend.",
    )
    parser.add_argument(
        "--cuda_visible_devices",
        default=os.getenv("CUDA_VISIBLE_DEVICES"),
        help="Optional CUDA_VISIBLE_DEVICES value to pin the process to one GPU.",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Run one tiny generation step after loading when using the Transformers backend.",
    )
    parser.add_argument(
        "--prompt",
        default="Hello",
        help="Warmup prompt for the Transformers backend.",
    )
    return parser.parse_args()


def _load_with_llama_server(model_id: str) -> int:
    if shutil.which("llama-server") is None:
        raise SystemExit("llama-server was not found on PATH. Install llama.cpp or use --backend transformers.")

    command = [
        "llama-server",
        "-hf",
        model_id,
        "-np",
        "2",
        "-c",
        "65536",
        "-fa",
        "on",
        "--swa-full",
    ]

    process = subprocess.Popen(command)
    print(f"Started llama-server with PID {process.pid}", flush=True)
    print("Only the Gemma LLM is being loaded in this process.", flush=True)

    try:
        while True:
            time.sleep(1)
            if process.poll() is not None:
                raise SystemExit(f"llama-server exited with code {process.returncode}")
    except KeyboardInterrupt:
        print("Stopping llama-server and exiting.", flush=True)
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()

    return 0


def _load_with_transformers(model_id: str, revision: str | None, dtype: str, warmup: bool, prompt: str) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. This script needs an NVIDIA GPU for the Transformers backend.")

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[dtype]

    print(f"Loading model: {model_id}", flush=True)
    print(f"Requested dtype: {dtype}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    ).to("cuda:0")
    model.eval()

    device_name = torch.cuda.get_device_name(0)
    total_vram_gib = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    allocated_gib = torch.cuda.memory_allocated(0) / (1024**3)
    reserved_gib = torch.cuda.memory_reserved(0) / (1024**3)

    print(f"Loaded on GPU: {device_name}", flush=True)
    print(f"Total VRAM: {total_vram_gib:.2f} GiB", flush=True)
    print(f"Allocated VRAM after load: {allocated_gib:.2f} GiB", flush=True)
    print(f"Reserved VRAM after load: {reserved_gib:.2f} GiB", flush=True)
    print("Only the Gemma LLM is loaded in this process.", flush=True)

    if warmup:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
        with torch.inference_mode():
            _ = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        torch.cuda.synchronize()
        allocated_gib = torch.cuda.memory_allocated(0) / (1024**3)
        reserved_gib = torch.cuda.memory_reserved(0) / (1024**3)
        print(f"Allocated VRAM after warmup: {allocated_gib:.2f} GiB", flush=True)
        print(f"Reserved VRAM after warmup: {reserved_gib:.2f} GiB", flush=True)

    print("Keeping the process alive. Press Ctrl+C to unload the model.", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Unloading model and exiting.", flush=True)
    finally:
        del model
        del tokenizer
        torch.cuda.empty_cache()

    return 0


def main() -> int:
    args = _parse_args()

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    elif "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    backend = args.backend
    model_id_lower = args.model_id.lower()
    if backend == "auto":
        backend = "llama-server" if ("gguf" in model_id_lower or model_id_lower.endswith(".gguf")) else "transformers"

    print(f"Backend: {backend}", flush=True)
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)

    if backend == "llama-server":
        return _load_with_llama_server(args.model_id)

    return _load_with_transformers(args.model_id, args.revision, args.dtype, args.warmup, args.prompt)


if __name__ == "__main__":
    raise SystemExit(main())
