"""Report the environment and exactly which commands this machine can run."""
from __future__ import annotations

import importlib
import platform
import shutil
import sys

from .paths import figures_dir, outputs_dir, results_dir


def _probe(mod: str) -> str | None:
    try:
        m = importlib.import_module(mod)
        return getattr(m, "__version__", "installed")
    except Exception:
        return None


def main(argv=None) -> int:
    print(f"cliff doctor\n{'=' * 62}")
    print(f"  python      {sys.version.split()[0]}  ({platform.machine()}, {platform.system()})")

    vers = {m: _probe(m) for m in
            ("numpy", "matplotlib", "torch", "transformers", "peft", "accelerate",
             "datasets", "bitsandbytes")}
    for m, v in vers.items():
        print(f"  {m:13s} {v or '-- not installed'}")

    device = "cpu"
    if vers["torch"]:
        import torch
        if torch.cuda.is_available():
            device = f"cuda ({torch.cuda.get_device_name(0)}, " \
                     f"{torch.cuda.get_device_properties(0).total_memory // 2**30} GB)"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps (Apple Silicon)"
    print(f"\n  compute     {device}")

    rd = results_dir()
    n_files = sum(1 for _ in rd.rglob("*")) if rd.exists() else 0
    print(f"  results     {rd}  ({n_files} files)" if rd.exists()
          else f"  results     {rd}  \033[31mMISSING\033[0m")
    print(f"  figures ->  {figures_dir()}")
    print(f"  outputs ->  {outputs_dir()}")
    if shutil.disk_usage(".").free < 30 * 2**30:
        print("  \033[33mnote\033[0m  <30 GB free; model downloads may not fit")

    print(f"\n{'=' * 62}\n  what you can run now:")
    ok_analysis = rd.exists()
    ok_fig = ok_analysis and vers["matplotlib"] and vers["numpy"]
    ok_gen = bool(vers["torch"] and vers["transformers"])
    ok_train = ok_gen and bool(vers["peft"] and vers["accelerate"])
    ok_data = bool(vers["datasets"])
    rows = [
        ("cliff verify", ok_analysis, "needs the results/ archive"),
        ("cliff figures", ok_fig, "needs matplotlib + numpy"),
        ("cliff sweep --backend echo", ok_data, "needs datasets"),
        ("cliff sweep --backend openai_compat", ok_data, "needs datasets + a reachable endpoint"),
        ("cliff sweep --backend hf_local", ok_gen and ok_data, "needs torch + transformers + datasets"),
        ("cliff mechanism", ok_gen and ok_data, "needs torch + transformers + datasets"),
        ("cliff audit", ok_gen and ok_data, "needs torch + transformers + datasets"),
        ("cliff intervene", ok_train and ok_data, "needs torch + transformers + peft + accelerate + datasets"),
    ]
    for name, ok, why in rows:
        mark = "\033[32myes\033[0m" if ok else "\033[31mno \033[0m"
        print(f"    {mark}  {name:38s} {'' if ok else why}")
    if not ok_gen:
        print("\n  install the experiment extras with:  pip install -e '.[experiments]'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
