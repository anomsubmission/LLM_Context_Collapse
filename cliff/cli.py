"""Single entry point:  cliff <command> [options]   (or python -m cliff ...)"""
from __future__ import annotations

import sys
import textwrap

COMMANDS = {
    "verify":    ("cliff.verify",                    "re-derive every reported number from the archive (no GPU, ~1s)"),
    "figures":   ("cliff.figures",                   "regenerate the main-text figures from the archive (no GPU)"),
    "sweep":     ("cliff.experiments.behavioural",   "measure the in-context cliff for a model (any backend)"),
    "intervene": ("cliff.experiments.intervention",  "region-restricted LoRA: remove or induce the cliff (needs weights)"),
    "mechanism": ("cliff.experiments.mechanism",     "function-vector and induction-head profiles (needs weights)"),
    "audit":     ("cliff.experiments.induce_audit",  "audit the induce-arm supervision target (needs weights)"),
    "doctor":    ("cliff.doctor",                    "report the environment and what you can run right now"),
}

def _block(keys):
    return "\n".join(f"      {k:11s} {COMMANDS[k][1]}" for k in keys)


USAGE = textwrap.dedent(f"""\
    cliff -- reproduce "Localizing and repairing the in-context cliff"

    usage: cliff <command> [options]

    Tier 0/1 -- no GPU, no network, no model downloads:
{_block(('verify', 'figures', 'doctor'))}

    Tier 2 -- re-run the experiments:
{_block(('sweep', 'intervene', 'mechanism', 'audit'))}

    Start here:
      cliff doctor                 # what can this machine run?
      cliff verify                 # confirm the paper's numbers against the data
      cliff figures                # redraw Figs 1, 3, 4

    Then, for example:
      cliff sweep     --backend echo --model echo --ks 0,1,4 --seeds 0   # 1s smoke test
      cliff sweep     --model meta-llama/Llama-3.1-8B-Instruct
      cliff intervene --model meta-llama/Llama-3.1-8B-Instruct --region early
      cliff intervene --model meta-llama/Llama-3.1-8B-Instruct --region late

    `cliff <command> --help` documents each command.
    """)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(USAGE)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd not in COMMANDS:
        print(f"unknown command {cmd!r}\n", file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2
    import importlib
    from .backends.base import BackendCapabilityError
    try:
        return importlib.import_module(COMMANDS[cmd][0]).main(rest) or 0
    except BackendCapabilityError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except FileNotFoundError as e:
        print(f"error: {e}\n\nIf the results archive is elsewhere, set "
              f"CLIFF_RESULTS=/path/to/results.", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
