import argparse
import gc
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from src.eval import UniformAgent, test_agent
from src.tb import check_model


METRICS = ["WINS ratio", "DRAWS ratio", "LOSSES ratio", "% Optimal Moves"]


def step_from_filename(path: Path) -> int:
    m = re.search(r"ckpt-(\d+)\.pt$", path.name)
    if not m:
        raise ValueError(f"Unexpected checkpoint name: {path.name}")
    return int(m.group(1))


def evaluate_all(ckpt_dir: Path) -> list[dict]:
    ckpt_paths = sorted(ckpt_dir.glob("ckpt-*.pt"), key=step_from_filename)
    if not ckpt_paths:
        raise SystemExit(f"No checkpoints found under {ckpt_dir}")

    results: list[dict] = []
    for path in ckpt_paths:
        step = step_from_filename(path)
        print(f"Evaluating {path.name} (step {step})")
        ckpt = torch.load(path, weights_only=False)
        model = ckpt["model"]
        env = ckpt["env"]

        try:
            check_model(model)
            res = test_agent(env, model, UniformAgent())
        except ValueError as e:
            first_line = str(e).splitlines()[0]
            print(f"  Skipped (corrupted weights): {first_line}")
            del model, env, ckpt
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            continue
        except Exception as e:
            print(f"  Skipped ({type(e).__name__}): {e}")
            del model, env, ckpt
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            continue

        res = {k: float(v) for k, v in res.items()}
        res["step"] = step
        res["checkpoint"] = path.name
        results.append(res)
        print(f"  {res}")

        del model, env, ckpt
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return results


def write_plot(results: list[dict], out_path: Path, title: str) -> None:
    steps = [r["step"] for r in results]
    fig, ax = plt.subplots(figsize=(9, 5))
    for metric in METRICS:
        ax.plot(
            steps,
            [r.get(metric, float("nan")) for r in results],
            marker="o",
            label=metric,
        )
    ax.set_xlabel("Training step")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_markdown(
    results: list[dict], out_path: Path, plot_name: str, ckpt_dir: Path
) -> None:
    header = "| Checkpoint | Step | " + " | ".join(METRICS) + " |"
    sep = "|" + "|".join(["---"] * (2 + len(METRICS))) + "|"
    rows = []
    for r in results:
        cells = [r["checkpoint"], str(r["step"])] + [
            f"{r.get(m, float('nan')):.4f}" for m in METRICS
        ]
        rows.append("| " + " | ".join(cells) + " |")

    best_step = max(results, key=lambda r: r.get("% Optimal Moves", -1))["step"]

    body = [
        f"# Evaluation summary — `{ckpt_dir}`",
        "",
        f"- Checkpoints evaluated: **{len(results)}**",
        f"- Step range: **{results[0]['step']} → {results[-1]['step']}**",
        f"- Best `% Optimal Moves` at step **{best_step}**",
        "",
        f"![performance]({plot_name})",
        "",
        header,
        sep,
        *rows,
        "",
    ]
    out_path.write_text("\n".join(body))


def main(args) -> None:
    ckpt_dir = Path(args.ckpt_dir)
    out_root = Path(args.out_dir)
    out_dir = out_root / ckpt_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    results = evaluate_all(ckpt_dir)
    if not results:
        raise SystemExit("No successful evaluations.")

    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    plot_name = "performance.png"
    write_plot(results, out_dir / plot_name, title=f"Checkpoint evaluation — {ckpt_dir.name}")
    write_markdown(results, out_dir / "summary.md", plot_name, ckpt_dir)

    print(f"\nWrote {out_dir / 'results.json'}")
    print(f"Wrote {out_dir / plot_name}")
    print(f"Wrote {out_dir / 'summary.md'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate every ckpt-*.pt in a folder and produce plots + a Markdown report."
    )
    parser.add_argument("ckpt_dir", type=str, help="Folder containing ckpt-*.pt files.")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="Outputs",
        help="Root directory for outputs (default: Outputs).",
    )
    args = parser.parse_args()
    main(args)
