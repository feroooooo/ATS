#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from statistics import fmean

METRIC_KEYS = ("test_top1_acc", "test_top5_acc", "mAP", "similarity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Average ATS test metrics across multiple subject result files."
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."), help="ATS repository root")
    parser.add_argument("--config", required=True, help="Config path relative to repo root or absolute path")
    parser.add_argument("--exp-name", required=True, help="Experiment directory name")
    parser.add_argument("--run-name", required=True, help="Run directory name")
    parser.add_argument("--subjects", nargs="+", required=True, help="Subject directory names, e.g. sub-01")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional summary json path. Defaults to <run_dir>/average_metrics.json",
    )
    return parser.parse_args()


def resolve_config_path(repo_root: Path, config_arg: str) -> Path:
    config_path = Path(config_arg)
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    return config_path.resolve()


def extract_save_dir(config_path: Path) -> str:
    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("save_dir:"):
            value = line.split(":", 1)[1].strip()
            if not value:
                raise ValueError(f"Empty save_dir in config: {config_path}")
            return value.strip().strip("\"'")
    raise ValueError(f"Could not find save_dir in config: {config_path}")


def load_subject_metrics(result_path: Path) -> dict:
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        if not payload:
            raise ValueError(f"Empty result list: {result_path}")
        payload = payload[0]
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected result format in {result_path}")

    metrics = {}
    for key in METRIC_KEYS:
        if key not in payload:
            raise KeyError(f"Missing metric '{key}' in {result_path}")
        metrics[key] = float(payload[key])
    return metrics


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    config_path = resolve_config_path(repo_root, args.config)
    save_dir = extract_save_dir(config_path)
    run_dir = (repo_root / save_dir / args.exp_name / args.run_name).resolve()

    missing_paths = []
    per_subject = {}
    for subject in args.subjects:
        result_path = run_dir / subject / "test_results.json"
        if not result_path.exists():
            missing_paths.append(str(result_path))
            continue
        per_subject[subject] = {
            "metrics": load_subject_metrics(result_path),
            "path": str(result_path),
        }

    if missing_paths:
        missing = "\n".join(missing_paths)
        raise FileNotFoundError(f"Missing test_results.json files:\n{missing}")

    averages = {
        key: fmean(subject_info["metrics"][key] for subject_info in per_subject.values())
        for key in METRIC_KEYS
    }

    output_path = args.output.resolve() if args.output else run_dir / "average_metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "config": str(config_path),
        "results_root": str(run_dir),
        "subjects": args.subjects,
        "average": averages,
        "per_subject": per_subject,
    }
    output_path.write_text(json.dumps(summary, indent=4), encoding="utf-8")

    print(f"Saved averaged metrics to: {output_path}")
    for key in METRIC_KEYS:
        print(f"{key}: {averages[key]:.10f}")


if __name__ == "__main__":
    main()
