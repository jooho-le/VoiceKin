import argparse
import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings
from app.services.model_provider import get_anti_spoofing_service
from app.utils.audio import cleanup_temp_files, convert_audio_to_standard_wav


DEFAULT_THRESHOLDS = "0.03,0.05,0.07,0.10,0.20,0.30,0.50"


def parse_thresholds(raw_value: str) -> list[float]:
    thresholds: list[float] = []
    for item in raw_value.split(","):
        value = item.strip()
        if not value:
            continue
        threshold = float(value)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")
        thresholds.append(threshold)
    if not thresholds:
        raise ValueError("at least one threshold is required")
    return thresholds


def collect_audio_files(dataset_dir: Path, allowed_extensions: tuple[str, ...]) -> list[tuple[Path, str]]:
    rows: list[tuple[Path, str]] = []
    allowed = {extension.lower().lstrip(".") for extension in allowed_extensions}

    for label in ("real", "fake"):
        label_dir = dataset_dir / label
        if not label_dir.exists():
            continue

        for path in sorted(label_dir.rglob("*")):
            if path.is_file() and path.suffix.lower().lstrip(".") in allowed:
                rows.append((path, label))

    return rows


def is_correct(label: str, is_spoofed: bool) -> bool:
    expected_spoofed = label == "fake"
    return expected_spoofed == is_spoofed


def compute_metrics(rows: list[dict[str, str]], threshold: float) -> dict[str, object]:
    tp = tn = fp = fn = 0

    for row in rows:
        label = row["label"]
        prediction_spoofed = float(row["spoof_score"]) >= threshold
        expected_spoofed = label == "fake"

        if prediction_spoofed and expected_spoofed:
            tp += 1
        elif prediction_spoofed and not expected_spoofed:
            fp += 1
        elif not prediction_spoofed and expected_spoofed:
            fn += 1
        else:
            tn += 1

    total = tp + tn + fp + fn
    correct = tp + tn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    false_positive_rate = fp / (fp + tn) if fp + tn else 0.0
    false_negative_rate = fn / (fn + tp) if fn + tp else 0.0

    return {
        "threshold": threshold,
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "false_positive_rate": round(false_positive_rate, 4),
        "false_negative_rate": round(false_negative_rate, 4),
    }


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(args: argparse.Namespace) -> None:
    settings = get_settings()
    dataset_dir = args.dataset.resolve()
    output_path = args.output.resolve()
    metrics_output_path = args.metrics_output.resolve()
    thresholds = parse_thresholds(args.thresholds)

    files = collect_audio_files(dataset_dir, settings.allowed_audio_extensions)
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        raise SystemExit(
            f"No audio files found. Put files under {dataset_dir}/real and {dataset_dir}/fake"
        )

    service = get_anti_spoofing_service()
    temp_paths: list[Path | None] = []
    result_rows: list[dict[str, object]] = []

    try:
        for index, (audio_path, label) in enumerate(files, start=1):
            print(f"[{index}/{len(files)}] anti-spoofing: {audio_path}")
            wav_path = convert_audio_to_standard_wav(
                input_path=audio_path,
                target_sample_rate=settings.target_sample_rate,
                min_audio_seconds=settings.min_audio_seconds,
            )
            temp_paths.append(wav_path)

            result = service.detect_file(wav_path)
            row = {
                "file": str(audio_path.relative_to(dataset_dir)),
                "label": label,
                "spoof_score": result.spoof_score,
                "configured_threshold": result.threshold,
                "is_spoofed": result.is_spoofed,
                "correct": is_correct(label, result.is_spoofed),
                "predicted_label": result.predicted_label,
                "predicted_score": result.predicted_score,
                "model_message": result.message,
                "analyzed_segments": result.analyzed_segments,
                "max_spoof_segment_index": result.max_spoof_segment_index,
                "segment_seconds": result.segment_seconds,
                "model_name": result.model_name,
                "label_scores_json": json.dumps(
                    [
                        {"label": label_score.label, "score": label_score.score}
                        for label_score in result.label_scores
                    ],
                    ensure_ascii=False,
                ),
            }
            result_rows.append(row)
    finally:
        cleanup_temp_files(temp_paths)

    result_fields = [
        "file",
        "label",
        "spoof_score",
        "configured_threshold",
        "is_spoofed",
        "correct",
        "predicted_label",
        "predicted_score",
        "model_message",
        "analyzed_segments",
        "max_spoof_segment_index",
        "segment_seconds",
        "model_name",
        "label_scores_json",
    ]
    write_csv(output_path, result_rows, result_fields)

    metric_rows = [compute_metrics(result_rows, threshold) for threshold in thresholds]
    metric_fields = [
        "threshold",
        "total",
        "correct",
        "accuracy",
        "tp",
        "tn",
        "fp",
        "fn",
        "precision",
        "recall",
        "false_positive_rate",
        "false_negative_rate",
    ]
    write_csv(metrics_output_path, metric_rows, metric_fields)

    best_metric = max(metric_rows, key=lambda row: (float(row["accuracy"]), float(row["recall"])))
    print(f"Saved per-file results: {output_path}")
    print(f"Saved threshold metrics: {metrics_output_path}")
    print(
        "Best threshold by accuracy: "
        f"{best_metric['threshold']} "
        f"(accuracy={best_metric['accuracy']}, "
        f"fp={best_metric['fp']}, fn={best_metric['fn']})"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the anti-spoofing model on datasets/anti_spoofing/real and fake.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("datasets/anti_spoofing"),
        help="Dataset directory containing real/ and fake/ subdirectories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/anti_spoofing_results.csv"),
        help="Per-file output CSV path.",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("reports/anti_spoofing_threshold_metrics.csv"),
        help="Threshold sweep metrics CSV path.",
    )
    parser.add_argument(
        "--thresholds",
        default=DEFAULT_THRESHOLDS,
        help="Comma-separated thresholds to evaluate. Example: 0.05,0.07,0.10",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of files to evaluate for quick smoke tests.",
    )
    return parser


if __name__ == "__main__":
    evaluate(build_parser().parse_args())
