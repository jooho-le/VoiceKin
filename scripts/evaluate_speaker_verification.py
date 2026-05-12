import argparse
import csv
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings
from app.services.model_provider import get_speaker_service
from app.utils.audio import cleanup_temp_files, convert_audio_to_standard_wav


DEFAULT_THRESHOLDS = "0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85"


def parse_thresholds(raw_value: str) -> list[float]:
    thresholds: list[float] = []
    for item in raw_value.split(","):
        value = item.strip()
        if not value:
            continue
        threshold = float(value)
        if not -1.0 <= threshold <= 1.0:
            raise ValueError("speaker threshold must be between -1.0 and 1.0")
        thresholds.append(threshold)
    if not thresholds:
        raise ValueError("at least one threshold is required")
    return thresholds


def normalize_label(label: str) -> str:
    value = label.strip().lower()
    if value in {"same", "same_speaker", "1", "true", "yes"}:
        return "same"
    if value in {"different", "different_speaker", "diff", "0", "false", "no"}:
        return "different"
    raise ValueError(f"unsupported label: {label}")


def resolve_audio_path(raw_path: str, dataset_root: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return dataset_root / path


def display_path(path: Path, base_dir: Path) -> str:
    try:
        return str(path.relative_to(base_dir))
    except ValueError:
        return str(path)


def load_pairs(pairs_csv: Path, dataset_root: Path) -> list[dict[str, object]]:
    if not pairs_csv.exists():
        raise SystemExit(f"pairs CSV not found: {pairs_csv}")

    rows: list[dict[str, object]] = []
    with pairs_csv.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        required_fields = {"audio_file_1", "audio_file_2", "label"}
        if not reader.fieldnames or not required_fields.issubset(reader.fieldnames):
            raise SystemExit(
                "pairs CSV must contain columns: audio_file_1,audio_file_2,label"
            )

        for row in reader:
            audio_file_1 = resolve_audio_path(row["audio_file_1"], dataset_root)
            audio_file_2 = resolve_audio_path(row["audio_file_2"], dataset_root)
            if not audio_file_1.exists():
                raise SystemExit(f"audio_file_1 not found: {audio_file_1}")
            if not audio_file_2.exists():
                raise SystemExit(f"audio_file_2 not found: {audio_file_2}")

            rows.append(
                {
                    "audio_file_1": audio_file_1,
                    "audio_file_2": audio_file_2,
                    "label": normalize_label(row["label"]),
                }
            )

    if not rows:
        raise SystemExit(f"pairs CSV has no rows: {pairs_csv}")
    return rows


def is_correct(label: str, is_same_speaker: bool) -> bool:
    expected_same = label == "same"
    return expected_same == is_same_speaker


def compute_metrics(rows: list[dict[str, object]], threshold: float) -> dict[str, object]:
    tp = tn = fp = fn = 0

    for row in rows:
        label = str(row["label"])
        prediction_same = float(row["similarity"]) >= threshold
        expected_same = label == "same"

        if prediction_same and expected_same:
            tp += 1
        elif prediction_same and not expected_same:
            fp += 1
        elif not prediction_same and expected_same:
            fn += 1
        else:
            tn += 1

    total = tp + tn + fp + fn
    correct = tp + tn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    false_accept_rate = fp / (fp + tn) if fp + tn else 0.0
    false_reject_rate = fn / (fn + tp) if fn + tp else 0.0

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
        "false_accept_rate": round(false_accept_rate, 4),
        "false_reject_rate": round(false_reject_rate, 4),
    }


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(args: argparse.Namespace) -> None:
    settings = get_settings()
    pairs_csv = args.pairs.resolve()
    dataset_root = (args.dataset_root or pairs_csv.parent).resolve()
    output_path = args.output.resolve()
    metrics_output_path = args.metrics_output.resolve()
    thresholds = parse_thresholds(args.thresholds)

    pairs = load_pairs(pairs_csv, dataset_root)
    if args.limit is not None:
        pairs = pairs[: args.limit]

    service = get_speaker_service()
    converted_cache: dict[Path, Path] = {}
    result_rows: list[dict[str, object]] = []

    try:
        for index, pair in enumerate(pairs, start=1):
            audio_file_1 = pair["audio_file_1"]
            audio_file_2 = pair["audio_file_2"]
            label = str(pair["label"])
            print(f"[{index}/{len(pairs)}] speaker pair: {audio_file_1} <-> {audio_file_2}")

            wav_1 = get_or_convert(audio_file_1, converted_cache, settings)
            wav_2 = get_or_convert(audio_file_2, converted_cache, settings)

            result = service.compare_files(wav_1, wav_2)
            row = {
                "audio_file_1": display_path(audio_file_1, dataset_root),
                "audio_file_2": display_path(audio_file_2, dataset_root),
                "label": label,
                "similarity": result.similarity,
                "configured_threshold": result.threshold,
                "is_same_speaker": result.is_same_speaker,
                "correct": is_correct(label, result.is_same_speaker),
                "message": result.message,
                "model_name": result.model_name,
            }
            result_rows.append(row)
    finally:
        cleanup_temp_files(converted_cache.values())

    result_fields = [
        "audio_file_1",
        "audio_file_2",
        "label",
        "similarity",
        "configured_threshold",
        "is_same_speaker",
        "correct",
        "message",
        "model_name",
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
        "false_accept_rate",
        "false_reject_rate",
    ]
    write_csv(metrics_output_path, metric_rows, metric_fields)

    best_metric = max(metric_rows, key=lambda row: (float(row["accuracy"]), float(row["recall"])))
    print(f"Saved per-pair results: {output_path}")
    print(f"Saved threshold metrics: {metrics_output_path}")
    print(
        "Best threshold by accuracy: "
        f"{best_metric['threshold']} "
        f"(accuracy={best_metric['accuracy']}, "
        f"false_accept_rate={best_metric['false_accept_rate']}, "
        f"false_reject_rate={best_metric['false_reject_rate']})"
    )


def get_or_convert(audio_path: Path, cache: dict[Path, Path], settings) -> Path:
    if audio_path in cache:
        return cache[audio_path]

    wav_path = convert_audio_to_standard_wav(
        input_path=audio_path,
        target_sample_rate=settings.target_sample_rate,
        min_audio_seconds=settings.min_audio_seconds,
    )
    cache[audio_path] = wav_path
    return wav_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate speaker verification on a labeled pairs CSV.",
    )
    parser.add_argument(
        "--pairs",
        type=Path,
        default=Path("datasets/speaker_verification/pairs.csv"),
        help="CSV with columns: audio_file_1,audio_file_2,label",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Base directory for relative paths in pairs CSV. Defaults to pairs CSV directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/speaker_verification_results.csv"),
        help="Per-pair output CSV path.",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("reports/speaker_verification_threshold_metrics.csv"),
        help="Threshold sweep metrics CSV path.",
    )
    parser.add_argument(
        "--thresholds",
        default=DEFAULT_THRESHOLDS,
        help="Comma-separated thresholds to evaluate. Example: 0.65,0.70,0.75",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of pairs to evaluate for quick smoke tests.",
    )
    return parser


if __name__ == "__main__":
    evaluate(build_parser().parse_args())
