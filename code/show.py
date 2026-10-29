"""Analyze model predictions and export validation mistakes to text/JSON files."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

from common import DATA, PROTOCOL, load_inputs, parse_answer, read_jsonl, validate_choices


def load_method(method_path):
    """Load a Python method file exposing predict(bank, arrays, checkpoint)."""
    method_path = Path(method_path).resolve()
    sys.path.insert(0, str(method_path.parent))
    spec = importlib.util.spec_from_file_location("analysis_method", method_path)
    method = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = method
    spec.loader.exec_module(method)
    if not hasattr(method, "predict"):
        raise AttributeError(f"{method_path} does not expose predict(bank, arrays, checkpoint)")
    return method


def generate_predictions(method_path, checkpoint, split, data_dir):
    """Run a model method on a split and return ids, selected choices, and timing info."""
    tick = time.perf_counter()
    bank, arrays = load_inputs(split, data_dir)
    method = load_method(method_path)
    choices = method.predict(bank, arrays, checkpoint)
    ids = [row["id"] for row in bank]
    validate_choices(bank, ids, choices)
    return bank, ids, choices, time.perf_counter() - tick


def load_predictions_file(predictions_path, split, data_dir):
    """Load an existing predictions JSON file and validate its format."""
    tick = time.perf_counter()
    predictions_path = Path(predictions_path)
    payload = json.loads(predictions_path.read_text())
    if payload.get("protocol") != PROTOCOL:
        raise ValueError("Prediction protocol does not match current dataset protocol")
    if payload.get("split") != split:
        raise ValueError(f"Prediction split mismatch: expected {split}, got {payload.get('split')}")
    bank, _ = load_inputs(split, data_dir)
    ids = payload["ids"]
    choices = payload["selected"]
    validate_choices(bank, ids, choices)
    return bank, ids, choices, time.perf_counter() - tick


def analyze_predictions(bank, choices, split, data_dir):
    """Compare selected candidates against gold answers and collect wrong cases."""
    references = read_jsonl(Path(data_dir) / f"gsm8k_{split}.jsonl")
    if len(references) != len(bank):
        raise ValueError("Reference length mismatch")

    errors = []
    correct = 0
    selected_counts = {str(i): 0 for i in range(4)}
    wrong_selected_counts = {str(i): 0 for i in range(4)}

    for row, gold, selected in zip(bank, references, choices):
        if row["id"] != gold["id"] or row["question"] != gold["question"]:
            raise ValueError("Reference/bank alignment mismatch")

        gold_answer = parse_answer(gold["answer"])
        candidates = []
        correct_indices = []

        for candidate in row["candidates"]:
            parsed = parse_answer(candidate["text"])
            item = {
                "index": candidate["index"],
                "text": candidate["text"],
                "parsed_answer": parsed,
            }
            candidates.append(item)
            if parsed is not None and parsed == gold_answer:
                correct_indices.append(candidate["index"])

        selected_counts[str(selected)] += 1
        selected_candidate = candidates[selected]
        is_correct = selected_candidate["parsed_answer"] is not None and selected_candidate["parsed_answer"] == gold_answer

        if is_correct:
            correct += 1
        else:
            wrong_selected_counts[str(selected)] += 1
            errors.append(
                {
                    "id": row["id"],
                    "question": row["question"],
                    "gold_answer_text": gold["answer"],
                    "gold_answer_parsed": gold_answer,
                    "selected_index": selected,
                    "selected_text": selected_candidate["text"],
                    "selected_parsed_answer": selected_candidate["parsed_answer"],
                    "correct_indices": correct_indices,
                    "correct_candidates": [c for c in candidates if c["index"] in correct_indices],
                    "candidates": candidates,
                }
            )

    total = len(bank)
    incorrect = total - correct
    summary = {
        "protocol": PROTOCOL,
        "split": split,
        "total_questions": total,
        "correct_questions": correct,
        "incorrect_questions": incorrect,
        "accuracy": 100.0 * correct / total if total else 0.0,
        "selected_index_counts": selected_counts,
        "wrong_selected_index_counts": wrong_selected_counts,
    }
    return summary, errors


def render_error_report(summary, errors):
    """Render a readable plain-text report for wrong predictions."""
    lines = [
        f"Split: {summary['split']}",
        f"Total questions: {summary['total_questions']}",
        f"Correct questions: {summary['correct_questions']}",
        f"Incorrect questions: {summary['incorrect_questions']}",
        f"Accuracy: {summary['accuracy']:.2f}%",
        f"Selected index counts: {json.dumps(summary['selected_index_counts'], ensure_ascii=False)}",
        f"Wrong selected index counts: {json.dumps(summary['wrong_selected_index_counts'], ensure_ascii=False)}",
        "",
        "=" * 100,
        "WRONG PREDICTIONS",
        "=" * 100,
        "",
    ]

    for idx, item in enumerate(errors, start=1):
        lines.append(f"[{idx}] Question ID: {item['id']}")
        lines.append(f"Question: {item['question']}")
        lines.append(f"Gold answer parsed: {item['gold_answer_parsed']}")
        lines.append("Gold answer raw:")
        lines.append(item["gold_answer_text"])
        lines.append("")
        lines.append(f"Model selected index: {item['selected_index']}")
        lines.append(f"Model selected parsed answer: {item['selected_parsed_answer']}")
        lines.append("Model selected text:")
        lines.append(item["selected_text"])
        lines.append("")
        lines.append(f"Correct candidate indices: {item['correct_indices'] if item['correct_indices'] else 'None'}")
        lines.append("All candidates:")
        for candidate in item["candidates"]:
            lines.append(f"  - Candidate {candidate['index']} | parsed_answer={candidate['parsed_answer']}")
            lines.append(candidate["text"])
            lines.append("")
        lines.append("-" * 100)

    return "\n".join(lines).rstrip() + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--method",
        type=Path,
        help="Python file exposing predict(bank, arrays, checkpoint), e.g. my_model.py",
    )
    source_group.add_argument(
        "--predictions",
        type=Path,
        help="Existing predictions JSON produced by predict.py",
    )
    parser.add_argument("--checkpoint", type=Path, help="Checkpoint path used together with --method")
    parser.add_argument("--split", choices=("train", "validation", "test", "audit"), default="validation")
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/show"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.method is not None:
        bank, ids, choices, prediction_seconds = generate_predictions(
            args.method, args.checkpoint, args.split, args.data_dir
        )
        source = {
            "type": "method",
            "method": str(args.method),
            "checkpoint": str(args.checkpoint) if args.checkpoint else None,
            "prediction_seconds": prediction_seconds,
        }
    else:
        bank, ids, choices, prediction_seconds = load_predictions_file(
            args.predictions, args.split, args.data_dir
        )
        source = {
            "type": "predictions_file",
            "predictions": str(args.predictions),
            "load_seconds": prediction_seconds,
        }

    started = time.perf_counter()
    summary, errors = analyze_predictions(bank, choices, args.split, args.data_dir)
    summary.update(
        {
            "source": source,
            "ids": ids,
            "selected": choices,
            "analysis_seconds": time.perf_counter() - started,
        }
    )

    summary_path = args.output_dir / f"{args.split}_summary.json"
    errors_json_path = args.output_dir / f"{args.split}_errors.json"
    errors_txt_path = args.output_dir / f"{args.split}_errors.txt"

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    errors_json_path.write_text(json.dumps(errors, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    errors_txt_path.write_text(render_error_report(summary, errors), encoding="utf-8")

    print(json.dumps(
        {
            "protocol": PROTOCOL,
            "split": args.split,
            "accuracy": summary["accuracy"],
            "incorrect_questions": summary["incorrect_questions"],
            "summary_file": str(summary_path),
            "errors_json_file": str(errors_json_path),
            "errors_txt_file": str(errors_txt_path),
        },
        indent=2,
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()