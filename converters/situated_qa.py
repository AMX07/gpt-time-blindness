"""Convert SituatedQA temporal subset to standard eval format.

Source: https://github.com/mikejqzhang/SituatedQA
Format: JSONL with fields: question, id, edited_question, date, date_type, answer, any_answer

Usage:
    python -m converters.situated_qa --output datasets/situated_qa.json --limit 100
"""

import argparse
import json
import os
import urllib.request
from pathlib import Path

# Raw GitHub URLs for the temporal QA data
SITUATED_QA_BASE = "https://raw.githubusercontent.com/mikejqzhang/SituatedQA/master/data/qa_data"
SPLITS = {
    "train": f"{SITUATED_QA_BASE}/temp.train.jsonl",
    "dev": f"{SITUATED_QA_BASE}/temp.dev.jsonl",
    "test": f"{SITUATED_QA_BASE}/temp.test.jsonl",
}

CACHE_DIR = Path(__file__).parent.parent / "datasets" / ".cache"


def download_split(split: str) -> Path:
    """Download a split if not already cached."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"situated_qa_{split}.jsonl"

    if cache_path.exists():
        print(f"  Using cached {split} split")
        return cache_path

    url = SPLITS[split]
    print(f"  Downloading {split} split from {url}")
    urllib.request.urlretrieve(url, cache_path)
    return cache_path


def parse_jsonl(path: Path) -> list[dict]:
    """Parse a JSONL file into a list of dicts."""
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def convert_entry(entry: dict) -> dict | None:
    """Convert a single SituatedQA entry to our standard eval format."""
    # Skip non-temporal questions
    if entry.get("date_type") == "orig":
        return None

    date = entry.get("date", "")
    question = entry.get("edited_question") or entry.get("question", "")
    answers = entry.get("answer", [])
    any_answers = entry.get("any_answer", [])

    return {
        "id": str(entry.get("id", "")),
        "description": f"Temporal QA (as of {date}): {entry.get('question', '')[:80]}",
        "messages": [{"role": "user", "content": question}],
        "time_offsets": [0],
        "expected_answer": answers,
        "metadata": {
            "source": "situated_qa",
            "date": date,
            "date_type": entry.get("date_type", ""),
            "original_question": entry.get("question", ""),
            "any_answer": any_answers,
        },
    }


def convert(splits: list[str], limit: int | None = None) -> list[dict]:
    """Download and convert SituatedQA data."""
    all_cases = []

    for split in splits:
        print(f"Processing {split}...")
        path = download_split(split)
        entries = parse_jsonl(path)

        for entry in entries:
            case = convert_entry(entry)
            if case:
                all_cases.append(case)

    if limit:
        all_cases = all_cases[:limit]

    print(f"Converted {len(all_cases)} temporal test cases")
    return all_cases


def main():
    parser = argparse.ArgumentParser(description="Convert SituatedQA to eval format")
    parser.add_argument("--output", default="datasets/situated_qa.json", help="Output path")
    parser.add_argument("--limit", type=int, default=None, help="Max number of test cases")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["dev"],
        choices=["train", "dev", "test"],
        help="Which splits to include (default: dev)",
    )
    args = parser.parse_args()

    cases = convert(args.splits, args.limit)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(cases, f, indent=2)

    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
