"""Convert LoCoMo temporal questions to standard eval format.

Source: https://github.com/snap-research/locomo
Format: JSON with conversation sessions + QA pairs
Temporal questions are category 2.

Usage:
    python -m converters.locomo --output datasets/locomo_temporal.json --limit 50
"""

import argparse
import json
import re
import urllib.request
from pathlib import Path

LOCOMO_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
CACHE_DIR = Path(__file__).parent.parent / "datasets" / ".cache"


def download_data() -> Path:
    """Download LoCoMo dataset if not cached."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / "locomo10.json"

    if cache_path.exists():
        print("  Using cached locomo10.json")
        return cache_path

    print(f"  Downloading from {LOCOMO_URL}")
    urllib.request.urlretrieve(LOCOMO_URL, cache_path)
    return cache_path


def extract_sessions(conv: dict) -> list[dict]:
    """Extract session timestamps and dialogue from a conversation object."""
    sessions = []

    # Find all session keys (session_1, session_2, etc.)
    session_keys = sorted(
        [k for k in conv.keys() if re.match(r"session_\d+$", k)],
        key=lambda k: int(k.split("_")[1]),
    )

    for sk in session_keys:
        num = sk.split("_")[1]
        date_key = f"session_{num}_date_time"
        timestamp = conv.get(date_key, "")

        turns = []
        for turn in conv[sk]:
            turns.append({
                "speaker": turn.get("speaker", ""),
                "text": turn.get("text", ""),
                "dia_id": turn.get("dia_id", ""),
            })

        sessions.append({
            "session_num": int(num),
            "timestamp": timestamp,
            "turns": turns,
        })

    return sessions


def build_conversation_history(sessions: list[dict], evidence_ids: list[str]) -> list[dict]:
    """Build a message history from sessions, including relevant evidence turns.

    We include the full conversation up to and including the evidence turns,
    with timestamps marking each session boundary.
    """
    # Find which sessions contain evidence
    evidence_sessions = set()
    for eid in evidence_ids:
        # Evidence format is "D1:3" meaning dialogue 1, turn 3
        match = re.match(r"D(\d+):(\d+)", eid)
        if match:
            evidence_sessions.add(int(match.group(1)))

    messages = []
    for session in sessions:
        # Add session timestamp as context
        if session["timestamp"]:
            session_marker = f"[Session {session['session_num']} — {session['timestamp']}]"
        else:
            session_marker = f"[Session {session['session_num']}]"

        for i, turn in enumerate(session["turns"]):
            # Alternate speakers as user/assistant
            role = "user" if i % 2 == 0 else "assistant"
            content = turn["text"]

            # Prepend session marker to first turn of each session
            if i == 0:
                content = f"{session_marker}\n{content}"

            messages.append({"role": role, "content": content})

    return messages


def convert_entry(conv_data: dict, qa: dict, conv_index: int, qa_index: int) -> dict:
    """Convert a single LoCoMo temporal QA to our eval format."""
    conv = conv_data["conversation"]
    sessions = extract_sessions(conv)
    evidence_ids = qa.get("evidence", [])

    # Build conversation history
    history = build_conversation_history(sessions, evidence_ids)

    # Add the temporal question as the final user message
    history.append({"role": "user", "content": qa["question"]})

    # Generate time offsets from session timestamps
    # (simplified: just use session index as offset since actual timestamps are dates)
    time_offsets = []
    session_idx = 0
    for msg in history:
        if "[Session" in msg.get("content", ""):
            session_idx += 1
        time_offsets.append(session_idx * 1440)  # days in minutes
    # Last message (the question) gets current time
    time_offsets[-1] = (session_idx + 1) * 1440

    answer = qa.get("answer", "")
    if isinstance(answer, (int, float)):
        answer = str(answer)

    return {
        "id": f"locomo_{conv_index}_q{qa_index}",
        "description": f"Temporal QA: {qa['question'][:80]}",
        "messages": history,
        "time_offsets": time_offsets,
        "expected_answer": [answer] if isinstance(answer, str) else answer,
        "metadata": {
            "source": "locomo",
            "category": qa.get("category", 0),
            "evidence": evidence_ids,
            "speaker_a": conv.get("speaker_a", ""),
            "speaker_b": conv.get("speaker_b", ""),
        },
    }


def convert(limit: int | None = None) -> list[dict]:
    """Download and convert LoCoMo temporal questions."""
    print("Downloading LoCoMo dataset...")
    path = download_data()

    with open(path) as f:
        data = json.load(f)

    all_cases = []

    for conv_idx, entry in enumerate(data):
        qa_list = entry.get("qa", [])
        conv_data = entry

        # Filter for temporal questions (category 2)
        temporal_qs = [q for q in qa_list if q.get("category") == 2]

        for qa_idx, qa in enumerate(temporal_qs):
            case = convert_entry(conv_data, qa, conv_idx, qa_idx)
            all_cases.append(case)

    if limit:
        all_cases = all_cases[:limit]

    print(f"Converted {len(all_cases)} temporal test cases from LoCoMo")
    return all_cases


def main():
    parser = argparse.ArgumentParser(description="Convert LoCoMo temporal QA to eval format")
    parser.add_argument("--output", default="datasets/locomo_temporal.json", help="Output path")
    parser.add_argument("--limit", type=int, default=None, help="Max number of test cases")
    args = parser.parse_args()

    cases = convert(args.limit)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(cases, f, indent=2)

    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
