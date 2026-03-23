"""Eval harness: run test cases with and without timestamps, store results."""

import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timezone, timedelta

import anthropic

from db import EvalStore, create_tables
from prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_NO_TIME

BEDROCK_MODELS = {
    "claude-opus-4-6": "us.anthropic.claude-opus-4-6-v1",
    "claude-sonnet-4-6": "arn:aws:bedrock:us-east-1:744741211997:inference-profile/us.anthropic.claude-sonnet-4-6",
    "claude-haiku-4-5-20251001": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "claude-3-5-sonnet-20241022": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    "claude-3-5-haiku-20241022": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
}


def get_client():
    has_aws = bool(os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION"))
    has_anthropic_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if has_aws:
        region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
        return anthropic.AnthropicBedrock(aws_region=region), True
    elif has_anthropic_key:
        return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"]), False
    else:
        print("No API configured.", file=sys.stderr)
        sys.exit(1)


def get_model_id(model: str, use_bedrock: bool) -> str:
    if not use_bedrock:
        return model
    return BEDROCK_MODELS.get(model, f"us.anthropic.{model}-v1:0")


def run_single(client, model_id: str, messages: list[dict], system: str) -> str:
    """Run a single prompt through the model."""
    response = client.messages.create(
        model=model_id,
        max_tokens=4096,
        system=system,
        messages=messages,
    )
    return response.content[0].text


def inject_timestamps(test_case: dict) -> list[dict]:
    """Add timestamps to a test case's messages.

    Each test case has a 'messages' list and optional 'time_offsets' (minutes from start).
    If no offsets, defaults to 1-minute spacing.
    """
    messages = test_case["messages"]
    offsets = test_case.get("time_offsets", list(range(len(messages))))
    base_time = datetime.now(timezone.utc).astimezone()

    stamped = []
    for msg, offset in zip(messages, offsets):
        if msg["role"] == "user":
            ts = base_time + timedelta(minutes=offset)
            stamped.append({
                "role": "user",
                "content": f"{msg['content']}\n\n[timestamp: {ts.isoformat()}]",
            })
        else:
            stamped.append(msg)
    return stamped


def score_response(response: str, expected: list[str] | str) -> str:
    """Check if any expected answer appears in the response (case-insensitive)."""
    if not expected:
        return ""
    if isinstance(expected, str):
        expected = [expected]
    response_lower = response.lower()
    for ans in expected:
        if ans.lower() in response_lower:
            return "match"
    return "no_match"


def run_eval(client, model_id: str, test_cases: list[dict],
             store: EvalStore | None, dataset_name: str):
    """Run all test cases in both modes and compare."""
    eval_id = str(uuid.uuid4())
    results = []
    match_with = 0
    match_without = 0
    scored = 0

    for i, tc in enumerate(test_cases):
        case_id = tc.get("id", f"case_{i}")
        description = tc.get("description", "")
        expected = tc.get("expected_answer", "")
        print(f"\n[{i+1}/{len(test_cases)}] {case_id}: {description}")

        # With timestamps
        stamped_messages = inject_timestamps(tc)
        resp_with = run_single(client, model_id, stamped_messages, SYSTEM_PROMPT)

        # Without timestamps
        plain_messages = [m for m in tc["messages"]]
        resp_without = run_single(client, model_id, plain_messages, SYSTEM_PROMPT_NO_TIME)

        # Score if expected answer provided
        score_with = score_response(resp_with, expected)
        score_without = score_response(resp_without, expected)

        if expected:
            scored += 1
            if score_with == "match":
                match_with += 1
            if score_without == "match":
                match_without += 1

        result = {
            "test_case_id": case_id,
            "description": description,
            "response_with_time": resp_with,
            "response_without_time": resp_without,
            "expected_answer": expected,
            "score_with_time": score_with,
            "score_without_time": score_without,
        }
        results.append(result)

        if store:
            store.save_result(
                eval_id=eval_id,
                test_case_id=case_id,
                prompt=json.dumps(tc["messages"]),
                response_with_time=resp_with,
                response_without_time=resp_without,
                dataset_name=dataset_name,
                model=model_id,
                expected_answer=json.dumps(expected) if isinstance(expected, list) else expected,
                score=f"with:{score_with}|without:{score_without}",
            )

        status = ""
        if expected:
            status = f" [with:{score_with}, without:{score_without}]"
        print(f"  WITH time:    {resp_with[:100]}...{status}")
        print(f"  WITHOUT time: {resp_without[:100]}...")

    # Summary
    if scored > 0:
        print(f"\n--- Scoring Summary ---")
        print(f"With timestamps:    {match_with}/{scored} ({match_with/scored*100:.0f}%)")
        print(f"Without timestamps: {match_without}/{scored} ({match_without/scored*100:.0f}%)")
        delta = match_with - match_without
        print(f"Delta:              {'+' if delta >= 0 else ''}{delta} cases")

    # Save locally as JSON
    out_path = f"eval_results_{eval_id[:8]}.json"
    with open(out_path, "w") as f:
        json.dump({
            "eval_id": eval_id,
            "model": model_id,
            "dataset": dataset_name,
            "results": results,
            "summary": {
                "total": len(results),
                "scored": scored,
                "match_with_time": match_with,
                "match_without_time": match_without,
            },
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")
    if store:
        print(f"DynamoDB eval_id: {eval_id}")

    return results


def load_test_cases(path: str) -> list[dict]:
    """Load test cases from a JSON file."""
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Eval harness for time-aware chatbot")
    parser.add_argument("test_file", help="Path to JSON file with test cases")
    parser.add_argument("--model", default="claude-sonnet-4-6", choices=list(BEDROCK_MODELS.keys()))
    parser.add_argument("--dataset", default="", help="Dataset name label for this eval run")
    parser.add_argument("--no-db", action="store_true", help="Skip DynamoDB storage")
    args = parser.parse_args()

    from env import load_env
    load_env()

    client, use_bedrock = get_client()
    model_id = get_model_id(args.model, use_bedrock)
    test_cases = load_test_cases(args.test_file)

    dataset_name = args.dataset or os.path.splitext(os.path.basename(args.test_file))[0]
    store = None if args.no_db else EvalStore()

    print(f"Running {len(test_cases)} test cases (model: {model_id}, dataset: {dataset_name})")
    run_eval(client, model_id, test_cases, store, dataset_name)


if __name__ == "__main__":
    main()
