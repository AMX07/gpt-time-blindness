"""Time-aware CLI chatbot using AWS Bedrock (Claude)."""

import argparse
import os
import sys
from datetime import datetime, timezone

import anthropic

from db import ConversationStore, create_tables
from prompts import get_system_prompt

# Cross-region inference profile IDs for Bedrock
BEDROCK_MODELS = {
    "claude-opus-4-6": "us.anthropic.claude-opus-4-6-v1",
    "claude-sonnet-4-6": "arn:aws:bedrock:us-east-1:744741211997:inference-profile/us.anthropic.claude-sonnet-4-6",
    "claude-haiku-4-5-20251001": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "claude-3-5-sonnet-20241022": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    "claude-3-5-haiku-20241022": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
}

DEFAULT_MODEL = "claude-opus-4-6"

# Extended thinking budget (tokens) — matches claude.ai's max thinking
THINKING_BUDGET = 10000


def get_client():
    """Get Anthropic client, auto-detecting Bedrock vs direct API."""
    has_aws = bool(os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION"))
    has_anthropic_key = bool(os.environ.get("ANTHROPIC_API_KEY"))

    if has_aws:
        region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
        return anthropic.AnthropicBedrock(aws_region=region), True
    elif has_anthropic_key:
        return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"]), False
    else:
        print(
            "No API configured. Set either:\n"
            "  - AWS_REGION + AWS credentials for Bedrock, or\n"
            "  - ANTHROPIC_API_KEY for direct Anthropic API",
            file=sys.stderr,
        )
        sys.exit(1)


def get_model_id(model: str, use_bedrock: bool) -> str:
    """Convert model name to appropriate ID for the client."""
    if not use_bedrock:
        return model
    return BEDROCK_MODELS.get(model, f"us.anthropic.{model}-v1:0")


def stamp(text: str) -> str:
    """Append an ISO 8601 timestamp to a message."""
    now = datetime.now(timezone.utc).astimezone()
    return f"{text}\n\n[timestamp: {now.isoformat()}]"


def extract_response_text(response) -> str:
    """Extract text from response, skipping thinking blocks."""
    for block in response.content:
        if block.type == "text":
            return block.text
    return ""


def chat_loop(client, model_id: str, use_timestamps: bool,
              use_thinking: bool, store: ConversationStore | None):
    """Run the interactive chat loop."""
    messages = []
    mode = "time-aware" if use_timestamps else "baseline (no time)"
    thinking_label = "thinking ON" if use_thinking else "thinking OFF"

    print(f"Time-aware chatbot (model: {model_id}, mode: {mode}, {thinking_label})")
    if store:
        print(f"Session: {store.session_id}")
    print("Type 'quit' or Ctrl-C to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Bye!")
            break

        content = stamp(user_input) if use_timestamps else user_input
        messages.append({"role": "user", "content": content})

        if store:
            store.save_message("user", content, model_id, use_timestamps)

        # Build system prompt fresh each call (so currentDateTime is accurate)
        system = get_system_prompt(use_timestamps)

        try:
            kwargs = {
                "model": model_id,
                "max_tokens": 16384,
                "system": system,
                "messages": messages,
            }

            if use_thinking:
                kwargs["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": THINKING_BUDGET,
                }

            response = client.messages.create(**kwargs)
        except anthropic.APIError as e:
            print(f"\nAPI error: {e}\n", file=sys.stderr)
            messages.pop()
            continue

        assistant_text = extract_response_text(response)

        # For conversation history, only include the text content (not thinking)
        messages.append({"role": "assistant", "content": assistant_text})

        if store:
            store.save_message("assistant", assistant_text, model_id, use_timestamps)

        print(f"\nAssistant: {assistant_text}\n")


def main():
    parser = argparse.ArgumentParser(description="Time-aware CLI chatbot")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=list(BEDROCK_MODELS.keys()),
        help=f"Claude model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--no-timestamps",
        action="store_true",
        help="Disable time awareness (baseline mode for A/B comparison)",
    )
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="Disable extended thinking",
    )
    parser.add_argument(
        "--no-db",
        action="store_true",
        help="Skip DynamoDB storage (for quick local testing)",
    )
    parser.add_argument(
        "--setup-db",
        action="store_true",
        help="Create DynamoDB tables and exit",
    )
    args = parser.parse_args()

    from env import load_env
    load_env()

    if args.setup_db:
        create_tables()
        return

    client, use_bedrock = get_client()
    model_id = get_model_id(args.model, use_bedrock)

    store = None if args.no_db else ConversationStore()

    chat_loop(client, model_id, not args.no_timestamps, not args.no_thinking, store)


if __name__ == "__main__":
    main()
