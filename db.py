"""DynamoDB storage for conversations and eval results."""

import os
import uuid
from datetime import datetime, timezone
from decimal import Decimal

import boto3
from boto3.dynamodb.conditions import Key

CONVERSATIONS_TABLE = "time-chatbot-conversations"
EVALS_TABLE = "time-chatbot-evals"


def _get_dynamodb():
    """Get DynamoDB resource using same AWS credentials as Bedrock."""
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    return boto3.resource("dynamodb", region_name=region)


def _decimal_to_native(obj):
    """Convert DynamoDB Decimal types to int/float for JSON serialization."""
    if isinstance(obj, Decimal):
        return int(obj) if obj == int(obj) else float(obj)
    if isinstance(obj, dict):
        return {k: _decimal_to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decimal_to_native(i) for i in obj]
    return obj


def create_tables():
    """Create DynamoDB tables if they don't exist."""
    import botocore.exceptions

    dynamodb = _get_dynamodb()

    try:
        existing = [t.name for t in dynamodb.tables.all()]
    except botocore.exceptions.ClientError as e:
        code = e.response["Error"]["Code"]
        msg = e.response["Error"]["Message"]
        if code in ("AccessDeniedException", "UnrecognizedClientException"):
            print(f"\n[PERMISSION ERROR] Cannot list DynamoDB tables.")
            print(f"  AWS error: {code} — {msg}")
            print(f"  Your IAM role/credentials likely lack dynamodb:ListTables permission.")
            print(f"  Ask your AWS admin to attach a policy with these actions:")
            print(f"    dynamodb:ListTables, dynamodb:CreateTable,")
            print(f"    dynamodb:PutItem, dynamodb:Query, dynamodb:Scan\n")
        elif code == "ExpiredTokenException":
            print(f"\n[AUTH ERROR] AWS session token has expired.")
            print(f"  Refresh your STS credentials and update .env\n")
        else:
            print(f"\n[AWS ERROR] {code}: {msg}\n")
        raise SystemExit(1)

    for table_name, schema in [
        (CONVERSATIONS_TABLE, {
            "keys": [
                {"AttributeName": "session_id", "KeyType": "HASH"},
                {"AttributeName": "message_index", "KeyType": "RANGE"},
            ],
            "attrs": [
                {"AttributeName": "session_id", "AttributeType": "S"},
                {"AttributeName": "message_index", "AttributeType": "N"},
            ],
        }),
        (EVALS_TABLE, {
            "keys": [
                {"AttributeName": "eval_id", "KeyType": "HASH"},
                {"AttributeName": "test_case_id", "KeyType": "RANGE"},
            ],
            "attrs": [
                {"AttributeName": "eval_id", "AttributeType": "S"},
                {"AttributeName": "test_case_id", "AttributeType": "S"},
            ],
        }),
    ]:
        if table_name in existing:
            print(f"Table already exists: {table_name}")
            continue

        try:
            dynamodb.create_table(
                TableName=table_name,
                KeySchema=schema["keys"],
                AttributeDefinitions=schema["attrs"],
                BillingMode="PAY_PER_REQUEST",
            )
            print(f"Created table: {table_name}")
        except botocore.exceptions.ClientError as e:
            code = e.response["Error"]["Code"]
            msg = e.response["Error"]["Message"]
            if code == "AccessDeniedException":
                print(f"\n[PERMISSION ERROR] Cannot create table '{table_name}'.")
                print(f"  AWS error: {code} — {msg}")
                print(f"  Your IAM role/credentials lack dynamodb:CreateTable permission.")
                print(f"  Ask your AWS admin to grant it, or have them create the tables manually.\n")
            elif code == "ExpiredTokenException":
                print(f"\n[AUTH ERROR] AWS session token has expired.")
                print(f"  Refresh your STS credentials and update .env\n")
            else:
                print(f"\n[AWS ERROR] Failed to create '{table_name}': {code} — {msg}\n")
            raise SystemExit(1)


class ConversationStore:
    """Store and retrieve chat sessions."""

    def __init__(self):
        self.dynamodb = _get_dynamodb()
        self.table = self.dynamodb.Table(CONVERSATIONS_TABLE)
        self.session_id = str(uuid.uuid4())
        self._index = 0

    def save_message(self, role: str, content: str, model: str, has_timestamps: bool):
        """Save a single message to the conversation."""
        self.table.put_item(Item={
            "session_id": self.session_id,
            "message_index": self._index,
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "has_timestamps": has_timestamps,
        })
        self._index += 1

    def get_session(self, session_id: str) -> list[dict]:
        """Retrieve all messages for a session, ordered."""
        resp = self.table.query(
            KeyConditionExpression=Key("session_id").eq(session_id),
        )
        items = sorted(resp["Items"], key=lambda x: int(x["message_index"]))
        return [_decimal_to_native(i) for i in items]

    def list_sessions(self) -> list[dict]:
        """List all sessions with summary info (scan — fine at small scale)."""
        resp = self.table.scan()
        items = resp["Items"]
        # Paginate if needed
        while "LastEvaluatedKey" in resp:
            resp = self.table.scan(ExclusiveStartKey=resp["LastEvaluatedKey"])
            items.extend(resp["Items"])

        # Group by session_id
        sessions = {}
        for item in items:
            sid = item["session_id"]
            if sid not in sessions:
                sessions[sid] = {
                    "session_id": sid,
                    "model": item.get("model", ""),
                    "has_timestamps": item.get("has_timestamps", False),
                    "message_count": 0,
                    "first_timestamp": item.get("timestamp", ""),
                    "last_timestamp": item.get("timestamp", ""),
                }
            sessions[sid]["message_count"] += 1
            ts = item.get("timestamp", "")
            if ts < sessions[sid]["first_timestamp"]:
                sessions[sid]["first_timestamp"] = ts
            if ts > sessions[sid]["last_timestamp"]:
                sessions[sid]["last_timestamp"] = ts

        result = sorted(sessions.values(), key=lambda x: x["last_timestamp"], reverse=True)
        return [_decimal_to_native(s) for s in result]


class EvalStore:
    """Store eval run results."""

    def __init__(self):
        self.dynamodb = _get_dynamodb()
        self.table = self.dynamodb.Table(EVALS_TABLE)

    def save_result(self, eval_id: str, test_case_id: str, prompt: str,
                    response_with_time: str, response_without_time: str,
                    dataset_name: str = "", model: str = "",
                    expected_answer: str = "", score: str = "", notes: str = ""):
        """Save a single eval comparison."""
        self.table.put_item(Item={
            "eval_id": eval_id,
            "test_case_id": test_case_id,
            "prompt": prompt,
            "response_with_time": response_with_time,
            "response_without_time": response_without_time,
            "dataset_name": dataset_name,
            "model": model,
            "expected_answer": expected_answer,
            "score": score,
            "notes": notes,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def get_eval_run(self, eval_id: str) -> list[dict]:
        """Retrieve all results for an eval run."""
        resp = self.table.query(
            KeyConditionExpression=Key("eval_id").eq(eval_id),
        )
        return [_decimal_to_native(i) for i in resp["Items"]]

    def list_eval_runs(self) -> list[dict]:
        """List all eval runs with summary info (scan — fine at small scale)."""
        resp = self.table.scan()
        items = resp["Items"]
        while "LastEvaluatedKey" in resp:
            resp = self.table.scan(ExclusiveStartKey=resp["LastEvaluatedKey"])
            items.extend(resp["Items"])

        # Group by eval_id
        runs = {}
        for item in items:
            eid = item["eval_id"]
            if eid not in runs:
                runs[eid] = {
                    "eval_id": eid,
                    "dataset_name": item.get("dataset_name", ""),
                    "model": item.get("model", ""),
                    "case_count": 0,
                    "timestamp": item.get("timestamp", ""),
                }
            runs[eid]["case_count"] += 1
            ts = item.get("timestamp", "")
            if ts > runs[eid]["timestamp"]:
                runs[eid]["timestamp"] = ts

        result = sorted(runs.values(), key=lambda x: x["timestamp"], reverse=True)
        return [_decimal_to_native(r) for r in result]
