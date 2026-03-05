#!/usr/bin/env python3
"""
LLM model escalation experiment.

Hypothesis:
Start with a cheaper model and escalate to an expensive model only when quality drops.
Compare cost against always using the expensive model.

Input JSON format (array):
[
  {
    "id": "session-1",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello"},
      {"role": "assistant", "content": "Hi!"},
      {"role": "user", "content": "Explain recursion simply."}
    ]
  }
]

Notes:
- The script evaluates each user turn in each conversation.
- For each turn it calls both cheap and expensive models.
- It then asks the expensive model to judge cheap response acceptability on 1-5 scale.
- Escalation point is first turn with score < threshold (default 3).
- Cost comparison excludes judge-call cost to reflect deployment policy cost.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _load_dotenv() -> None:
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()


DEFAULTS: Dict[str, Dict[str, Any]] = {
    "openai": {
        "cheap_model": "gpt-4o-mini",
        "expensive_model": "gpt-4o",
        # USD per 1M tokens. Override with CLI flags for latest prices.
        "cheap_input_per_m": 0.15,
        "cheap_output_per_m": 0.60,
        "expensive_input_per_m": 5.00,
        "expensive_output_per_m": 15.00,
    },
    "anthropic": {
        "cheap_model": "claude-haiku-4-5-20251001",
        "expensive_model": "claude-sonnet-4-6",
        # USD per 1M tokens. Override with CLI flags for latest prices.
        "cheap_input_per_m": 0.80,
        "cheap_output_per_m": 4.00,
        "expensive_input_per_m": 3.00,
        "expensive_output_per_m": 15.00,
    },
    "groq": {
        "cheap_model": "llama-3.1-8b-instant",
        "expensive_model": "llama-3.3-70b-versatile",
        # Groq pricing is roughly this (free tier is $0, but using nominal values)
        "cheap_input_per_m": 0.05,
        "cheap_output_per_m": 0.08,
        "expensive_input_per_m": 0.59,
        "expensive_output_per_m": 0.79,
    },
    # Uses the local `claude` CLI binary (Claude Code). No API key required —
    # authentication is handled by Claude Code itself (OAuth / keychain).
    "claude-code": {
        "cheap_model": "claude-haiku-4-5-20251001",
        "expensive_model": "claude-sonnet-4-6",
        "cheap_input_per_m": 0.80,
        "cheap_output_per_m": 4.00,
        "expensive_input_per_m": 3.00,
        "expensive_output_per_m": 15.00,
    },
}


@dataclass
class Usage:
    input_tokens: int
    output_tokens: int


@dataclass
class TurnEval:
    turn_index: int
    cheap_score: int
    cheap_usage: Usage
    expensive_usage: Usage


@dataclass
class SessionEval:
    session_id: str
    user_turn_count: int
    failure_turn: Optional[int]
    turn_evals: List[TurnEval]


def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be an array of conversations.")
    return data


def conversation_id(conversation: Dict[str, Any], idx: int) -> str:
    cid = conversation.get("id")
    return str(cid) if cid is not None else f"session-{idx + 1}"


def extract_messages(conversation: Dict[str, Any]) -> List[Dict[str, str]]:
    messages = conversation.get("messages")
    if not isinstance(messages, list):
        raise ValueError("Each conversation must include a 'messages' list.")

    cleaned: List[Dict[str, str]] = []
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise ValueError(f"messages[{i}] must be an object.")
        role = msg.get("role")
        content = msg.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"messages[{i}].role must be system/user/assistant.")
        if not isinstance(content, str):
            raise ValueError(f"messages[{i}].content must be a string.")
        cleaned.append({"role": role, "content": content})
    return cleaned


def user_turn_indices(messages: List[Dict[str, str]]) -> List[int]:
    return [i for i, m in enumerate(messages) if m["role"] == "user"]


def parse_score(text: str) -> int:
    m = re.search(r"\b([1-5])\b", text)
    if not m:
        raise ValueError(f"Could not parse score 1-5 from judge output: {text!r}")
    return int(m.group(1))


def cost_usd(usage: Usage, in_per_m: float, out_per_m: float) -> float:
    return (usage.input_tokens / 1_000_000.0) * in_per_m + (usage.output_tokens / 1_000_000.0) * out_per_m


class OpenAIAdapter:
    def __init__(self, api_key: str, max_output_tokens: Optional[int] = None):
        from openai import OpenAI  # type: ignore

        self.client = OpenAI(api_key=api_key)
        self.max_output_tokens = max_output_tokens

    def generate(self, model: str, messages: List[Dict[str, str]], temperature: float = 0.2) -> Tuple[str, Usage]:
        kwargs: Dict[str, Any] = dict(model=model, messages=messages, temperature=temperature)
        if self.max_output_tokens is not None:
            kwargs["max_tokens"] = self.max_output_tokens
        resp = self.client.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content or ""
        usage = Usage(
            input_tokens=int(resp.usage.prompt_tokens or 0),
            output_tokens=int(resp.usage.completion_tokens or 0),
        )
        return text, usage


class GroqAdapter:
    def __init__(self, api_key: str, max_output_tokens: Optional[int] = None):
        from openai import OpenAI  # type: ignore

        self.client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)
        self.max_output_tokens = max_output_tokens

    def generate(self, model: str, messages: List[Dict[str, str]], temperature: float = 0.2) -> Tuple[str, Usage]:
        kwargs: Dict[str, Any] = dict(model=model, messages=messages, temperature=temperature)
        if self.max_output_tokens is not None:
            kwargs["max_tokens"] = self.max_output_tokens
        resp = self.client.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content or ""
        usage = Usage(
            input_tokens=int(resp.usage.prompt_tokens or 0),
            output_tokens=int(resp.usage.completion_tokens or 0),
        )
        time.sleep(2.0)
        return text, usage


class AnthropicAdapter:
    def __init__(self, api_key: str, max_output_tokens: int = 1024):
        from anthropic import Anthropic  # type: ignore

        self.client = Anthropic(api_key=api_key)
        self.max_output_tokens = max_output_tokens

    def generate(self, model: str, messages: List[Dict[str, str]], temperature: float = 0.2) -> Tuple[str, Usage]:
        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        api_messages = [{"role": m["role"], "content": m["content"]} for m in messages if m["role"] != "system"]

        kwargs: Dict[str, Any] = dict(
            model=model,
            max_tokens=self.max_output_tokens,
            temperature=temperature,
            messages=api_messages,
        )
        if system_parts:
            kwargs["system"] = "\n\n".join(system_parts)
        resp = self.client.messages.create(**kwargs)

        chunks: List[str] = []
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                chunks.append(getattr(block, "text", ""))

        text = "".join(chunks)
        usage = Usage(
            input_tokens=int(getattr(resp.usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(resp.usage, "output_tokens", 0) or 0),
        )
        return text, usage


class ClaudeCodeAdapter:
    """
    Uses the local `claude` CLI (Claude Code) for generation via:
        claude -p "<prompt>" --output-format json --model <model> --max-turns 1

    The JSON response includes `result` (text) and `usage` (input/output tokens),
    so cost tracking works identically to the SDK-based adapters.

    Notes:
    - No API key required — Claude Code handles auth (OAuth / keychain).
    - Multi-turn history is serialised as plain text in the prompt because
      the CLI's `-p` flag is single-shot. The model understands the ROLE: format.
    - `temperature` is not supported by the CLI; the parameter is accepted but ignored.
    """

    def generate(self, model: str, messages: List[Dict[str, str]], temperature: float = 0.2) -> Tuple[str, Usage]:
        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        non_system = [m for m in messages if m["role"] != "system"]

        # Single user turn → pass content directly; multi-turn → serialise as transcript.
        if len(non_system) == 1:
            prompt = non_system[0]["content"]
        else:
            lines = [f"{m['role'].upper()}: {m['content']}" for m in non_system]
            lines.append("\nRespond only as ASSISTANT to the final USER message above.")
            prompt = "\n".join(lines)

        cmd = [
            "claude", "-p", prompt,
            "--output-format", "json",
            "--model", model,
            "--max-turns", "1",
        ]
        if system_parts:
            cmd += ["--append-system-prompt", "\n\n".join(system_parts)]

        # Strip CLAUDECODE so the CLI can run even when invoked from inside a Claude Code session.
        env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if proc.returncode != 0:
            raise RuntimeError(
                f"claude CLI failed (exit {proc.returncode}): {proc.stderr.strip()}"
            )

        data = json.loads(proc.stdout)
        text = data.get("result", "")
        raw = data.get("usage") or {}
        usage = Usage(
            input_tokens=int(raw.get("input_tokens", 0)),
            output_tokens=int(raw.get("output_tokens", 0)),
        )
        return text, usage


def make_judge_prompt(history: List[Dict[str, str]], cheap_response: str, expensive_response: str) -> List[Dict[str, str]]:
    history_text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in history)
    rubric = (
        "You are grading whether a cheap model response is acceptable for the user's latest turn.\n"
        "Return ONLY an integer 1-5.\n"
        "5 = excellent, 4 = good, 3 = acceptable, 2 = poor, 1 = unacceptable.\n"
        "Acceptable means safe, relevant, and sufficiently helpful for normal use."
    )
    prompt = (
        f"Conversation so far:\n{history_text}\n\n"
        f"Cheap model response:\n{cheap_response}\n\n"
        f"Expensive model response (reference):\n{expensive_response}\n\n"
        "Score the cheap response (1-5). Return only the number."
    )
    return [
        {"role": "system", "content": rubric},
        {"role": "user", "content": prompt},
    ]


def evaluate_session(
    adapter: Any,
    cheap_model: str,
    expensive_model: str,
    messages: List[Dict[str, str]],
    fail_threshold: int,
    verbose: bool = False,
) -> SessionEval:
    turns = user_turn_indices(messages)
    evaluations: List[TurnEval] = []
    failure_turn: Optional[int] = None

    for idx, msg_idx in enumerate(turns, start=1):
        history = messages[: msg_idx + 1]
        user_msg = history[-1]["content"]

        cheap_text, cheap_usage = adapter.generate(cheap_model, history)
        expensive_text, expensive_usage = adapter.generate(expensive_model, history)

        judge_messages = make_judge_prompt(history, cheap_text, expensive_text)
        judge_text, _judge_usage = adapter.generate(expensive_model, judge_messages, temperature=0.0)
        score = parse_score(judge_text)

        evaluations.append(
            TurnEval(
                turn_index=idx,
                cheap_score=score,
                cheap_usage=cheap_usage,
                expensive_usage=expensive_usage,
            )
        )

        if verbose:
            status = "FAIL" if score < fail_threshold else "OK  "
            print(f"  Turn {idx} [{status}] score={score}/5 | user: {user_msg[:60]!r}")
            print(f"    cheap ({cheap_usage.input_tokens}+{cheap_usage.output_tokens} tok): {cheap_text[:80].strip()!r}")
            if score < fail_threshold:
                print(f"    expensive ref: {expensive_text[:80].strip()!r}")

        if failure_turn is None and score < fail_threshold:
            failure_turn = idx

    return SessionEval(
        session_id="",
        user_turn_count=len(turns),
        failure_turn=failure_turn,
        turn_evals=evaluations,
    )


def aggregate_costs(
    sessions: List[SessionEval],
    cheap_in_per_m: float,
    cheap_out_per_m: float,
    expensive_in_per_m: float,
    expensive_out_per_m: float,
) -> Dict[str, float]:
    all_expensive = 0.0
    escalation = 0.0

    for s in sessions:
        for t in s.turn_evals:
            all_expensive += cost_usd(t.expensive_usage, expensive_in_per_m, expensive_out_per_m)

        fail = s.failure_turn
        for t in s.turn_evals:
            if fail is None or t.turn_index < fail:
                escalation += cost_usd(t.cheap_usage, cheap_in_per_m, cheap_out_per_m)
            else:
                escalation += cost_usd(t.expensive_usage, expensive_in_per_m, expensive_out_per_m)

    savings = all_expensive - escalation
    savings_pct = (savings / all_expensive * 100.0) if all_expensive > 0 else 0.0
    return {
        "all_expensive_usd": all_expensive,
        "escalation_usd": escalation,
        "savings_usd": savings,
        "savings_pct": savings_pct,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run LLM escalation cost experiment.")
    parser.add_argument("--provider", choices=["openai", "anthropic", "claude-code", "groq"], required=True)
    parser.add_argument("--input", required=True, help="Path to conversations JSON file.")
    parser.add_argument("--fail-threshold", type=int, default=3, help="Escalate when score < threshold (default: 3).")

    parser.add_argument("--cheap-model", default=None)
    parser.add_argument("--expensive-model", default=None)

    parser.add_argument("--cheap-input-per-m", type=float, default=None)
    parser.add_argument("--cheap-output-per-m", type=float, default=None)
    parser.add_argument("--expensive-input-per-m", type=float, default=None)
    parser.add_argument("--expensive-output-per-m", type=float, default=None)

    parser.add_argument("--max-output-tokens", type=int, default=None,
                        help="Cap output tokens per generation call (default: 1024 for Anthropic, unlimited for OpenAI).")
    parser.add_argument("--verbose", action="store_true", help="Print per-turn scores and snippets.")
    parser.add_argument("--output", default=None, help="Optional path to write full results JSON.")

    args = parser.parse_args()

    defaults = DEFAULTS[args.provider]
    cheap_model = args.cheap_model or defaults["cheap_model"]
    expensive_model = args.expensive_model or defaults["expensive_model"]

    cheap_in = args.cheap_input_per_m if args.cheap_input_per_m is not None else defaults["cheap_input_per_m"]
    cheap_out = args.cheap_output_per_m if args.cheap_output_per_m is not None else defaults["cheap_output_per_m"]
    exp_in = args.expensive_input_per_m if args.expensive_input_per_m is not None else defaults["expensive_input_per_m"]
    exp_out = args.expensive_output_per_m if args.expensive_output_per_m is not None else defaults["expensive_output_per_m"]

    if args.provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Missing OPENAI_API_KEY", file=sys.stderr)
            return 2
        adapter = OpenAIAdapter(api_key, max_output_tokens=args.max_output_tokens)
    elif args.provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("Missing ANTHROPIC_API_KEY", file=sys.stderr)
            return 2
        adapter = AnthropicAdapter(api_key, max_output_tokens=args.max_output_tokens or 1024)
    elif args.provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print("Missing GROQ_API_KEY", file=sys.stderr)
            return 2
        adapter = GroqAdapter(api_key, max_output_tokens=args.max_output_tokens)
    else:  # claude-code
        import shutil
        if not shutil.which("claude"):
            print("'claude' binary not found in PATH. Is Claude Code installed?", file=sys.stderr)
            return 2
        adapter = ClaudeCodeAdapter()
        print("Using Claude Code CLI for generation (no API key required).")

    conversations = load_json(args.input)
    session_results: List[SessionEval] = []

    for i, conv in enumerate(conversations):
        sid = conversation_id(conv, i)
        messages = extract_messages(conv)
        if not user_turn_indices(messages):
            continue

        print(f"\nEvaluating {sid} ({len(user_turn_indices(messages))} user turns)...")
        result = evaluate_session(
            adapter=adapter,
            cheap_model=cheap_model,
            expensive_model=expensive_model,
            messages=messages,
            fail_threshold=args.fail_threshold,
            verbose=args.verbose,
        )
        result.session_id = sid
        session_results.append(result)
        status = f"escalates at turn {result.failure_turn}" if result.failure_turn else "fully handled by cheap model"
        print(f"  -> {status}")

    if not session_results:
        print("No valid sessions with user turns found.", file=sys.stderr)
        return 1

    total_sessions = len(session_results)
    fully_cheap = sum(1 for s in session_results if s.failure_turn is None)
    fully_cheap_pct = fully_cheap / total_sessions * 100.0

    escalation_turns = [s.failure_turn for s in session_results if s.failure_turn is not None]
    avg_escalation_turn = statistics.mean(escalation_turns) if escalation_turns else math.nan

    costs = aggregate_costs(session_results, cheap_in, cheap_out, exp_in, exp_out)

    print("\n=== Experiment Results ===")
    print(f"Provider: {args.provider}")
    print(f"Cheap model: {cheap_model}")
    print(f"Expensive model: {expensive_model}")
    print(f"Sessions: {total_sessions}")
    print(f"% sessions cheap handles fully: {fully_cheap_pct:.2f}% ({fully_cheap}/{total_sessions})")
    if escalation_turns:
        print(f"Average escalation turn: {avg_escalation_turn:.2f}")
    else:
        print("Average escalation turn: N/A (no escalations)")

    print("\nCost comparison (generation only, excludes judge-call cost):")
    print(f"All-expensive cost: ${costs['all_expensive_usd']:.6f}")
    print(f"Escalation cost:   ${costs['escalation_usd']:.6f}")
    print(f"Savings:           ${costs['savings_usd']:.6f} ({costs['savings_pct']:.2f}%)")

    if args.output:
        output_data = {
            "provider": args.provider,
            "cheap_model": cheap_model,
            "expensive_model": expensive_model,
            "fail_threshold": args.fail_threshold,
            "summary": {
                "total_sessions": total_sessions,
                "fully_cheap_sessions": fully_cheap,
                "fully_cheap_pct": fully_cheap_pct,
                "avg_escalation_turn": None if math.isnan(avg_escalation_turn) else avg_escalation_turn,
            },
            "costs": costs,
            "sessions": [
                {
                    "session_id": s.session_id,
                    "user_turn_count": s.user_turn_count,
                    "failure_turn": s.failure_turn,
                    "turns": [
                        {
                            "turn_index": t.turn_index,
                            "cheap_score": t.cheap_score,
                            "cheap_input_tokens": t.cheap_usage.input_tokens,
                            "cheap_output_tokens": t.cheap_usage.output_tokens,
                            "expensive_input_tokens": t.expensive_usage.input_tokens,
                            "expensive_output_tokens": t.expensive_usage.output_tokens,
                        }
                        for t in s.turn_evals
                    ],
                }
                for s in session_results
            ],
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nFull results saved to: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
