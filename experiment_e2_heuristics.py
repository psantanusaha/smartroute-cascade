#!/usr/bin/env python3
"""
Experiment E2: Escalation Signal Comparison - Heuristics
========================================================
Goal: Test if simple, zero-cost heuristics on the cheap model's response 
can accurately predict when escalation is needed (judge score < threshold).

Heuristics tested:
1. Refusal/Hedging keywords ("I'm sorry", "I cannot", "I'm not sure")
2. Response length (too short)
"""

import os
import json
import re
import argparse
import sys
from typing import List, Dict, Any, Tuple
from pathlib import Path

# Reuse adapters from escalation_experiment.py
from escalation_experiment import (
    AnthropicAdapter, OpenAIAdapter, ClaudeCodeAdapter, GroqAdapter,
    load_json, extract_messages, user_turn_indices,
    make_judge_prompt, parse_score, DEFAULTS
)

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

# --- Heuristic Logic ---

REFUSAL_PATTERNS = [
    r"i'm sorry",
    r"i am sorry",
    r"i cannot",
    r"i am unable",
    r"i don't have",
    r"i do not have",
    r"as an ai",
    r"my knowledge cutoff",
    r"i'm not sure",
    r"i am not sure",
    r"it's unclear",
    r"i don't know",
    r"i do not know",
    r"policy",
    r"restricted",
]

def check_heuristics(text: str, min_length: int = 20) -> Dict[str, bool]:
    text_lower = text.lower()
    
    # 1. Refusal/Hedging check
    has_refusal = any(re.search(pattern, text_lower) for pattern in REFUSAL_PATTERNS)
    
    # 2. Length check
    is_too_short = len(text.strip()) < min_length
    
    return {
        "refusal": has_refusal,
        "too_short": is_too_short,
        "any": has_refusal or is_too_short
    }

def main():
    parser = argparse.ArgumentParser(description="Run E2 Heuristics Experiment.")
    parser.add_argument("--provider", choices=["openai", "anthropic", "claude-code", "groq"], required=True)
    parser.add_argument("--input", required=True, help="Path to conversations JSON file.")
    parser.add_argument("--fail-threshold", type=int, default=3)
    parser.add_argument("--min-length", type=int, default=20)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output", default="results/experiment_e2_heuristics.json")
    
    args = parser.parse_args()

    # Ensure results directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    defaults = DEFAULTS[args.provider]
    cheap_model = defaults["cheap_model"]
    expensive_model = defaults["expensive_model"]

    if args.provider == "openai":
        adapter = OpenAIAdapter(os.getenv("OPENAI_API_KEY"))
    elif args.provider == "anthropic":
        adapter = AnthropicAdapter(os.getenv("ANTHROPIC_API_KEY"))
    elif args.provider == "groq":
        adapter = GroqAdapter(os.getenv("GROQ_API_KEY"))
    else:
        adapter = ClaudeCodeAdapter()

    conversations = load_json(args.input)
    
    y_true = [] # 1 if escalated (score < threshold), 0 otherwise
    y_pred = [] # 1 if heuristic triggered, 0 otherwise
    
    results = []

    print(f"Running Heuristic Signal Test on {args.input}...")
    
    for i, conv in enumerate(conversations):
        messages = extract_messages(conv)
        turns = user_turn_indices(messages)
        
        for turn_idx in turns:
            history = messages[: turn_idx + 1]
            
            # 1. Get cheap model response
            cheap_text, _ = adapter.generate(cheap_model, history)
            
            # 2. Check heuristics (Our Signal)
            signals = check_heuristics(cheap_text, min_length=args.min_length)
            heuristic_triggered = signals["any"]
            
            # 3. Get Ground Truth (Judge score)
            expensive_text, _ = adapter.generate(expensive_model, history)
            judge_messages = make_judge_prompt(history, cheap_text, expensive_text)
            judge_text, _ = adapter.generate(expensive_model, judge_messages, temperature=0.0)
            score = parse_score(judge_text)
            
            actual_escalation = 1 if score < args.fail_threshold else 0
            predicted_escalation = 1 if heuristic_triggered else 0
            
            y_true.append(actual_escalation)
            y_pred.append(predicted_escalation)
            
            res = {
                "session": conv.get("id", f"s{i}"),
                "turn": turn_idx,
                "score": score,
                "heuristic": signals,
                "triggered": heuristic_triggered,
                "actual": actual_escalation,
                "correct": heuristic_triggered == actual_escalation,
                "cheap_response_preview": cheap_text[:200]
            }
            results.append(res)
            
            if args.verbose:
                status = "MATCH" if res["correct"] else "MISMATCH"
                print(f"Turn {turn_idx}: Score={score}, Heuristic={heuristic_triggered} -> {status}")
                if not res["correct"]:
                    print(f"  Text: {cheap_text[:100]}...")

    # Calculate Metrics
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    
    accuracy = (tp + tn) / len(y_true) if y_true else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\n=== Heuristic Signal Performance ===")
    print(f"Accuracy:  {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall:    {recall:.2%}")
    print(f"F1 Score:  {f1:.2%}")
    print(f"\nConfusion Matrix:")
    print(f"TP: {tp} | FP: {fp}")
    print(f"FN: {fn} | TN: {tn}")

    # Save to file
    with open(args.output, "w") as f:
        json.dump({
            "metrics": {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
            },
            "results": results
        }, f, indent=2)
    print(f"\nFull results captured in: {args.output}")

if __name__ == "__main__":
    main()
