#!/usr/bin/env python3
"""
Experiment E2: Escalation Signal Comparison - Self-Rating
========================================================
Goal: Test if the cheap model can accurately "self-correct" or flag 
its own low-quality responses by providing a confidence score.

Process:
1. Cheap model generates response.
2. Cheap model is asked to rate its own response 1-5.
3. Compare self-rating vs. Expensive Judge score.
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

def get_self_rating(adapter, model, history, response_text):
    history_text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in history)
    prompt = (
        f"Review the conversation and your latest response below.\n\n"
        f"CONVERSATION HISTORY:\n{history_text}\n\n"
        f"YOUR RESPONSE:\n{response_text}\n\n"
        "Rate your confidence that your response is completely correct, follows all constraints, and fully answers the user. "
        "Use a scale of 1-5:\n"
        "5 = Extremely confident (Correct and complete)\n"
        "4 = Confident (Mostly correct, minor gaps possible)\n"
        "3 = Moderately confident (Acceptable but could be better)\n"
        "2 = Low confidence (Likely contains errors or misses constraints)\n"
        "1 = Not confident (Incorrect or fails to answer)\n\n"
        "Return ONLY the integer 1-5."
    )
    rating_msg = [{"role": "user", "content": prompt}]
    rating_text, _ = adapter.generate(model, rating_msg, temperature=0.0)
    try:
        match = re.search(r"\b([1-5])\b", rating_text)
        return int(match.group(1)) if match else 3
    except:
        return 3

def main():
    parser = argparse.ArgumentParser(description="Run E2 Self-Rating Experiment.")
    parser.add_argument("--provider", choices=["openai", "anthropic", "claude-code", "groq"], required=True)
    parser.add_argument("--input", required=True, help="Path to conversations JSON file.")
    parser.add_argument("--fail-threshold", type=int, default=3, help="Judge threshold")
    parser.add_argument("--signal-threshold", type=int, default=3, help="Self-rating threshold (escalate if < this)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output", default="results/experiment_e2_self_rating.json")
    
    args = parser.parse_args()

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
    
    y_true = [] # 1 if judge < fail_threshold
    y_pred = [] # 1 if self_rating < signal_threshold
    
    results = []

    print(f"Running Self-Rating Signal Test on {args.input}...")
    
    for i, conv in enumerate(conversations):
        messages = extract_messages(conv)
        turns = user_turn_indices(messages)
        
        for turn_idx in turns:
            history = messages[: turn_idx + 1]
            
            # 1. Get cheap model response
            cheap_text, _ = adapter.generate(cheap_model, history)
            
            # 2. Get Self-Rating (Our Signal)
            self_rating = get_self_rating(adapter, cheap_model, history, cheap_text)
            signal_triggered = self_rating < args.signal_threshold
            
            # 3. Get Ground Truth (Judge score)
            expensive_text, _ = adapter.generate(expensive_model, history)
            judge_messages = make_judge_prompt(history, cheap_text, expensive_text)
            judge_text, _ = adapter.generate(expensive_model, judge_messages, temperature=0.0)
            score = parse_score(judge_text)
            
            actual_escalation = 1 if score < args.fail_threshold else 0
            predicted_escalation = 1 if signal_triggered else 0
            
            y_true.append(actual_escalation)
            y_pred.append(predicted_escalation)
            
            res = {
                "session": conv.get("id", f"s{i}"),
                "turn": turn_idx,
                "judge_score": score,
                "self_rating": self_rating,
                "triggered": signal_triggered,
                "actual": actual_escalation,
                "correct": signal_triggered == actual_escalation,
                "cheap_response_preview": cheap_text[:200]
            }
            results.append(res)
            
            if args.verbose:
                status = "MATCH" if res["correct"] else "MISMATCH"
                print(f"Turn {turn_idx}: Judge={score}, Self={self_rating} -> {status}")
                if not res["correct"]:
                    print(f"  (False {'Pos' if signal_triggered else 'Neg'})")

    # Calculate Metrics
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    
    accuracy = (tp + tn) / len(y_true) if y_true else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\n=== Self-Rating Signal Performance ===")
    print(f"Accuracy:  {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall:    {recall:.2%}")
    print(f"F1 Score:  {f1:.2%}")
    print(f"\nConfusion Matrix:")
    print(f"TP: {tp} | FP: {fp}")
    print(f"FN: {fn} | TN: {tn}")

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
