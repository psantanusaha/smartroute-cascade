#!/usr/bin/env python3
"""
Experiment E2: Escalation Signal Comparison - Prompt Classifier
==============================================================
Goal: Test if pre-classifying the prompt's difficulty can accurately 
predict when escalation is needed, BEFORE the cheap model even answers.

Process:
1. Cheap model classifies the user prompt into a skill category.
2. If the skill is "Hard" (not in the 'cheap' tier), trigger escalation.
3. Compare this pre-emptive decision vs. the actual Judge score of the cheap response.
"""

import os
import json
import re
import argparse
import sys
from typing import List, Dict, Any, Tuple
from pathlib import Path

# Reuse adapters and config
from escalation_experiment import (
    AnthropicAdapter, OpenAIAdapter, ClaudeCodeAdapter, GroqAdapter,
    load_json, extract_messages, user_turn_indices,
    make_judge_prompt, parse_score, DEFAULTS
)
from config import SKILL_TAXONOMY

# Classifier prompt from experiment_3
CLASSIFIER_PROMPT = """Classify the following user prompt into exactly ONE of these skill categories:

Categories:
- factual_qa: Simple lookups, definitions, general knowledge
- summarization: Condensing text, extracting key points
- basic_code: Single-function scripts, syntax help, simple bugs
- creative_simple: Short emails, social posts, basic rewriting
- multi_step_reasoning: Chain-of-thought math, logic puzzles, word problems
- complex_code: System design, multi-file code, design patterns, debugging
- data_analysis: Statistical reasoning, SQL, interpreting datasets
- nuanced_creative: Voice, tone, literary devices, long-form narrative
- multi_constraint: Itineraries, schedules, optimization under constraints
- formal_reasoning: Mathematical proofs, formal logic, derivations
- agentic: Multi-tool workflows, pipeline design, automation
- ambiguous_open: Philosophical, ethical, highly subjective questions

Respond with ONLY the category slug, nothing else.

User prompt: {prompt}"""

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

def get_predicted_skill(adapter, model, prompt):
    msg = [{"role": "user", "content": CLASSIFIER_PROMPT.format(prompt=prompt)}]
    res_text, _ = adapter.generate(model, msg, temperature=0.0)
    
    response = res_text.strip().lower().replace("-", "_").replace(" ", "_")
    valid_skills = set(SKILL_TAXONOMY.keys())
    
    if response in valid_skills:
        return response
    
    # Fuzzy match
    for skill in valid_skills:
        if skill in response or response in skill:
            return skill
    return "ambiguous_open"

def main():
    parser = argparse.ArgumentParser(description="Run E2 Prompt Classifier Experiment.")
    parser.add_argument("--provider", choices=["openai", "anthropic", "claude-code", "groq"], default="groq")
    parser.add_argument("--cheap-provider", choices=["openai", "anthropic", "claude-code", "groq"], default=None)
    parser.add_argument("--expensive-provider", choices=["openai", "anthropic", "claude-code", "groq"], default=None)
    parser.add_argument("--input", required=True, help="Path to conversations JSON file.")
    parser.add_argument("--fail-threshold", type=int, default=3)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output", default="results/experiment_e2_classifier.json")
    
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    cheap_p = args.cheap_provider or args.provider
    exp_p = args.expensive_provider or args.provider

    def get_adapter(provider):
        if provider == "openai":
            return OpenAIAdapter(os.getenv("OPENAI_API_KEY"))
        elif provider == "anthropic":
            return AnthropicAdapter(os.getenv("ANTHROPIC_API_KEY"))
        elif provider == "groq":
            return GroqAdapter(os.getenv("GROQ_API_KEY"))
        else:
            return ClaudeCodeAdapter()

    cheap_adapter = get_adapter(cheap_p)
    exp_adapter = get_adapter(exp_p)

    cheap_model = DEFAULTS[cheap_p]["cheap_model"]
    expensive_model = DEFAULTS[exp_p]["expensive_model"]

    conversations = load_json(args.input)
    
    # Load existing results if they exist (Resume Capability)
    existing_results = []
    if os.path.exists(args.output):
        try:
            with open(args.output, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    existing_results = data
                else:
                    existing_results = data.get("results", [])
                print(f"Resuming from {len(existing_results)} existing turn results...")
        except:
            pass
    
    processed_keys = {f"{r['session']}_{r['turn']}" for r in existing_results}
    results = existing_results
    
    print(f"Running Prompt Classifier Signal Test (Cheap: {cheap_p}, Exp: {exp_p}) on {args.input}...")
    
    for i, conv in enumerate(conversations):
        sid = conv.get("id", f"s{i}")
        messages = extract_messages(conv)
        turns = user_turn_indices(messages)
        
        for turn_idx in turns:
            key = f"{sid}_{turn_idx}"
            if key in processed_keys:
                continue
                
            history = messages[: turn_idx + 1]
            user_prompt = history[-1]["content"]
            
            # 1. Pre-classify (Our Signal)
            predicted_skill = get_predicted_skill(cheap_adapter, cheap_model, user_prompt)
            expected_tier = SKILL_TAXONOMY.get(predicted_skill, {}).get("expected_min_tier", "mid")
            
            # Trigger escalation if the expected tier is NOT 'cheap'
            signal_triggered = expected_tier != "cheap"
            
            # 2. Get cheap model response (to judge)
            cheap_text, _ = cheap_adapter.generate(cheap_model, history)
            
            # 3. Get Ground Truth (Judge score)
            expensive_text, _ = exp_adapter.generate(expensive_model, history)
            judge_messages = make_judge_prompt(history, cheap_text, expensive_text)
            judge_text, _ = exp_adapter.generate(expensive_model, judge_messages, temperature=0.0)
            score = parse_score(judge_text)
            
            actual_escalation = 1 if score < args.fail_threshold else 0
            predicted_escalation = 1 if signal_triggered else 0
            
            res = {
                "session": sid,
                "turn": turn_idx,
                "judge_score": score,
                "predicted_skill": predicted_skill,
                "expected_tier": expected_tier,
                "triggered": signal_triggered,
                "actual": actual_escalation,
                "correct": signal_triggered == actual_escalation
            }
            results.append(res)
            
            # Save intermediate results after each turn to prevent data loss
            with open(args.output, "w") as f:
                json.dump({"results": results}, f, indent=2)
            
            if args.verbose:
                status = "MATCH" if res["correct"] else "MISMATCH"
                print(f"Turn {turn_idx}: Skill={predicted_skill} ({expected_tier}), Judge={score} -> {status}")
                if not res["correct"]:
                    print(f"  (False {'Pos' if signal_triggered else 'Neg'})")

    # Final Metrics Calculation
    y_true = [r["actual"] for r in results]
    y_pred = [r["triggered"] for r in results]

    if not y_true:
        print("No turns to process.")
        return

    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    
    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\n=== Prompt Classifier Signal Performance ===")
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
