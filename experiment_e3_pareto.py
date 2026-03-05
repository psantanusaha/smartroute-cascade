#!/usr/bin/env python3
"""
Experiment E3: The Cost-Quality Frontier (Pareto Curve) - Split Provider Mode
===========================================================================
Goal: Map the relationship between cost savings and response quality
while splitting expensive calls between multiple providers (e.g. Groq + Anthropic)
to stay under rate limits.
"""

import os
import json
import argparse
import sys
from typing import List, Dict, Any, Tuple
from pathlib import Path

from escalation_experiment import (
    GroqAdapter, AnthropicAdapter, OpenAIAdapter, ClaudeCodeAdapter,
    load_json, extract_messages, user_turn_indices,
    make_judge_prompt, parse_score, DEFAULTS, cost_usd, Usage
)
from config import SKILL_TAXONOMY
from experiment_e2_classifier import get_predicted_skill

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

def get_adapter(provider):
    if provider == "openai":
        return OpenAIAdapter(os.getenv("OPENAI_API_KEY"))
    elif provider == "anthropic":
        return AnthropicAdapter(os.getenv("ANTHROPIC_API_KEY"))
    elif provider == "groq":
        return GroqAdapter(os.getenv("GROQ_API_KEY"))
    else:
        return ClaudeCodeAdapter()

def main():
    parser = argparse.ArgumentParser(description="Run E3 Pareto Experiment with Split Providers.")
    parser.add_argument("--cheap-provider", default="groq")
    parser.add_argument("--expensive-providers", default="groq,anthropic", help="Comma-separated providers (e.g. groq,anthropic)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output", default="results/experiment_e3_pareto.json")
    
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    cheap_adapter = get_adapter(args.cheap_provider)
    cheap_model = DEFAULTS[args.cheap_provider]["cheap_model"]
    
    # Setup multiple expensive adapters
    exp_provider_names = args.expensive_providers.split(",")
    exp_adapters = {p: get_adapter(p) for p in exp_provider_names}
    exp_models = {p: DEFAULTS[p]["expensive_model"] for p in exp_provider_names}

    all_convs = load_json("conversations.json") + load_json("conversations_hard.json")
    
    # Resume logic
    results_raw = []
    if os.path.exists(args.output):
        try:
            with open(args.output, "r") as f:
                data = json.load(f)
                results_raw = data.get("raw_data", [])
                print(f"Resuming from {len(results_raw)} existing turn results...")
        except:
            pass
    
    processed_keys = {f"{r['session']}_{r['turn']}" for r in results_raw}
    
    # Counter for round-robin
    call_idx = len(results_raw)

    print(f"Running Experiment E3 (Cheap: {args.cheap_provider}, Exp: {args.expensive_providers})")

    for i, conv in enumerate(all_convs):
        sid = conv.get("id", f"s{i}")
        messages = extract_messages(conv)
        turns = user_turn_indices(messages)
        
        for turn_idx in turns:
            key = f"{sid}_{turn_idx}"
            if key in processed_keys:
                continue
                
            # Pick expensive provider for this turn
            p_name = exp_provider_names[call_idx % len(exp_provider_names)]
            exp_adapter = exp_adapters[p_name]
            exp_model = exp_models[p_name]
            
            history = messages[: turn_idx + 1]
            user_prompt = history[-1]["content"]
            
            # 1. Signal (Pre-Classifier)
            predicted_skill = get_predicted_skill(cheap_adapter, cheap_model, user_prompt)
            expected_tier = SKILL_TAXONOMY.get(predicted_skill, {}).get("expected_min_tier", "mid")
            classifier_trigger = expected_tier != "cheap"

            # 2. Get Responses
            cheap_text, cheap_usage = cheap_adapter.generate(cheap_model, history)
            expensive_text, expensive_usage = exp_adapter.generate(exp_model, history)
            
            # 3. Get Judge Score
            judge_messages = make_judge_prompt(history, cheap_text, expensive_text)
            judge_text, _ = exp_adapter.generate(exp_model, judge_messages, temperature=0.0)
            score = parse_score(judge_text)
            
            results_raw.append({
                "session": sid,
                "turn": turn_idx,
                "score": score,
                "classifier_trigger": classifier_trigger,
                "exp_provider_used": p_name,
                "cheap_usage": {"input": cheap_usage.input_tokens, "output": cheap_usage.output_tokens},
                "expensive_usage": {"input": expensive_usage.input_tokens, "output": expensive_usage.output_tokens},
                "cheap_p": args.cheap_provider,
                "exp_p": p_name
            })
            
            call_idx += 1
            with open(args.output, "w") as f:
                json.dump({"raw_data": results_raw}, f, indent=2)
            
            if args.verbose:
                print(f"  {sid} Turn {turn_idx}: Score={score}, Classifier={classifier_trigger} (via {p_name})")

    # --- Pareto Sweep ---
    thresholds = [1, 2, 3, 4, 5]
    frontier = []

    # Baseline: What if we ALWAYS used the most expensive model (Anthropic Sonnet)?
    # We use Anthropic as the cost baseline because it's a fixed known price.
    baseline_p = "anthropic"
    base_in = DEFAULTS[baseline_p]["expensive_input_per_m"]
    base_out = DEFAULTS[baseline_p]["expensive_output_per_m"]

    total_expensive_cost = 0.0
    for r in results_raw:
        u = Usage(r["expensive_usage"]["input"], r["expensive_usage"]["output"])
        total_expensive_cost += cost_usd(u, base_in, base_out)

    for T in thresholds:
        cascade_cost = 0.0
        total_score = 0.0
        
        for r in results_raw:
            # Reconstruct usage
            u_cheap = Usage(r["cheap_usage"]["input"], r["cheap_usage"]["output"])
            u_exp = Usage(r["expensive_usage"]["input"], r["expensive_usage"]["output"])
            
            # Get specific pricing for the providers used in this result
            c_p = r["cheap_p"]
            e_p = r["exp_p"]
            
            if r["classifier_trigger"]:
                cascade_cost += cost_usd(u_exp, DEFAULTS[e_p]["expensive_input_per_m"], DEFAULTS[e_p]["expensive_output_per_m"])
                total_score += 5 
            else:
                cascade_cost += cost_usd(u_cheap, DEFAULTS[c_p]["cheap_input_per_m"], DEFAULTS[c_p]["cheap_output_per_m"])
                total_score += r["score"]

        avg_quality = total_score / len(results_raw)
        # Savings calculated against the constant baseline
        savings_pct = (1 - (cascade_cost / total_expensive_cost)) * 100.0 if total_expensive_cost > 0 else 0
        
        frontier.append({
            "threshold": T,
            "savings_pct": savings_pct,
            "avg_quality": avg_quality,
            "cost_usd": cascade_cost
        })

    print("\n=== Experiment E3: Cost-Quality Frontier (Split Provider) ===")
    print(f"{'Threshold':<10} | {'Savings %':<12} | {'Avg Quality (1-5)':<15}")
    print("-" * 45)
    for f in frontier:
        print(f"{f['threshold']:<10} | {f['savings_pct']:>11.2f}% | {f['avg_quality']:>15.2f}")

    with open(args.output, "w") as f:
        json.dump({"frontier": frontier, "raw_data": results_raw}, f, indent=2)
    print(f"\nFinal results saved to: {args.output}")

if __name__ == "__main__":
    main()
