"""
Experiment 1: Capability Gap Analysis
======================================
Hypothesis: Small models fail PREDICTABLY on specific task types, not randomly.

What it does:
- Sends every test prompt to all 3 model tiers (cheap, mid, expensive)
- Uses the mid/expensive model as judge to score each response
- Maps which skills each tier can/cannot handle

Expected output:
- A capability matrix showing pass/fail rates per skill per model tier
- Evidence of whether failures cluster by skill (predictable) or are random

Runtime: ~15-20 minutes (180 API calls + 120 judge calls)
Rate limits: Groq free tier is ~30 req/min, script auto-throttles
"""

import argparse
import json
from pathlib import Path
from tqdm import tqdm
from config import (
    get_client, MODELS, SKILL_TAXONOMY, TEST_PROMPTS,
    call_model, judge_response, save_results, rate_limit_pause,
)

RESULTS_FILE = "experiment_1_capability_gaps.json"


def load_existing() -> list:
    """Load previously saved results so we can resume/append."""
    filepath = Path(__file__).parent / "results" / RESULTS_FILE
    if filepath.exists():
        with open(filepath) as f:
            return json.load(f)
    return []


def run_experiment(skills_filter: list = None):
    client = get_client()
    tiers_to_test = ["cheap", "mid"]

    # Load any previously completed results
    existing = load_existing()
    done_ids = {r["prompt_id"] for r in existing}

    # Filter prompts
    prompts = [
        (i, t) for i, t in enumerate(TEST_PROMPTS)
        if i not in done_ids and (skills_filter is None or t["skill"] in skills_filter)
    ]

    if not prompts:
        print("Nothing to run — all selected skills already completed.")
        print_summary(existing)
        return existing

    skills_label = ", ".join(skills_filter) if skills_filter else "all"
    print(f"Experiment 1: Capability Gap Analysis")
    print(f"Skills: {skills_label}")
    print(f"Prompts to run: {len(prompts)} ({len(done_ids)} already done)")
    print(f"API calls: {len(prompts) * len(tiers_to_test) * 2} (response + judge each)\n")

    results = list(existing)

    for i, test in tqdm(prompts, desc="Prompts"):
        prompt_result = {
            "prompt_id": i,
            "skill": test["skill"],
            "difficulty": test["difficulty"],
            "prompt": test["prompt"],
            "tiers": {},
        }

        for tier_name in tiers_to_test:
            model = MODELS[tier_name]

            response = call_model(client, model["id"], test["prompt"])
            rate_limit_pause(2)

            if response["error"]:
                prompt_result["tiers"][tier_name] = {
                    "response": None,
                    "latency_s": response["latency_s"],
                    "error": response["error"],
                    "judge": {"score": 0, "pass": False, "reason": "Model error"},
                }
                continue

            judgment = judge_response(client, test["prompt"], response["content"])
            rate_limit_pause(2)

            prompt_result["tiers"][tier_name] = {
                "response": response["content"][:500],
                "latency_s": response["latency_s"],
                "input_tokens": response["input_tokens"],
                "output_tokens": response["output_tokens"],
                "judge": judgment,
            }

        results.append(prompt_result)

    # Sort by prompt_id so order is consistent
    results.sort(key=lambda r: r["prompt_id"])
    save_results(results, RESULTS_FILE)
    print_summary(results)
    return results


def print_summary(results):
    """Print a quick summary of capability gaps."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1 RESULTS: Capability Gap Analysis")
    print("=" * 70)

    # Group by skill
    skills = {}
    for r in results:
        skill = r["skill"]
        if skill not in skills:
            skills[skill] = {"cheap_pass": 0, "cheap_total": 0, "mid_pass": 0, "mid_total": 0}

        for tier in ["cheap", "mid"]:
            if tier in r["tiers"]:
                skills[skill][f"{tier}_total"] += 1
                if r["tiers"][tier].get("judge", {}).get("pass", False):
                    skills[skill][f"{tier}_pass"] += 1

    print(f"\n{'Skill':<30} {'Cheap Pass%':>12} {'Mid Pass%':>12} {'Gap':>8}")
    print("-" * 70)

    for skill, counts in sorted(skills.items()):
        cheap_pct = (counts["cheap_pass"] / max(counts["cheap_total"], 1)) * 100
        mid_pct = (counts["mid_pass"] / max(counts["mid_total"], 1)) * 100
        gap = mid_pct - cheap_pct
        marker = " <<<" if gap > 30 else ""
        print(f"{skill:<30} {cheap_pct:>10.0f}% {mid_pct:>10.0f}% {gap:>+7.0f}%{marker}")

    # Overall
    total_cheap_pass = sum(s["cheap_pass"] for s in skills.values())
    total_cheap = sum(s["cheap_total"] for s in skills.values())
    total_mid_pass = sum(s["mid_pass"] for s in skills.values())
    total_mid = sum(s["mid_total"] for s in skills.values())

    print("-" * 70)
    print(f"{'OVERALL':<30} {total_cheap_pass/max(total_cheap,1)*100:>10.0f}% {total_mid_pass/max(total_mid,1)*100:>10.0f}%")
    print(f"\nCheap model adequate for {total_cheap_pass}/{total_cheap} prompts ({total_cheap_pass/max(total_cheap,1)*100:.0f}%)")
    print(f"Mid model adequate for {total_mid_pass}/{total_mid} prompts ({total_mid_pass/max(total_mid,1)*100:.0f}%)")
    print(f"\nSkills with >30% gap marked with <<<  — these are routing opportunities")


if __name__ == "__main__":
    all_skills = list(SKILL_TAXONOMY.keys())

    parser = argparse.ArgumentParser(description="Experiment 1: Capability Gap Analysis")
    parser.add_argument(
        "--skills", nargs="+", choices=all_skills, default=None,
        metavar="SKILL",
        help=f"Skills to run (default: all). Choices: {', '.join(all_skills)}",
    )
    parser.add_argument("--list", action="store_true", help="List all skill names and exit.")
    args = parser.parse_args()

    if args.list:
        print("Available skills:")
        for s in all_skills:
            print(f"  {s}")
    else:
        run_experiment(skills_filter=args.skills)
