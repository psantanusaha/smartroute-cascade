"""
Quick Smoke Test — Run a mini version of all 3 experiments
============================================================
Use this to verify your setup works before running the full experiments.
Tests 6 prompts (1 per difficulty tier) instead of 60.

Runtime: ~3-5 minutes
"""

from config import get_client, MODELS, call_model, judge_response, rate_limit_pause
from experiment_3_skill_classifier import classify_by_keywords, classify_by_llm
import json

QUICK_PROMPTS = [
    # Should be easy for cheap model
    {"skill": "factual_qa", "prompt": "What is the capital of Australia?", "expected_tier": "cheap"},
    {"skill": "basic_code", "prompt": "Write a Python function to check if a string is a palindrome.", "expected_tier": "cheap"},
    # Should need mid model
    {"skill": "multi_step_reasoning", "prompt": "I have 3 boxes. Box A has 2 red and 3 blue balls. Box B has 4 red and 1 blue. Box C has 1 red and 4 blue. I pick a box at random, then draw a ball. It's red. What's the probability it came from Box B?", "expected_tier": "mid"},
    {"skill": "complex_code", "prompt": "Design a Python class for an LRU cache with O(1) get and put operations. Include type hints and handle edge cases.", "expected_tier": "mid"},
    # Should need expensive model
    {"skill": "formal_reasoning", "prompt": "Prove that the square root of 2 is irrational using proof by contradiction.", "expected_tier": "expensive"},
    {"skill": "multi_constraint", "prompt": "Plan a 3-day Tokyo itinerary for a family with a toddler. Constraints: budget under $200/day, no more than 2 subway rides per day, must include one cultural site and one park daily, hotel is in Shinjuku.", "expected_tier": "expensive"},
]


def run():
    client = get_client()
    print("SmartRoute Quick Smoke Test")
    print("=" * 60)

    for test in QUICK_PROMPTS:
        print(f"\n{'─' * 60}")
        print(f"Skill: {test['skill']} | Expected tier: {test['expected_tier']}")
        print(f"Prompt: {test['prompt'][:80]}...")

        # 1. Classify (keyword)
        kw_class = classify_by_keywords(test["prompt"])
        print(f"\n  Keyword classifier → {kw_class} {'✅' if kw_class == test['skill'] else '❌'}")

        # 2. Classify (LLM)
        llm_class = classify_by_llm(client, test["prompt"])
        rate_limit_pause(2)
        print(f"  LLM classifier    → {llm_class} {'✅' if llm_class == test['skill'] else '❌'}")

        # 3. Test on cheap model
        print(f"\n  Testing cheap model ({MODELS['cheap']['label']})...")
        response = call_model(client, MODELS["cheap"]["id"], test["prompt"])
        rate_limit_pause(2)

        if response["error"]:
            print(f"  ❌ Error: {response['error']}")
            continue

        print(f"  Response preview: {response['content'][:150]}...")
        print(f"  Latency: {response['latency_s']}s")

        # 4. Judge
        judgment = judge_response(client, test["prompt"], response["content"])
        rate_limit_pause(2)

        score = judgment.get("score", 0)
        passed = judgment.get("pass", False)
        reason = judgment.get("reason", "")
        emoji = "✅" if passed else "❌"
        print(f"  Judge: {emoji} score={score}/5 — {reason}")

        if passed and test["expected_tier"] != "cheap":
            print(f"  💡 Cheap model handled this! Expected to need {test['expected_tier']}")
        elif not passed and test["expected_tier"] == "cheap":
            print(f"  ⚠️ Cheap model failed on what should be an easy task")

    print(f"\n{'=' * 60}")
    print("Smoke test complete! If you saw results above, your setup works.")
    print("Next steps:")
    print("  python experiment_3_skill_classifier.py  (5 min)")
    print("  python experiment_1_capability_gaps.py   (20 min)")
    print("  python experiment_2_traffic_distribution.py (10 min)")
    print("  python analyze_results.py")


if __name__ == "__main__":
    run()
