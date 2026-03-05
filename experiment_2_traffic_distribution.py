"""
Experiment 2: Traffic Distribution Analysis
=============================================
Hypothesis: 60-80% of typical LLM requests can be handled by the cheapest model.

What it does:
- Uses a REALISTIC distribution of prompt types (weighted toward simple tasks)
- Tests each on the cheap model only
- Judges quality to determine what % could stay cheap

Why this matters:
- If most real traffic is simple, routing saves a LOT of money
- If real traffic is mostly complex, routing has limited value

The distribution below is based on typical developer/enterprise LLM usage patterns:
  ~40% simple Q&A, summarization, basic rewrites
  ~30% basic code, simple analysis, creative tasks
  ~20% multi-step reasoning, complex code
  ~10% formal reasoning, agentic, multi-constraint

Runtime: ~8-10 minutes (50 API calls + 50 judge calls)
"""

import random
from tqdm import tqdm
from config import (
    get_client, MODELS, TEST_PROMPTS,
    call_model, judge_response, save_results, rate_limit_pause,
)

# Realistic traffic distribution weights by skill
TRAFFIC_WEIGHTS = {
    "factual_qa":           0.20,
    "summarization":        0.12,
    "creative_simple":      0.10,
    "basic_code":           0.15,
    "data_analysis":        0.08,
    "multi_step_reasoning": 0.10,
    "nuanced_creative":     0.05,
    "complex_code":         0.08,
    "multi_constraint":     0.04,
    "ambiguous_open":       0.04,
    "formal_reasoning":     0.02,
    "agentic":              0.02,
}

# Additional "real world" prompts that represent everyday usage
REALISTIC_PROMPTS = [
    {"skill": "factual_qa", "prompt": "What's the difference between a list and a tuple in Python?"},
    {"skill": "factual_qa", "prompt": "How do I center a div in CSS?"},
    {"skill": "factual_qa", "prompt": "What does HTTP 403 mean?"},
    {"skill": "factual_qa", "prompt": "What is a foreign key in SQL?"},
    {"skill": "factual_qa", "prompt": "What's the time complexity of binary search?"},
    {"skill": "summarization", "prompt": "What does CORS stand for and why does it exist? Keep it brief."},
    {"skill": "summarization", "prompt": "Explain Docker containers vs VMs in simple terms."},
    {"skill": "creative_simple", "prompt": "Write a git commit message for: fixed the login button not redirecting after OAuth callback."},
    {"skill": "creative_simple", "prompt": "Help me write a Slack message asking my team to update their Jira tickets by EOD."},
    {"skill": "creative_simple", "prompt": "Make this error message more user friendly: 'Error: NullPointerException at line 42'"},
    {"skill": "basic_code", "prompt": "Write a Python function to remove duplicates from a list while preserving order."},
    {"skill": "basic_code", "prompt": "How do I read environment variables in Node.js?"},
    {"skill": "basic_code", "prompt": "Write a SQL query to count orders by status."},
    {"skill": "data_analysis", "prompt": "What's a good way to handle missing values in a pandas DataFrame?"},
    {"skill": "multi_step_reasoning", "prompt": "I have a list of intervals [[1,3],[2,6],[8,10],[15,18]]. How would I merge overlapping intervals? Walk me through the logic."},
    {"skill": "complex_code", "prompt": "I need to implement pagination for a REST API in FastAPI. The endpoint should support cursor-based pagination with filters. Show me the approach."},
    {"skill": "multi_constraint", "prompt": "I need to deploy a Python app. Requirements: must handle 1000 req/s, budget under $200/month, needs a Postgres DB, and must be in us-west-2. What's my best option?"},
    {"skill": "formal_reasoning", "prompt": "Prove that log base 2 of 3 is irrational."},
    {"skill": "agentic", "prompt": "Design a CI/CD pipeline that runs tests, checks coverage, does a canary deploy to 5% of traffic, monitors error rates for 10 minutes, and either promotes to 100% or rolls back."},
    {"skill": "ambiguous_open", "prompt": "When should I use NoSQL vs SQL? I keep getting different answers."},
]


def build_sample(n=50):
    """Build a sample of n prompts following the realistic traffic distribution."""
    all_prompts = TEST_PROMPTS + REALISTIC_PROMPTS

    # Group prompts by skill
    by_skill = {}
    for p in all_prompts:
        skill = p["skill"]
        if skill not in by_skill:
            by_skill[skill] = []
        by_skill[skill].append(p)

    # Sample according to weights
    sample = []
    for skill, weight in TRAFFIC_WEIGHTS.items():
        count = max(1, round(n * weight))
        pool = by_skill.get(skill, [])
        if pool:
            chosen = random.choices(pool, k=count)
            sample.extend(chosen)

    random.shuffle(sample)
    return sample[:n]


def run_experiment(n=50):
    client = get_client()
    sample = build_sample(n)

    print(f"Experiment 2: Traffic Distribution Analysis")
    print(f"Testing {len(sample)} prompts on CHEAP model only")
    print(f"Estimated time: {len(sample) * 4 // 60} minutes\n")

    # Show distribution
    skill_counts = {}
    for p in sample:
        skill_counts[p["skill"]] = skill_counts.get(p["skill"], 0) + 1
    print("Sample distribution:")
    for skill, count in sorted(skill_counts.items(), key=lambda x: -x[1]):
        print(f"  {skill:<30} {count:>3} ({count/len(sample)*100:.0f}%)")
    print()

    results = []
    cheap_model = MODELS["cheap"]["id"]

    for i, test in enumerate(tqdm(sample, desc="Testing cheap model")):
        # Get cheap model response
        response = call_model(client, cheap_model, test["prompt"])
        rate_limit_pause(2)

        if response["error"]:
            results.append({
                "prompt": test["prompt"],
                "skill": test["skill"],
                "cheap_adequate": False,
                "score": 0,
                "reason": f"Error: {response['error']}",
                "latency_s": response["latency_s"],
            })
            continue

        # Judge it
        judgment = judge_response(client, test["prompt"], response["content"])
        rate_limit_pause(2)

        results.append({
            "prompt": test["prompt"],
            "skill": test["skill"],
            "cheap_adequate": judgment.get("pass", False),
            "score": judgment.get("score", 0),
            "reason": judgment.get("reason", ""),
            "latency_s": response["latency_s"],
            "response_preview": response["content"][:200] if response["content"] else None,
        })

    save_results(results, "experiment_2_traffic_distribution.json")
    print_summary(results)
    return results


def print_summary(results):
    """Print traffic distribution analysis."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2 RESULTS: Traffic Distribution Analysis")
    print("=" * 70)

    adequate = sum(1 for r in results if r["cheap_adequate"])
    total = len(results)

    print(f"\nCheap model adequate for: {adequate}/{total} ({adequate/total*100:.1f}%)")

    # Breakdown by skill
    skills = {}
    for r in results:
        skill = r["skill"]
        if skill not in skills:
            skills[skill] = {"pass": 0, "fail": 0, "avg_score": []}
        if r["cheap_adequate"]:
            skills[skill]["pass"] += 1
        else:
            skills[skill]["fail"] += 1
        skills[skill]["avg_score"].append(r["score"])

    print(f"\n{'Skill':<30} {'Pass':>6} {'Fail':>6} {'Rate':>8} {'Avg Score':>10}")
    print("-" * 70)

    for skill, data in sorted(skills.items()):
        total_skill = data["pass"] + data["fail"]
        rate = data["pass"] / total_skill * 100
        avg = sum(data["avg_score"]) / len(data["avg_score"])
        emoji = "✅" if rate >= 80 else "⚠️" if rate >= 50 else "❌"
        print(f"{skill:<30} {data['pass']:>6} {data['fail']:>6} {rate:>7.0f}% {avg:>9.1f}  {emoji}")

    # Cost projection
    print(f"\n--- Cost Projection ---")
    print(f"If 100% of traffic goes to mid tier: $1.00 (baseline)")
    cheap_pct = adequate / total
    # Rough cost ratio: 8B is ~10x cheaper than 70B
    cost_with_routing = cheap_pct * 0.1 + (1 - cheap_pct) * 1.0
    savings = (1 - cost_with_routing) * 100
    print(f"With routing ({cheap_pct*100:.0f}% stays cheap): ${cost_with_routing:.2f} ({savings:.0f}% savings)")


if __name__ == "__main__":
    run_experiment(n=50)
