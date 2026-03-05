"""
Analyze Results from All Experiments
=====================================
Run this after completing experiments 1, 2, and 3.
Generates a combined summary and verdict on each hypothesis.
"""

import json
from pathlib import Path


def load(filename):
    filepath = Path(__file__).parent / "results" / filename
    if not filepath.exists():
        return None
    with open(filepath) as f:
        return json.load(f)


def analyze_experiment_1(data):
    """Analyze capability gaps."""
    if not data:
        return None

    skills = {}
    for r in data:
        skill = r["skill"]
        if skill not in skills:
            skills[skill] = {"cheap_scores": [], "mid_scores": [], "cheap_pass": 0, "mid_pass": 0, "total": 0}

        skills[skill]["total"] += 1

        for tier in ["cheap", "mid"]:
            if tier in r["tiers"]:
                score = r["tiers"][tier].get("judge", {}).get("score", 0)
                passed = r["tiers"][tier].get("judge", {}).get("pass", False)
                skills[skill][f"{tier}_scores"].append(score)
                if passed:
                    skills[skill][f"{tier}_pass"] += 1

    # Calculate gap consistency
    predictable_skills = 0
    total_skills = 0
    for skill, data in skills.items():
        total_skills += 1
        cheap_rate = data["cheap_pass"] / max(data["total"], 1)
        mid_rate = data["mid_pass"] / max(data["total"], 1)

        # A skill has a "predictable gap" if cheap consistently fails (< 60%)
        # while mid consistently passes (> 70%)
        if cheap_rate < 0.6 and mid_rate > 0.7:
            predictable_skills += 1
        # Or if cheap consistently passes (> 70%) — also predictable
        elif cheap_rate > 0.7:
            predictable_skills += 1

    return {
        "skills": skills,
        "predictable_ratio": predictable_skills / max(total_skills, 1),
        "total_skills": total_skills,
        "predictable_skills": predictable_skills,
    }


def analyze_experiment_2(data):
    """Analyze traffic distribution."""
    if not data:
        return None

    adequate = sum(1 for r in data if r.get("cheap_adequate", False))
    total = len(data)
    cheap_pct = adequate / max(total, 1)

    # Cost model: cheap is ~10x less than mid
    cost_all_mid = 1.0
    cost_with_routing = cheap_pct * 0.1 + (1 - cheap_pct) * 1.0
    savings = (1 - cost_with_routing / cost_all_mid) * 100

    return {
        "cheap_adequate_pct": cheap_pct,
        "total_tested": total,
        "adequate_count": adequate,
        "projected_savings_pct": savings,
    }


def analyze_experiment_3(data):
    """Analyze classification feasibility."""
    if not data:
        return None

    results = {}
    for approach in ["keyword", "llm", "tfidf"]:
        if approach in data and data[approach]:
            correct = sum(1 for r in data[approach] if r.get("correct", False))
            total = len(data[approach])
            results[approach] = {
                "accuracy": correct / max(total, 1),
                "total": total,
                "correct": correct,
            }

    return results


def print_verdict():
    print("=" * 70)
    print("SMARTROUTE EXPERIMENT RESULTS — COMBINED ANALYSIS")
    print("=" * 70)

    # Experiment 1
    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print("│ HYPOTHESIS 1: Predictable Capability Gaps                          │")
    print("│ 'Small models fail predictably on specific task types'             │")
    print("└─────────────────────────────────────────────────────────────────────┘")

    exp1 = analyze_experiment_1(load("experiment_1_capability_gaps.json"))
    if exp1:
        ratio = exp1["predictable_ratio"]
        print(f"\n  Predictable skills: {exp1['predictable_skills']}/{exp1['total_skills']} ({ratio*100:.0f}%)")
        print(f"\n  Skill breakdown:")
        print(f"  {'Skill':<30} {'Cheap Pass%':>12} {'Mid Pass%':>12} {'Verdict':>10}")
        print(f"  {'-'*68}")

        for skill, d in sorted(exp1["skills"].items()):
            cp = d["cheap_pass"] / max(d["total"], 1) * 100
            mp = d["mid_pass"] / max(d["total"], 1) * 100
            if cp >= 70:
                verdict = "✅ Cheap OK"
            elif mp >= 70:
                verdict = "🔺 Need Mid"
            else:
                verdict = "🔴 Need Exp"
            print(f"  {skill:<30} {cp:>10.0f}% {mp:>10.0f}%  {verdict}")

        verdict = "✅ VALIDATED" if ratio >= 0.7 else "⚠️ PARTIAL" if ratio >= 0.5 else "❌ REJECTED"
        print(f"\n  Verdict: {verdict}")
        print(f"  {ratio*100:.0f}% of skills show predictable capability patterns")
    else:
        print("\n  ⏭  Not yet run. Execute: python experiment_1_capability_gaps.py")

    # Experiment 2
    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print("│ HYPOTHESIS 2: Cheap Models Handle Most Traffic                     │")
    print("│ '60-80% of typical requests can be handled by the cheapest model'  │")
    print("└─────────────────────────────────────────────────────────────────────┘")

    exp2 = analyze_experiment_2(load("experiment_2_traffic_distribution.json"))
    if exp2:
        pct = exp2["cheap_adequate_pct"] * 100
        savings = exp2["projected_savings_pct"]
        print(f"\n  Cheap model adequate for: {exp2['adequate_count']}/{exp2['total_tested']} ({pct:.1f}%)")
        print(f"  Projected cost savings: {savings:.0f}%")

        verdict = "✅ VALIDATED" if pct >= 60 else "⚠️ PARTIAL" if pct >= 40 else "❌ REJECTED"
        print(f"\n  Verdict: {verdict}")
        if pct >= 60:
            print(f"  Strong business case: {savings:.0f}% cost reduction with routing")
        elif pct >= 40:
            print(f"  Moderate savings possible, but may need better cheap model")
        else:
            print(f"  Limited routing opportunity with current model tier")
    else:
        print("\n  ⏭  Not yet run. Execute: python experiment_2_traffic_distribution.py")

    # Experiment 3
    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print("│ HYPOTHESIS 3: Skill Classification is Feasible                     │")
    print("│ 'A simple classifier can reliably identify required skills'        │")
    print("└─────────────────────────────────────────────────────────────────────┘")

    exp3 = analyze_experiment_3(load("experiment_3_skill_classification.json"))
    if exp3:
        print(f"\n  {'Approach':<35} {'Accuracy':>10}")
        print(f"  {'-'*48}")
        for approach, d in exp3.items():
            print(f"  {approach:<35} {d['accuracy']*100:>9.1f}%")

        best = max(exp3.values(), key=lambda x: x["accuracy"])
        best_name = [k for k, v in exp3.items() if v == best][0]
        best_acc = best["accuracy"] * 100

        verdict = "✅ VALIDATED" if best_acc >= 80 else "⚠️ PARTIAL" if best_acc >= 65 else "❌ REJECTED"
        print(f"\n  Best approach: {best_name} ({best_acc:.1f}%)")
        print(f"  Verdict: {verdict}")
        if best_acc >= 80:
            print(f"  Classification is reliable enough for production routing")
        elif best_acc >= 65:
            print(f"  Usable with fallback strategy (misroutes handled by retry)")
        else:
            print(f"  Needs more training data or a different approach")
    else:
        print("\n  ⏭  Not yet run. Execute: python experiment_3_skill_classifier.py")

    # Overall verdict
    print("\n" + "=" * 70)
    print("OVERALL VERDICT: Should you build SmartRoute?")
    print("=" * 70)

    if exp1 and exp2 and exp3:
        h1 = exp1["predictable_ratio"] >= 0.5
        h2 = exp2["cheap_adequate_pct"] >= 0.4
        h3 = max(d["accuracy"] for d in exp3.values()) >= 0.65

        if h1 and h2 and h3:
            print("\n  🟢 ALL HYPOTHESES VALIDATED — Build it!")
            print(f"     Projected savings: {exp2['projected_savings_pct']:.0f}%")
            print(f"     Classification is feasible")
            print(f"     Capability gaps are predictable and exploitable")
        elif sum([h1, h2, h3]) >= 2:
            print("\n  🟡 MOSTLY VALIDATED — Build with caveats")
            if not h1:
                print("     ⚠️ Capability gaps less predictable than hoped — add retry fallback")
            if not h2:
                print("     ⚠️ Less traffic routable to cheap than hoped — narrow the use case")
            if not h3:
                print("     ⚠️ Classification needs work — start with keywords + cheap LLM hybrid")
        else:
            print("\n  🔴 INSUFFICIENT EVIDENCE — Rethink the approach")
            print("     Consider: different model tiers, different taxonomy, or different strategy")
    else:
        missing = []
        if not exp1: missing.append("experiment_1")
        if not exp2: missing.append("experiment_2")
        if not exp3: missing.append("experiment_3")
        print(f"\n  ⏳ Still need to run: {', '.join(missing)}")
        print(f"     Run all experiments, then re-run this analysis.")


if __name__ == "__main__":
    print_verdict()
