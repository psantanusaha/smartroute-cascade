import json
from collections import defaultdict

def main():
    try:
        with open("results/experiment_e3_pareto.json") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Results file not found.")
        return

    turns = data.get("raw_data", [])
    skills_data = defaultdict(list)

    for turn in turns:
        # Some results might have predicted_skill, others might not
        # If not, we can infer from session name or just skip
        skill = turn.get("predicted_skill")
        if not skill:
            # Infer from session ID if possible
            sid = turn.get("session", "")
            if "algorithms" in sid: skill = "complex_code"
            elif "clinical" in sid: skill = "data_analysis"
            elif "quant" in sid: skill = "multi_constraint"
            elif "logic" in sid: skill = "multi_step_reasoning"
            elif "collab" in sid: skill = "complex_code"
            else: skill = "routine_qa"

        skills_data[skill].append(turn["classifier_trigger"])

    print("| Skill | Escalation Rate | Count |")
    print("|---|---|---|")
    
    # Sort by escalation rate descending
    sorted_skills = sorted(skills_data.items(), key=lambda x: sum(x[1])/len(x[1]), reverse=True)
    
    for skill, triggers in sorted_skills:
        rate = sum(triggers) / len(triggers)
        count = len(triggers)
        print(f"| {skill} | {rate:7.0%} | {count:5} |")

if __name__ == "__main__":
    main()
