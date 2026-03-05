"""
Experiment 3: Skill Classification Feasibility
===============================================
Hypothesis: You can reliably classify a prompt's required skill from text alone.

What it does:
- Tests 3 classification approaches (cheapest to most expensive):
  1. Keyword heuristics (free, instant)
  2. The cheap LLM itself as classifier (1 API call, fast)
  3. TF-IDF + simple ML classifier (free, local, needs training data)

- Measures accuracy against the ground-truth skill labels
- Answers: can we route WITHOUT calling an expensive model?

Runtime: ~5-8 minutes
"""

import re
import json
from collections import Counter
from tqdm import tqdm
from config import (
    get_client, MODELS, TEST_PROMPTS, SKILL_TAXONOMY,
    call_model, save_results, rate_limit_pause,
)


# =========================================================================
# Approach 1: Keyword Heuristics (free, instant)
# =========================================================================
KEYWORD_RULES = {
    "factual_qa": {
        "patterns": [
            r"\bwhat is\b", r"\bwhat are\b", r"\bwho (is|was|wrote)\b",
            r"\bdefine\b", r"\bexplain the difference\b", r"\bwhat does .+ mean\b",
            r"\bwhat's the\b",
        ],
        "anti_patterns": [r"\bprove\b", r"\bimplement\b", r"\bwrite.*(code|function|class)\b"],
    },
    "summarization": {
        "patterns": [
            r"\bsummarize\b", r"\bkey (ideas|points|takeaways)\b",
            r"\bin (plain english|simple terms)\b", r"\bbrief\b", r"\btl;?dr\b",
        ],
        "anti_patterns": [],
    },
    "basic_code": {
        "patterns": [
            r"\bwrite a (python |bash |sql |)function\b",
            r"\bwrite a (python |bash |sql |)script\b",
            r"\bone-liner\b", r"\bhow do I .+ in (python|javascript|node|bash)\b",
        ],
        "anti_patterns": [r"\bdesign pattern\b", r"\bthread-safe\b", r"\bunit test\b", r"\barchitect\b"],
    },
    "creative_simple": {
        "patterns": [
            r"\bwrite.*(email|message|post|note)\b", r"\brewrite\b",
            r"\bsubject line\b", r"\bcommit message\b", r"\bslack message\b",
            r"\bmore (professional|friendly|formal)\b",
        ],
        "anti_patterns": [r"\bnarrative\b", r"\bnovel\b", r"\bshort story\b"],
    },
    "multi_step_reasoning": {
        "patterns": [
            r"\bstep by step\b", r"\bshow your work\b",
            r"\bwhat time\b.*\b(meet|arrive)\b", r"\bprobability\b",
            r"\bbalance scale\b", r"\bcounterfeit\b",
            r"\bif .+ then .+ what\b",
        ],
        "anti_patterns": [],
    },
    "complex_code": {
        "patterns": [
            r"\bdesign a.*(class|system|pattern)\b", r"\bthread-safe\b",
            r"\bunit test\b", r"\bcircuit breaker\b", r"\bLRU cache\b",
            r"\bO\(1\)\b", r"\bpub/sub\b", r"\brate limit\b",
            r"\bimplementation\b",
        ],
        "anti_patterns": [],
    },
    "data_analysis": {
        "patterns": [
            r"\bSQL query\b", r"\bwindow function\b", r"\brolling average\b",
            r"\bA/B test\b", r"\bsample size\b", r"\bmissing values\b",
            r"\bType I\b.*\bType II\b", r"\bdiagnose\b.*\bdata\b",
        ],
        "anti_patterns": [],
    },
    "nuanced_creative": {
        "patterns": [
            r"\bopening paragraph\b", r"\bshort story\b", r"\btwist ending\b",
            r"\bnarrative\b", r"\bvoice\b.*\b(sarcastic|noir|literary)\b",
            r"\bpersuasive essay\b", r"\bsensory details\b", r"\bmood\b",
        ],
        "anti_patterns": [],
    },
    "multi_constraint": {
        "patterns": [
            r"\bconstraints?\b.*\b(budget|under|must|can't|no more)\b",
            r"\bplan\b.*\b(itinerary|schedule|meal|budget)\b",
            r"\bdesign a\b.*\barrangement\b",
            r"\ballocate\b.*\bbudget\b",
        ],
        "anti_patterns": [],
    },
    "formal_reasoning": {
        "patterns": [
            r"\bprove (that|by)\b", r"\bproof by\b", r"\binduction\b",
            r"\birrational\b", r"\bmaster theorem\b", r"\brecurrence\b",
        ],
        "anti_patterns": [],
    },
    "agentic": {
        "patterns": [
            r"\bpipeline\b.*\b(step|tool|API)\b", r"\borchestrat\b",
            r"\bagent\b", r"\bautomatically\b.*\b(creates?|sends?|assigns?)\b",
            r"\btool (call|interface)\b", r"\bCI/CD\b",
        ],
        "anti_patterns": [],
    },
    "ambiguous_open": {
        "patterns": [
            r"\bshould\b.*\b(startup|team|company)\b.*\b\?\b",
            r"\bis it (ethical|better|appropriate)\b",
            r"\bwhen (should|is it)\b.*\bvs\b",
            r"\bconflicting advice\b",
        ],
        "anti_patterns": [],
    },
}


def classify_by_keywords(prompt):
    """Rule-based classification using keyword patterns."""
    prompt_lower = prompt.lower()
    scores = {}

    for skill, rules in KEYWORD_RULES.items():
        score = 0
        # Check positive patterns
        for pattern in rules["patterns"]:
            if re.search(pattern, prompt_lower):
                score += 1
        # Check anti-patterns (reduce score)
        for pattern in rules.get("anti_patterns", []):
            if re.search(pattern, prompt_lower):
                score -= 0.5
        scores[skill] = score

    if not scores or max(scores.values()) == 0:
        return "ambiguous_open"  # default fallback

    return max(scores, key=scores.get)


# =========================================================================
# Approach 2: Cheap LLM as Classifier
# =========================================================================
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


def classify_by_llm(client, prompt):
    """Use the cheap LLM to classify the prompt."""
    result = call_model(
        client,
        MODELS["cheap"]["id"],
        CLASSIFIER_PROMPT.format(prompt=prompt),
        temperature=0.0,
        max_tokens=50,
    )
    if result["error"] or not result["content"]:
        return "ambiguous_open"

    # Extract the skill slug from response
    response = result["content"].strip().lower().replace("-", "_").replace(" ", "_")

    # Find best match
    valid_skills = set(SKILL_TAXONOMY.keys())
    if response in valid_skills:
        return response

    # Fuzzy match
    for skill in valid_skills:
        if skill in response or response in skill:
            return skill

    return "ambiguous_open"


# =========================================================================
# Approach 3: TF-IDF Classifier (local, no API)
# =========================================================================
def classify_by_tfidf(prompt, model=None, vectorizer=None):
    """Use a pre-trained TF-IDF + logistic regression model."""
    if model is None or vectorizer is None:
        return None  # Model not trained yet

    features = vectorizer.transform([prompt])
    return model.predict(features)[0]


def train_tfidf_classifier():
    """Train a simple TF-IDF classifier on the test prompts."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
    except ImportError:
        print("scikit-learn not installed, skipping TF-IDF approach")
        return None, None, None

    texts = [p["prompt"] for p in TEST_PROMPTS]
    labels = [p["skill"] for p in TEST_PROMPTS]

    vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2), stop_words="english")
    X = vectorizer.fit_transform(texts)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, labels)

    # Cross-validation score (leave-one-out style since dataset is small)
    from sklearn.model_selection import StratifiedKFold
    cv = StratifiedKFold(n_splits=min(5, min(Counter(labels).values())), shuffle=True, random_state=42)
    try:
        scores = cross_val_score(model, X, labels, cv=cv, scoring="accuracy")
        cv_accuracy = scores.mean()
    except ValueError:
        # If some classes have too few samples for stratified CV
        cv_accuracy = None

    return model, vectorizer, cv_accuracy


# =========================================================================
# Run all three approaches
# =========================================================================
def run_experiment():
    client = get_client()
    results = {"keyword": [], "llm": [], "tfidf": []}

    print("Experiment 3: Skill Classification Feasibility")
    print(f"Testing {len(TEST_PROMPTS)} prompts × 3 classification approaches\n")

    # --- Approach 1: Keywords (instant, no API) ---
    print("Approach 1: Keyword Heuristics...")
    correct = 0
    for test in TEST_PROMPTS:
        predicted = classify_by_keywords(test["prompt"])
        is_correct = predicted == test["skill"]
        correct += is_correct
        results["keyword"].append({
            "prompt": test["prompt"],
            "true_skill": test["skill"],
            "predicted_skill": predicted,
            "correct": is_correct,
        })
    keyword_acc = correct / len(TEST_PROMPTS)
    print(f"  Accuracy: {correct}/{len(TEST_PROMPTS)} ({keyword_acc*100:.1f}%)\n")

    # --- Approach 2: LLM Classifier ---
    print("Approach 2: Cheap LLM as Classifier...")
    correct = 0
    for test in tqdm(TEST_PROMPTS, desc="LLM classify"):
        predicted = classify_by_llm(client, test["prompt"])
        rate_limit_pause(1.5)
        is_correct = predicted == test["skill"]
        correct += is_correct
        results["llm"].append({
            "prompt": test["prompt"],
            "true_skill": test["skill"],
            "predicted_skill": predicted,
            "correct": is_correct,
        })
    llm_acc = correct / len(TEST_PROMPTS)
    print(f"  Accuracy: {correct}/{len(TEST_PROMPTS)} ({llm_acc*100:.1f}%)\n")

    # --- Approach 3: TF-IDF ---
    print("Approach 3: TF-IDF + Logistic Regression...")
    model, vectorizer, cv_accuracy = train_tfidf_classifier()
    if model is not None:
        correct = 0
        for test in TEST_PROMPTS:
            predicted = classify_by_tfidf(test["prompt"], model, vectorizer)
            is_correct = predicted == test["skill"]
            correct += is_correct
            results["tfidf"].append({
                "prompt": test["prompt"],
                "true_skill": test["skill"],
                "predicted_skill": predicted,
                "correct": is_correct,
            })
        tfidf_acc = correct / len(TEST_PROMPTS)
        print(f"  Train accuracy: {correct}/{len(TEST_PROMPTS)} ({tfidf_acc*100:.1f}%)")
        if cv_accuracy is not None:
            print(f"  Cross-val accuracy: {cv_accuracy*100:.1f}%")
        print(f"  (Note: train accuracy is overfitted; CV score is more realistic)\n")
    else:
        tfidf_acc = None

    save_results(results, "experiment_3_skill_classification.json")
    print_summary(results, keyword_acc, llm_acc, tfidf_acc, cv_accuracy)
    return results


def print_summary(results, keyword_acc, llm_acc, tfidf_acc, cv_accuracy):
    print("\n" + "=" * 70)
    print("EXPERIMENT 3 RESULTS: Skill Classification Feasibility")
    print("=" * 70)

    print(f"\n{'Approach':<35} {'Accuracy':>10} {'Cost':>12} {'Latency':>10}")
    print("-" * 70)
    print(f"{'Keyword heuristics':<35} {keyword_acc*100:>9.1f}% {'$0':>12} {'<1ms':>10}")
    print(f"{'Cheap LLM (8B) classifier':<35} {llm_acc*100:>9.1f}% {'~$0.001':>12} {'~200ms':>10}")
    if tfidf_acc is not None:
        cv_str = f"{cv_accuracy*100:.1f}%" if cv_accuracy else "N/A"
        print(f"{'TF-IDF + LogReg (train)':<35} {tfidf_acc*100:>9.1f}% {'$0':>12} {'<5ms':>10}")
        print(f"{'TF-IDF + LogReg (cross-val)':<35} {cv_str:>10} {'$0':>12} {'<5ms':>10}")

    # Error analysis for keyword approach
    print(f"\n--- Keyword Heuristic Error Analysis ---")
    confusion = {}
    for r in results["keyword"]:
        if not r["correct"]:
            key = f"{r['true_skill']} → {r['predicted_skill']}"
            confusion[key] = confusion.get(key, 0) + 1

    if confusion:
        print("Most common misclassifications:")
        for pair, count in sorted(confusion.items(), key=lambda x: -x[1])[:10]:
            print(f"  {pair}: {count}")
    else:
        print("No misclassifications!")

    # Error analysis for LLM approach
    print(f"\n--- LLM Classifier Error Analysis ---")
    confusion = {}
    for r in results["llm"]:
        if not r["correct"]:
            key = f"{r['true_skill']} → {r['predicted_skill']}"
            confusion[key] = confusion.get(key, 0) + 1

    if confusion:
        print("Most common misclassifications:")
        for pair, count in sorted(confusion.items(), key=lambda x: -x[1])[:10]:
            print(f"  {pair}: {count}")
    else:
        print("No misclassifications!")

    # Routing accuracy — what matters isn't exact skill, but correct TIER
    print(f"\n--- Routing-Level Accuracy (correct tier selection) ---")
    print("(Even if the exact skill is wrong, did we pick the right model tier?)\n")

    tier_map = {s: v["expected_min_tier"] for s, v in SKILL_TAXONOMY.items()}

    for approach_name in ["keyword", "llm"]:
        correct_tier = 0
        total = len(results[approach_name])
        for r in results[approach_name]:
            true_tier = tier_map.get(r["true_skill"], "mid")
            pred_tier = tier_map.get(r["predicted_skill"], "mid")
            if true_tier == pred_tier:
                correct_tier += 1
        print(f"  {approach_name}: {correct_tier}/{total} ({correct_tier/total*100:.1f}%) correct tier")

    print(f"\n💡 Key insight: Even if skill classification isn't perfect,")
    print(f"   routing accuracy (correct tier) is what actually matters for cost savings.")


if __name__ == "__main__":
    run_experiment()
