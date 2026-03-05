# SmartRoute Hypothesis Validation Experiments

## What are we testing?

Before building a routing layer, we need to validate 3 core hypotheses:

### Hypothesis 1: Predictable Capability Gaps
**"Small models fail predictably on specific task types, not randomly."**
- If failures are random, routing is pointless
- If failures cluster by skill category, routing works

### Hypothesis 2: Cheap Models Handle Most Traffic
**"60-80% of typical LLM requests can be handled by the cheapest model."**
- If it's only 20%, the cost savings don't justify a router
- If it's 80%, the business case is obvious

### Hypothesis 3: Skill Classification is Feasible
**"You can reliably classify a prompt's required skill from the text alone."**
- If classification is unreliable, pre-routing fails
- If a simple classifier gets 85%+ accuracy, the approach works

## Setup

```bash
# 1. Get a free Groq API key at https://console.groq.com
# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your API key
export GROQ_API_KEY="your-key-here"

# 4. Run experiments
python experiment_1_capability_gaps.py
python experiment_2_traffic_distribution.py
python experiment_3_skill_classifier.py

# 5. Analyze results
python analyze_results.py
```

## Model Tiers (via Groq free tier)

| Tier | Model | Equivalent |
|------|-------|------------|
| Cheap | llama-3.1-8b-instant | Haiku-class |
| Mid | llama-3.3-70b-versatile | Sonnet-class |
| Expensive | deepseek-r1-distill-llama-70b | Opus-class |

## Output

All results are saved to `results/` as JSON files with full traces.
`analyze_results.py` generates summary tables and charts.
