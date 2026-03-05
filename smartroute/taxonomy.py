# smartroute/taxonomy.py

DEFAULT_SKILL_TAXONOMY = {
    "factual_qa": "cheap",
    "summarization": "cheap",
    "basic_code": "cheap",
    "creative_simple": "cheap",
    "multi_step_reasoning": "mid",
    "complex_code": "mid",
    "data_analysis": "mid",
    "nuanced_creative": "mid",
    "multi_constraint": "mid",
    "formal_reasoning": "expensive",
    "agentic": "expensive",
    "ambiguous_open": "mid",
}

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
