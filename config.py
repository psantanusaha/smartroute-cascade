"""
Shared configuration and utilities for SmartRoute experiments.
"""
import os
import json
import time
from pathlib import Path
from openai import OpenAI


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

# ---------------------------------------------------------------------------
# Groq client setup (free tier, OpenAI-compatible)
# ---------------------------------------------------------------------------
def get_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "Set GROQ_API_KEY environment variable. "
            "Get a free key at https://console.groq.com"
        )
    return OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)


# ---------------------------------------------------------------------------
# Model tiers
# ---------------------------------------------------------------------------
MODELS = {
    "cheap": {
        "id": "llama-3.1-8b-instant",
        "label": "Llama 3.1 8B",
        "tier": "cheap",
    },
    "mid": {
        "id": "llama-3.3-70b-versatile",
        "label": "Llama 3.3 70B",
        "tier": "mid",
    },
    "expensive": {
        "id": "deepseek-r1-distill-llama-70b",
        "label": "DeepSeek R1 70B",
        "tier": "expensive",
    },
}

# ---------------------------------------------------------------------------
# Skill taxonomy (v1 — 12 categories)
# ---------------------------------------------------------------------------
SKILL_TAXONOMY = {
    "factual_qa": {
        "label": "Factual Q&A",
        "description": "Simple lookups, definitions, general knowledge",
        "expected_min_tier": "cheap",
    },
    "summarization": {
        "label": "Summarization",
        "description": "Condensing text, extracting key points",
        "expected_min_tier": "cheap",
    },
    "basic_code": {
        "label": "Basic Code Generation",
        "description": "Single-function scripts, syntax help, simple bugs",
        "expected_min_tier": "cheap",
    },
    "creative_simple": {
        "label": "Simple Creative Writing",
        "description": "Short emails, social posts, basic rewriting",
        "expected_min_tier": "cheap",
    },
    "multi_step_reasoning": {
        "label": "Multi-step Reasoning",
        "description": "Chain-of-thought math, logic puzzles, word problems",
        "expected_min_tier": "mid",
    },
    "complex_code": {
        "label": "Complex Code / Architecture",
        "description": "System design, multi-file, design patterns, debugging",
        "expected_min_tier": "mid",
    },
    "data_analysis": {
        "label": "Data Analysis & Interpretation",
        "description": "Statistical reasoning, interpreting datasets, SQL",
        "expected_min_tier": "mid",
    },
    "nuanced_creative": {
        "label": "Nuanced Creative Writing",
        "description": "Voice, tone, literary devices, long-form narrative",
        "expected_min_tier": "mid",
    },
    "multi_constraint": {
        "label": "Multi-constraint Planning",
        "description": "Itineraries, schedules, optimization under constraints",
        "expected_min_tier": "mid",
    },
    "formal_reasoning": {
        "label": "Formal Reasoning / Proofs",
        "description": "Mathematical proofs, formal logic, derivations",
        "expected_min_tier": "expensive",
    },
    "agentic": {
        "label": "Agentic / Tool Orchestration",
        "description": "Multi-tool workflows, function calling, planning",
        "expected_min_tier": "expensive",
    },
    "ambiguous_open": {
        "label": "Ambiguous / Open-ended",
        "description": "Philosophical, ethical, highly subjective questions",
        "expected_min_tier": "mid",
    },
}

# ---------------------------------------------------------------------------
# Test prompts — 5 per skill category = 60 total
# These are designed to probe capability boundaries
# ---------------------------------------------------------------------------
TEST_PROMPTS = [
    # ---- factual_qa (5) ----
    {"skill": "factual_qa", "prompt": "What is the capital of Australia?", "difficulty": "easy"},
    {"skill": "factual_qa", "prompt": "Who wrote 'Pride and Prejudice'?", "difficulty": "easy"},
    {"skill": "factual_qa", "prompt": "What is the speed of light in meters per second?", "difficulty": "easy"},
    {"skill": "factual_qa", "prompt": "Explain the difference between TCP and UDP in 2-3 sentences.", "difficulty": "medium"},
    {"skill": "factual_qa", "prompt": "What are the ACID properties in database systems?", "difficulty": "medium"},

    # ---- summarization (5) ----
    {"skill": "summarization", "prompt": "Summarize the key ideas of the Agile Manifesto in 3 bullet points.", "difficulty": "easy"},
    {"skill": "summarization", "prompt": "Summarize the concept of MapReduce in plain English for a non-technical person.", "difficulty": "easy"},
    {"skill": "summarization", "prompt": "Explain the main argument of 'The Lean Startup' methodology in 2 sentences.", "difficulty": "medium"},
    {"skill": "summarization", "prompt": "What are the 3 most important takeaways from the CAP theorem for distributed systems?", "difficulty": "medium"},
    {"skill": "summarization", "prompt": "Summarize the differences between microservices and monolithic architecture. Cover tradeoffs.", "difficulty": "medium"},

    # ---- basic_code (5) ----
    {"skill": "basic_code", "prompt": "Write a Python function to check if a string is a palindrome.", "difficulty": "easy"},
    {"skill": "basic_code", "prompt": "Write a Python function to find the nth Fibonacci number using recursion.", "difficulty": "easy"},
    {"skill": "basic_code", "prompt": "Write a bash one-liner to find all .py files modified in the last 7 days.", "difficulty": "medium"},
    {"skill": "basic_code", "prompt": "Write a Python function that reads a CSV file and returns the top 5 rows sorted by a given column name.", "difficulty": "medium"},
    {"skill": "basic_code", "prompt": "Write a Python decorator that retries a function up to 3 times if it raises an exception, with exponential backoff.", "difficulty": "medium"},

    # ---- creative_simple (5) ----
    {"skill": "creative_simple", "prompt": "Write a professional email declining a meeting invitation due to a scheduling conflict.", "difficulty": "easy"},
    {"skill": "creative_simple", "prompt": "Rewrite this sentence to sound more professional: 'Hey, the thing you sent is kinda broken.'", "difficulty": "easy"},
    {"skill": "creative_simple", "prompt": "Write a LinkedIn post announcing a job change to a new company.", "difficulty": "easy"},
    {"skill": "creative_simple", "prompt": "Write 3 subject line options for a marketing email about a 30% off summer sale.", "difficulty": "easy"},
    {"skill": "creative_simple", "prompt": "Write a short thank-you note to a mentor who helped with a career transition.", "difficulty": "easy"},

    # ---- multi_step_reasoning (5) ----
    {"skill": "multi_step_reasoning", "prompt": "A train leaves Station A at 9:00 AM traveling at 60 mph. Another train leaves Station B (300 miles away) at 10:00 AM traveling toward Station A at 90 mph. At what time do they meet?", "difficulty": "medium"},
    {"skill": "multi_step_reasoning", "prompt": "I have 3 boxes. Box A has 2 red and 3 blue balls. Box B has 4 red and 1 blue. Box C has 1 red and 4 blue. I pick a box at random, then draw a ball. It's red. What's the probability it came from Box B?", "difficulty": "hard"},
    {"skill": "multi_step_reasoning", "prompt": "If all Bloops are Razzies, and all Razzies are Lazzies, and some Lazzies are Tazzies, can we conclude that some Bloops are Tazzies? Explain your reasoning step by step.", "difficulty": "hard"},
    {"skill": "multi_step_reasoning", "prompt": "A farmer has 100 meters of fencing. What dimensions of a rectangular pen maximize the enclosed area? Show your work.", "difficulty": "medium"},
    {"skill": "multi_step_reasoning", "prompt": "You have 12 coins, one is counterfeit (either heavier or lighter). Using a balance scale exactly 3 times, describe a strategy to find the counterfeit coin and determine if it's heavier or lighter.", "difficulty": "hard"},

    # ---- complex_code (5) ----
    {"skill": "complex_code", "prompt": "Design a Python class for an LRU cache with O(1) get and put operations. Include type hints and handle edge cases.", "difficulty": "medium"},
    {"skill": "complex_code", "prompt": "Implement a rate limiter using the token bucket algorithm in Python. It should support configurable rate and burst size, be thread-safe, and include unit tests.", "difficulty": "hard"},
    {"skill": "complex_code", "prompt": "Write a Python function that takes a nested JSON object of arbitrary depth and flattens it into a single-level dictionary with dot-notation keys. Handle arrays, nulls, and empty objects correctly.", "difficulty": "medium"},
    {"skill": "complex_code", "prompt": "Design a simple pub/sub event system in Python. Support subscribe, unsubscribe, and publish. Handle wildcard topic patterns (e.g., 'user.*' matches 'user.created' and 'user.deleted').", "difficulty": "hard"},
    {"skill": "complex_code", "prompt": "Implement a basic retry mechanism with circuit breaker pattern in Python. After 5 consecutive failures, the circuit should open for 30 seconds before allowing a retry.", "difficulty": "hard"},

    # ---- data_analysis (5) ----
    {"skill": "data_analysis", "prompt": "Write a SQL query to find the top 3 customers by total order value in the last 90 days, including their order count and average order value.", "difficulty": "medium"},
    {"skill": "data_analysis", "prompt": "I have a dataset with columns: date, revenue, marketing_spend. Revenue has been declining while marketing_spend increased. What analyses would you run to diagnose the issue? Outline your approach.", "difficulty": "medium"},
    {"skill": "data_analysis", "prompt": "Explain the difference between Type I and Type II errors with a concrete business example. When would you prefer to minimize each?", "difficulty": "medium"},
    {"skill": "data_analysis", "prompt": "Write a SQL query using window functions to calculate a 7-day rolling average of daily sales, along with the percent change from the previous day.", "difficulty": "hard"},
    {"skill": "data_analysis", "prompt": "Design an A/B test for a checkout flow change. Specify: sample size calculation approach, primary/secondary metrics, guardrail metrics, and when you'd call the test.", "difficulty": "hard"},

    # ---- nuanced_creative (5) ----
    {"skill": "nuanced_creative", "prompt": "Write the opening paragraph of a noir detective story set in a futuristic Tokyo. Use sensory details and establish mood.", "difficulty": "medium"},
    {"skill": "nuanced_creative", "prompt": "Write a product description for a $200 mechanical keyboard that makes the reader feel the experience of typing on it. No bullet points — pure narrative.", "difficulty": "medium"},
    {"skill": "nuanced_creative", "prompt": "Rewrite this corporate announcement in the voice of a sarcastic standup comedian: 'We are excited to announce our new synergy-driven initiative to leverage cross-functional alignment.'", "difficulty": "medium"},
    {"skill": "nuanced_creative", "prompt": "Write a 200-word short story where the twist ending recontextualizes everything the reader assumed. The story should be about a lighthouse keeper.", "difficulty": "hard"},
    {"skill": "nuanced_creative", "prompt": "Write a persuasive essay arguing that failure is more valuable than success. Use at least one unexpected analogy and avoid cliches.", "difficulty": "hard"},

    # ---- multi_constraint (5) ----
    {"skill": "multi_constraint", "prompt": "Plan a 3-day Tokyo itinerary for a family with a toddler. Constraints: budget under $200/day, no more than 2 subway rides per day, must include one cultural site and one park daily, hotel is in Shinjuku.", "difficulty": "hard"},
    {"skill": "multi_constraint", "prompt": "Schedule 5 meetings into a workday (9am-5pm) with these constraints: Meeting A (1hr) must be before lunch, Meeting B (30min) must be after Meeting C (45min), Meeting D (1hr) can't overlap with A, and Meeting E (2hr) needs to end by 3pm. Lunch is 12-1pm.", "difficulty": "hard"},
    {"skill": "multi_constraint", "prompt": "Design a weekly meal plan for 2 adults: vegetarian, under $75/week total, max 30 min prep per meal, must include 50g+ protein per day, and no recipe should repeat within the week.", "difficulty": "hard"},
    {"skill": "multi_constraint", "prompt": "Allocate a $10,000 marketing budget across 4 channels (social, email, SEO, paid ads) for a SaaS startup. Constraints: social can't exceed 30%, email must be at least $1000, ROI targets differ per channel. Justify your allocation.", "difficulty": "hard"},
    {"skill": "multi_constraint", "prompt": "Design a seating arrangement for a dinner party of 8. Constraints: Alice can't sit next to Bob, Charlie must sit next to Dana, couples shouldn't sit together, and the host must be at the head. Provide the arrangement and explain your reasoning.", "difficulty": "hard"},

    # ---- formal_reasoning (5) ----
    {"skill": "formal_reasoning", "prompt": "Prove that the square root of 2 is irrational using proof by contradiction.", "difficulty": "hard"},
    {"skill": "formal_reasoning", "prompt": "Prove by induction that the sum of the first n natural numbers is n(n+1)/2.", "difficulty": "medium"},
    {"skill": "formal_reasoning", "prompt": "Prove that for any integer n, if n² is even, then n is even.", "difficulty": "medium"},
    {"skill": "formal_reasoning", "prompt": "Prove that there are infinitely many prime numbers.", "difficulty": "hard"},
    {"skill": "formal_reasoning", "prompt": "Given the recurrence T(n) = 2T(n/2) + n with T(1) = 1, prove that T(n) = O(n log n) using the Master Theorem or substitution method.", "difficulty": "hard"},

    # ---- agentic (5) ----
    {"skill": "agentic", "prompt": "I want to build a system that monitors a GitHub repo, and when a new issue is labeled 'bug', automatically creates a Jira ticket, assigns it based on the file paths mentioned, and sends a Slack notification. Describe the architecture and the tools/APIs needed.", "difficulty": "hard"},
    {"skill": "agentic", "prompt": "Design a pipeline where: (1) a user uploads a PDF, (2) the system extracts text, (3) identifies key entities, (4) searches a database for related records, (5) generates a summary email. Define the tools, data flow, and error handling.", "difficulty": "hard"},
    {"skill": "agentic", "prompt": "I have a function that fetches weather data, another that queries a flight API, and another that checks hotel availability. Write a plan for an agent that takes 'Plan me a weekend trip from SFO to somewhere warm under $500' and orchestrates these tools to produce an answer.", "difficulty": "hard"},
    {"skill": "agentic", "prompt": "Design an automated code review agent. It should: clone a PR branch, run linters, run tests, identify changed files, analyze diffs for common patterns (security issues, performance), and post a structured review comment. Outline the steps and tool calls.", "difficulty": "hard"},
    {"skill": "agentic", "prompt": "Build a data reconciliation agent that compares records between two databases (Postgres and MongoDB), identifies mismatches, classifies them by severity, and generates a report. Define the tool interfaces and decision logic.", "difficulty": "hard"},

    # ---- ambiguous_open (5) ----
    {"skill": "ambiguous_open", "prompt": "Is it ethical for AI companies to use publicly posted internet data to train models? Consider the perspectives of creators, users, and companies.", "difficulty": "medium"},
    {"skill": "ambiguous_open", "prompt": "What makes a great engineering manager? I've heard conflicting advice.", "difficulty": "medium"},
    {"skill": "ambiguous_open", "prompt": "Should a startup prioritize growth or profitability in its first 2 years? What factors should influence this decision?", "difficulty": "medium"},
    {"skill": "ambiguous_open", "prompt": "Is remote work better than in-office for software engineering teams? Consider productivity, culture, and career growth.", "difficulty": "medium"},
    {"skill": "ambiguous_open", "prompt": "When is it appropriate to rewrite a system from scratch versus incrementally refactoring it? What signals should you look for?", "difficulty": "medium"},
]


# ---------------------------------------------------------------------------
# Judging prompt — used by the expensive model to evaluate responses
# ---------------------------------------------------------------------------
JUDGE_SYSTEM_PROMPT = """You are an expert evaluator of LLM responses. 
Rate the given response on a scale of 1-5:

5 = Excellent: Correct, complete, well-structured, insightful
4 = Good: Mostly correct, covers key points, minor gaps
3 = Adequate: Partially correct, misses some important aspects
2 = Poor: Significant errors or major omissions
1 = Fail: Wrong, incoherent, or completely misses the point

Respond ONLY with a JSON object:
{"score": <1-5>, "pass": <true if score >= 4>, "reason": "<1 sentence explanation>"}"""


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def call_model(client, model_id, prompt, system_prompt=None, temperature=0.3, max_tokens=2048):
    """Call a model and return response with metadata."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    start = time.time()
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        elapsed = time.time() - start
        content = response.choices[0].message.content
        usage = response.usage
        return {
            "content": content,
            "latency_s": round(elapsed, 3),
            "input_tokens": usage.prompt_tokens if usage else 0,
            "output_tokens": usage.completion_tokens if usage else 0,
            "error": None,
        }
    except Exception as e:
        return {
            "content": None,
            "latency_s": round(time.time() - start, 3),
            "input_tokens": 0,
            "output_tokens": 0,
            "error": str(e),
        }


def judge_response(client, prompt, response_text, model_id=None):
    """Use the expensive model to judge a response. Returns score dict."""
    if model_id is None:
        model_id = MODELS["mid"]["id"]  # Use 70B as judge to save rate limits

    judge_prompt = f"""Original question: {prompt}

Response to evaluate:
{response_text}

Rate this response."""

    result = call_model(client, model_id, judge_prompt, system_prompt=JUDGE_SYSTEM_PROMPT, temperature=0.0)

    if result["error"] or not result["content"]:
        return {"score": 0, "pass": False, "reason": f"Judge error: {result['error']}"}

    try:
        # Try to parse JSON from response (handle markdown code blocks)
        text = result["content"].strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        return json.loads(text)
    except (json.JSONDecodeError, IndexError):
        # Fallback: try to extract score from text
        return {"score": 0, "pass": False, "reason": f"Parse error: {result['content'][:100]}"}


def save_results(results, filename):
    """Save results to JSON file in results/ directory."""
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    filepath = results_dir / filename
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {filepath}")
    return filepath


def load_results(filename):
    """Load results from JSON file."""
    filepath = Path(__file__).parent / "results" / filename
    with open(filepath) as f:
        return json.load(f)


def rate_limit_pause(seconds=2):
    """Pause to respect Groq free tier rate limits."""
    time.sleep(seconds)
