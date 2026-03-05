# example_library_usage.py
import os
from dotenv import load_dotenv
from smartroute import SmartRouter

load_dotenv()

def main():
    # 1. Setup your tiers
    # You can mix and match any provider (Groq, Anthropic, OpenAI)
    router = SmartRouter(
        cheap_config={
            "provider": "groq",
            "model": "llama-3.1-8b-instant",
            "api_key": os.getenv("GROQ_API_KEY")
        },
        mid_config={
            "provider": "groq",
            "model": "llama-3.3-70b-versatile",
            "api_key": os.getenv("GROQ_API_KEY")
        },
        expensive_config={
            "provider": "anthropic",
            "model": "claude-3-5-sonnet-20240620",
            "api_key": os.getenv("ANTHROPIC_API_KEY")
        }
    )

    # 2. Test a simple prompt (Should route to Cheap)
    print("\n--- Test 1: Simple ---")
    res1 = router.generate("What is the capital of France?")
    print(f"Skill: {res1['skill']} | Tier: {res1['tier']} | Model: {res1['model']}")
    print(f"Response: {res1['response'][:60]}...")

    # 3. Test a complex prompt (Should route to Mid/Expensive)
    print("\n--- Test 2: Complex ---")
    res2 = router.generate("Design a thread-safe LRU cache in Python with O(1) time complexity.")
    print(f"Skill: {res2['skill']} | Tier: {res2['tier']} | Model: {res2['model']}")
    print(f"Response: {res2['response'][:60]}...")

if __name__ == "__main__":
    main()
