# I tried to stop paying Claude Sonnet prices for questions that don't need Claude Sonnet

So I've been building on top of LLM APIs for a while now, and at some point I got into this habit of just defaulting to the best model available — Sonnet, GPT-4o, whatever was at the top of the benchmark list that week — because that way I didn't have to think about whether the response was good enough, it just was. And that worked fine until it didn't, which is when I started actually looking at what these API calls were being spent on.

Honestly most of it was embarrassing. "What does this error mean." "Write a commit message for this diff." "Summarize these three bullet points into two bullet points." I was running all of this through Sonnet at $3 per million input tokens and $15 per million output tokens, which is fine if you're doing something that actually needs that, but for half the stuff in my logs it was like hiring a specialist surgeon to put on a bandaid.

So I started thinking about this problem more seriously — not just "use a cheaper model sometimes" but actually trying to build something systematic around it, run some real experiments, and figure out what actually works vs what sounds good in theory.

---

## the naive approach and why it didn't work

The first thing I tried, and honestly the first thing anyone tries, is to just send the request to the cheap model and then check if the answer is good enough before returning it to the user. This is basically what FrugalGPT does, what RouteLLM does, what everyone in this space is doing in some form. The intuition is straightforward — try cheap, verify, escalate if needed.

The problem I kept running into is: how do you verify? I tried the obvious things. I looked for hedging language in the response — stuff like "I'm not sure" or "I don't have enough information" or "I cannot" — figuring that if the cheap model knew it was out of its depth it would say so, and I could catch those and escalate. I ran this against five deliberately hard sessions: a TSP algorithm implementation with Held-Karp and full path reconstruction, a clinical case with Sgarbossa criteria assessment, a Monte Carlo options pricer with control variates, a 15-clue logic puzzle, and a real-time collaborative doc editor design at 10 million users. Out of those five, the heuristic triggered on exactly zero of them. Zero recall. Completely useless. What I found instead is that Llama 8B, when it doesn't know something, doesn't tell you it doesn't know — it just answers anyway, confidently, with a nicely formatted response and everything, and the answer is wrong. I gave it the Held-Karp TSP implementation and it wrote me clean, well-commented Python that had fundamental errors in the DP table construction. No hedging, no caveats, just confident wrongness.

Then I tried asking the model to rate its own answer before returning it — like give yourself a score 1-5, and if you score yourself below 3 I'll escalate. This was better, kind of. On the logic puzzle, the model actually gave itself a 1 and that one correctly triggered escalation — it seemed to know it was stuck. But on the TSP Held-Karp implementation, it rated itself a 4 while Sonnet, looking at the same output, scored it a 2. That's the frustrating part — a model that doesn't know its code is wrong isn't going to know that when you ask it again a different way. Self-rating got me 50% recall, and the 50% it missed was exactly the kind of failure I most needed to catch: code that looks correct but isn't.

Here's how the three signaling approaches compared across the five hard sessions:

| Signal | Recall | Precision | F1 | Latency | Extra Cost |
|---|---|---|---|---|---|
| Heuristics (hedging phrases) | 0% | — | 0.00 | <1ms | Free |
| Self-rating (ask model to score itself) | 50% | 100% | 0.67 | +1 LLM call | ~$0.001/req |
| Pre-classifier (LLM taxonomy) | **100%** | **100%** | **1.00** | ~200ms | ~$0.001/req |

---

## the thing that actually worked

At some point I flipped the whole approach around and asked a different question: instead of looking at the output to figure out if I should have used a better model, what if I look at the input before I generate anything?

What I mean is — before I send the user's prompt to any model, I run a quick classification step that tries to figure out what *kind* of task it is. I built a taxonomy of 12 skill categories — things like `factual_qa`, `summarization`, `basic_code` on the simple end, and `complex_code`, `formal_reasoning`, `agentic` on the harder end — and I use Llama 3.1 8B running on Groq to classify the incoming prompt against that taxonomy. The whole thing takes under 200ms and costs basically nothing, and then based on what category it lands in, I route to the appropriate model tier before any generation happens.

The classification accuracy was 87% on the broad prompt set, which sounds maybe not that impressive but in practice is pretty good because the errors tend to be on genuinely ambiguous prompts where it's not obvious even to a human which tier they need. For the five hard sessions I'd specifically designed to break cheap models — the ones where heuristics and self-rating both struggled — the recall was 100%. The TSP Held-Karp implementation got classified as `complex_code`. The Monte Carlo pricer got `complex_code`. The 15-clue logic puzzle got `multi_step_reasoning`. The collaborative doc editor got `agentic`. All five were routed to Sonnet before a single token of cheap-model generation happened, zero false positives, zero false negatives.

---

## the numbers from the actual experiments

I ran two sets of sessions. The first five were deliberately easy: a simple greeting exchange, basic Python questions (list comprehensions, recursion), high-level concepts like AI ethics and the CAP theorem, creative writing, and a technical deep-dive on transformers and attention mechanisms.

| Session | Turns | Min Score | Escalated | Cost Savings |
|---|---|---|---|---|
| simple-greeting | 3 | 5/5 | No | 75% |
| coding-basics | 4 | 4/5 | No | 75% |
| complex-reasoning | 3 | 5/5 | No | 75% |
| creative-writing | 2 | 4/5 | No | 75% |
| technical-deep-dive | 3 | 4/5 | No | 75% |

For all five, the cheap model handled every turn without escalation. Average quality 4.7/5, cost savings 75.1%.

The second five were hard sessions designed to push cheap models to their limit: implementing TSP with Held-Karp and full path reconstruction, clinical differential diagnosis with Sgarbossa criteria, a Monte Carlo options pricer with control variates and convergence tables, a 15-clue Zebra-style logic puzzle, and designing a real-time collaborative document editor for 10 million users. The pattern that emerged was consistent — the opening turns scored 4 or 5 even in the hard sessions, because "explain what dynamic programming is" and "explain what Black-Scholes does" are well within a cheap model's comfort zone. The final turns, where I asked for precise multi-constraint implementations, is where the quality dropped.

| Session | Final Prompt | Cheap Model Score | Classifier Label | Escalated |
|---|---|---|---|---|
| algorithms-tsp | Held-Karp TSP + path reconstruction | 2/5 | `complex_code` | Yes |
| clinical-reasoning | LBBB + Sgarbossa + modified score | 2/5 | `data_analysis` | Yes |
| quant-finance | Monte Carlo pricer + control variates | 1/5 | `complex_code` | Yes |
| logic-puzzle | 15-clue Zebra puzzle | 1/5 | `multi_step_reasoning` | Yes |
| collab-doc-design | Real-time doc editor, 10M users, OT vs CRDT | 2/5 | `agentic` | Yes |

Combined across all 10 sessions, the SmartRoute approach using the pre-classifier landed at 46.94% cost savings compared to always using Sonnet, with an average quality score of 4.43 out of 5.

| | All-Sonnet | SmartRoute (pre-classifier) |
|---|---|---|
| Cost (10 sessions) | $0.196 | $0.104 |
| Savings | — | **46.94%** |
| Avg Quality (1–5) | 5.00 | 4.43 |

The drop from the all-Sonnet baseline is 0.57 points, and it's worth being honest about where that comes from — it's not evenly distributed. The easy sessions that never escalated scored 5/5, basically indistinguishable from Sonnet. The hard sessions where the final turn correctly escalated scored 5/5 there because they got routed to the right model. The 0.57 drag comes from the middle turns in hard sessions that scored 3 or 4 — good enough to not trigger escalation, but not quite what Sonnet would have done. Still a 4, but a different kind of 4.

Something that genuinely surprised me in the experiments was how capable Haiku actually is for the broad, non-implementation questions even when the topic sounds hard. Economics reasoning, clinical concepts, distributed systems design questions at a conceptual level — it kept scoring 4 or 5 on all of those. The one thing that reliably broke it was very precise, multi-constraint code generation where you need the model to simultaneously hold a complex algorithm in mind, implement it correctly, handle specific edge cases, and produce test cases with manually verified outputs — that specific combination is where it falls apart. But "explain the Phillips curve and its modern critiques" or "design a collaborative document editor for 10 million users at an architectural level" — Haiku handles those well.

---

## how this compares to what's already been published

FrugalGPT from Stanford is probably the most cited paper in this space and their cost reductions are genuinely impressive — up to 98% on some benchmarks — but they need to train a DistilBERT scorer on your specific task distribution, and their benchmarks are fairly narrow (financial news classification, legal sentence classification, conversational QA). RouteLLM from Berkeley takes a different approach using human preference data from Chatbot Arena to train routers, and their performance on things like MT-Bench is strong, but again you're training on 80,000 preference pairs and the generalization to new model pairs isn't guaranteed.

The thing I was trying to do differently is make something that works out of the box without any training, that you can point at Anthropic or OpenAI or Groq or whatever combination, and that doesn't break when a new model comes out. The taxonomy is just a JSON config — if Haiku 5 turns out to be better at formal reasoning than I'd expect, I change one line and the routing changes. No retraining, no new preference data collection. And the signal comparison experiment I ran — comparing heuristics vs self-rating vs pre-classification on the same sessions — is, as far as I can tell, not something the existing papers have done head-to-head on commercial API pairs, so that at least is a contribution that doesn't show up in FrugalGPT or RouteLLM.

---

## caveats and what I'd do differently

This is a fairly small experiment — 10 sessions, one cheap/expensive model pair at a time, a taxonomy I designed myself that probably somewhat favors the hypothesis I was trying to prove. For a more rigorous claim you'd want a few hundred sessions with real user traffic, multiple provider combinations tested simultaneously, and ideally some human evaluation to validate that the LLM judge scores actually correlate with what real users would think of the responses. The fact that Sonnet is both the expensive model and the judge is an obvious conflict of interest that I haven't resolved.

The other thing I haven't fully addressed is latency. The classification step adds around 200ms upfront to every request, which is a real tradeoff — you're paying that even on the simple requests that were never going to need escalation anyway. In most web applications that's probably fine, but if you're doing something latency-sensitive it's worth thinking about whether to use the keyword-based classifier (80% accuracy, basically instant) instead of the LLM classifier.

---

All the code for this is on GitHub at github.com/psantanusaha/smartroute-cascade — the experiments, the taxonomy, the routing logic, the full result JSON files. I'm working on a more formal writeup of this as a preprint, so if you've done work on LLM routing and want to compare notes or point me at things I've missed, I'm around on LinkedIn.
