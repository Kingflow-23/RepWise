# Red-team and safety checks

Executable safety and robustness cases are stored separately from nominal generation cases in [`evals/red_team/red_team_cases.jsonl`](../../evals/red_team/red_team_cases.jsonl). RT001–RT008 cover profile-conflicting schedule and equipment requests, unrealistic session duration, prompt injection/fabricated evidence, medical diagnosis and treatment, supplement advice, duplicate training days, and impossible weekly volume. Their profiles are embedded directly in each case.

Run the cases against the real RAG generator:

```powershell
python -m src.evaluation.runner --mode red-team --llm ollama --retriever hybrid
```

The evaluator checks expected response status and case-specific required or forbidden patterns. For released plans it also checks structure, available days, session duration, equipment constraints, and that citation IDs came from supplied evidence. A disallowed `ok` response is reported as an unsafe release.

These are targeted checks, not a complete safety evaluation. Pattern matching can miss paraphrases or flag negations, each risk has limited wording coverage, and passing does not establish that recommendations are medically appropriate or semantically grounded. Inspect failures and real Ollama outputs manually. Citation ID validity alone does not prove that evidence supports a recommendation.

The benchmark intentionally does not claim to test multi-session adaptation, which RepWise does not implement. Add such cases only after the behavior exists and the test has concrete inputs and expected outcomes.
