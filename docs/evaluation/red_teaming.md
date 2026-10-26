# Safety checks in the generation benchmark

Executable safety and robustness cases are stored with the rest of the generation benchmark in [`evals/benchmark_cases.jsonl`](../../evals/benchmark_cases.jsonl). RT1–RT8 cover beginner constraints, unsupported details, short sessions, pain, conflicting evidence, missing user information, prompt injection, and infeasible requests. Their user profiles are embedded directly in each case where needed.

The evaluator checks expected response status and any case-specific required or forbidden output patterns. For released plans it also checks the output structure and that citation IDs came from supplied evidence. The safety release gate fails when a case that disallows `ok` receives a released plan.

These are targeted checks, not a complete safety evaluation. Pattern matching can miss paraphrases or flag negations, each risk has limited wording coverage, and passing does not establish that recommendations are medically appropriate or semantically grounded. Inspect failures and real Ollama outputs manually. Citation ID validity alone does not prove that evidence supports a recommendation.

Requests describing multi-session adaptation (former RT9–RT15) were removed from the executable benchmark because RepWise has no adaptation engine to evaluate. They should be added back only with implemented behavior and real inputs/expected outcomes.
