# Rubric-order experiment

Judge: qwen3.5:4b
Cases: G001, G002, G003, G004, G005
Completed judge calls: 60/60
Judge errors: 1
Cases eligible for comparison: 4/5

The same saved answers, evidence, requests and schemas are used in every order.
Score meanings are unchanged. Evidence support is excluded from answer quality.
Cases missing any arm/order score are excluded from all comparisons; errors are not zero scores.

| Order | Model | Answer quality |
|---|---|---|
| 0-1-2 | gemma4:12b + RAG | 0.8571 |
| 0-1-2 | qwen3.5:9b + RAG | 0.8929 |
| 0-2-1 | gemma4:12b + RAG | 0.8571 |
| 0-2-1 | qwen3.5:9b + RAG | 0.8571 |
| 1-0-2 | gemma4:12b + RAG | 0.9464 |
| 1-0-2 | qwen3.5:9b + RAG | 0.9464 |
| 1-2-0 | gemma4:12b + RAG | 0.9464 |
| 1-2-0 | qwen3.5:9b + RAG | 0.8929 |
| 2-0-1 | gemma4:12b + RAG | 0.7857 |
| 2-0-1 | qwen3.5:9b + RAG | 0.7857 |
| 2-1-0 | gemma4:12b + RAG | 0.7857 |
| 2-1-0 | qwen3.5:9b + RAG | 0.7857 |

Model ranking changes across orders: True

| Model | Dimension | Cases with changed scores | Largest score range |
|---|---|---|---|
| gemma4:12b + RAG | request_fit | 0/4 | 0 |
| gemma4:12b + RAG | schedule_fit | 0/4 | 0 |
| gemma4:12b + RAG | profile_safety | 0/4 | 0 |
| gemma4:12b + RAG | domain_safety | 0/4 | 0 |
| gemma4:12b + RAG | clarity | 0/4 | 0 |
| gemma4:12b + RAG | json_validity | 4/4 | 1 |
| gemma4:12b + RAG | plan_schema_validity | 4/4 | 2 |
| gemma4:12b + RAG | evidence_support | 0/4 | 0 |
| qwen3.5:9b + RAG | request_fit | 0/4 | 0 |
| qwen3.5:9b + RAG | schedule_fit | 0/4 | 0 |
| qwen3.5:9b + RAG | profile_safety | 1/4 | 1 |
| qwen3.5:9b + RAG | domain_safety | 0/4 | 0 |
| qwen3.5:9b + RAG | clarity | 0/4 | 0 |
| qwen3.5:9b + RAG | json_validity | 4/4 | 1 |
| qwen3.5:9b + RAG | plan_schema_validity | 4/4 | 2 |
| qwen3.5:9b + RAG | evidence_support | 1/4 | 1 |

Per-order rankings (equal scores remain tied):

- 0-1-2: qwen3.5:9b + RAG > gemma4:12b + RAG
- 0-2-1: gemma4:12b + RAG = qwen3.5:9b + RAG
- 1-0-2: gemma4:12b + RAG = qwen3.5:9b + RAG
- 1-2-0: gemma4:12b + RAG > qwen3.5:9b + RAG
- 2-0-1: gemma4:12b + RAG = qwen3.5:9b + RAG
- 2-1-0: gemma4:12b + RAG = qwen3.5:9b + RAG

This five-case diagnostic measures order sensitivity. Stable scores do not prove correctness or absence of other biases.
Execution order is shuffled with a recorded seed; there are no generation calls or validation retries.
The current benchmark dataset supplies the requests; its hash and the source-results hash are recorded.
Raw judgments, rationales, exact rubrics, dimension means and frozen inputs are in results.json.
