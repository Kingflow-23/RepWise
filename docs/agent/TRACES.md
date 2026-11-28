# Langfuse run notes

Run `python -m scripts.run_agent` after configuring the project keys in the
ignored `.env`. The program prints call-level metrics and flushes the traces.
The five requests ran on 2026-11-24 with the configured Langfuse project.
Count generations for model calls and spans named `agent request: ...` for
model-requested tool calls. Use the root request duration and reported usage
details for totals. All generations in this run reported usage; totals below
sum input and output tokens.

| # | Request | Model calls | Tool calls | Total tokens | Total time | Langfuse trace |
|---|---|---:|---:|---:|---:|---|
| 1 | Find the nearest gym | 4 | 1 | 6,585 | 146.922 s | [Trace](https://cloud.langfuse.com/project/cmwv8s6o50ctrad0js5otakqt/traces/0f4a0f5ca96df3d8ff7d968e16496189) |
| 2 | Find a nearby store selling dumbbells | 4 | 1 | 6,874 | 29.663 s | [Trace](https://cloud.langfuse.com/project/cmwv8s6o50ctrad0js5otakqt/traces/550da37ab0e0e7db17dafa7b19be4187) |
| 3 | Find a nearby store selling creatine | 5 | 1 | 6,606 | 17.530 s | [Trace](https://cloud.langfuse.com/project/cmwv8s6o50ctrad0js5otakqt/traces/685063214dd747640c81836b7819674d) |
| 4 | Organize this week's sessions around the current plan | 4 | 1 | 8,697 | 41.842 s | [Trace](https://cloud.langfuse.com/project/cmwv8s6o50ctrad0js5otakqt/traces/e734a611e92ee4ca3c1710e6439cce5b) |
| 5 | Create a leg-muscle-building plan | 6 | 1 | 24,504 | 302.545 s | [Trace](https://cloud.langfuse.com/project/cmwv8s6o50ctrad0js5otakqt/traces/10cedcb9fc6c43b77d2a95cb33248f95) |

## Trace issue and fix

Request 4's final answer invented a Sources section and literal location/time
placeholders (`[Lieu 1]`, `[Créneau 1]`) despite having no place or knowledge
results. It also duplicated the schedule details that the application appends
from verified tool output. This is visible in the [request 4 trace](https://cloud.langfuse.com/project/cmwv8s6o50ctrad0js5otakqt/traces/e734a611e92ee4ca3c1710e6439cce5b).

Tightened the responder instructions against placeholders and duplicate
code-appended places/times. Also added a deterministic guard that removes a
generated Sources section unless the knowledge specialist returned a title and
URL. Re-ran request 4 after the prompt change; the bogus heading still appeared
in the raw model answer, demonstrating why the guard is necessary. The final
response was re-run after the guard was added: the trace shows 6 model calls,
2 tool calls, 12,529 tokens, and 31.746 s, with no unverified Sources heading
or placeholder entries in the final answer. [Post-fix validation trace](https://cloud.langfuse.com/project/cmwv8s6o50ctrad0js5otakqt/traces/5fe2d082a223f827344e2b0f8ece5b11).
