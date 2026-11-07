**Planning Strategy: Plan-and-Execute**

The agent uses the **Plan-and-Execute strategy**. It first breaks a user's request into smaller, manageable steps, then executes those steps in sequence. Once execution is complete, it checks the outcome and provides a response. If a step fails, the agent can revise its plan.

**Memory Design**

- **Working memory**: Stores the current task, plan and tool results while the agent is running.

- **Semantic memory**: Stores stable user preferences and useful facts across sessions.

- **Episodic memory**: Stores relevant past interactions, completed tasks and decisions. (Retained maybe 90days - to be confirmed)

---

**Implementation decisions** (how the design above was implemented; details in [the agents guide](../agents.md)):

- Plan-and-Execute is the orchestrator's LangGraph graph: load context, plan, delegate to one specialist per step, check, replan once on failure, respond, remember. Every loop has step and time limits.
- Working memory is the graph's state for one request; conversation turns are kept per conversation ID while the process runs.
- Semantic memory is `memory_facts`, written only when the student asks; profile data is not duplicated there.
- Episodic memory is `memory_episodes`, one code-written line per request. **Retention is 90 days**, configurable with `REPWISE_EPISODE_RETENTION_DAYS`.
- Health details (pain, injury, medical) are never stored in long-term memory.
