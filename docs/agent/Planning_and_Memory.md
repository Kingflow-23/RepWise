**Planning Strategy: Plan-and-Execute**

The agent uses the **Plan-and-Execute strategy**. It first breaks a user's request into smaller, manageable steps, then executes those steps in sequence. Once execution is complete, it checks the outcome and provides a response. If a step fails, the agent can revise its plan.

**Memory Design**

- **Working memory**: Stores the current task, plan and tool results while the agent is running.

- **Semantic memory**: Stores stable user preferences and useful facts across sessions.

- **Episodic memory**: Stores relevant past interactions, completed tasks and decisions. (Retained maybe 90days - to be confirmed)
