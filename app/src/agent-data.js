/** Pure helpers for the assistant chat: request bodies and response reading. */

export const AGENT_MESSAGE_LIMIT = 1000;

const AGENT_LABELS = {
  planning: 'Workout planning',
  performance: 'Performance',
  schedule: 'Schedule',
  location: 'Location',
  knowledge: 'Fitness knowledge',
};

const STEP_STATES = {
  completed: { label: 'done', state: 'ok' },
  failed: { label: 'failed', state: 'failed' },
  limit_reached: { label: 'stopped at a limit', state: 'failed' },
  not_run: { label: 'not run', state: 'skipped' },
};

/** Return the browser's IANA time zone, or null when it is unavailable. */
export function browserTimezone() {
  try {
    const zone = Intl.DateTimeFormat().resolvedOptions().timeZone;
    return typeof zone === 'string' && zone.length <= 60 ? zone : null;
  } catch { return null; }
}

/**
 * Build the body of POST /agent/requests.
 * Throws a readable error when the message is empty or too long.
 */
export function agentRequestBody(message, conversationId = null, timezone = browserTimezone()) {
  const text = typeof message === 'string' ? message.trim() : '';
  if (!text) throw new Error('Write a message for RepWise first.');
  if (text.length > AGENT_MESSAGE_LIMIT) throw new Error(`Keep your message within ${AGENT_MESSAGE_LIMIT} characters.`);
  const body = { message: text };
  if (typeof conversationId === 'string' && conversationId && conversationId.length <= 100) body.conversation_id = conversationId;
  if (typeof timezone === 'string' && timezone) body.timezone = timezone;
  return body;
}

/**
 * Keep only what the chat shows from an AgentResponse.
 * Unknown or malformed parts are dropped; a missing answer is an error.
 */
export function readAgentResponse(payload) {
  if (!payload || typeof payload.answer !== 'string' || !payload.answer.trim()) {
    throw new Error('RepWise returned an incomplete answer. Please try again.');
  }
  const steps = Array.isArray(payload.steps) ? payload.steps
    .filter((step) => step && typeof step.agent === 'string')
    .map((step) => {
      const outcome = STEP_STATES[step.status] || { label: 'unknown', state: 'skipped' };
      return { agent: step.agent, label: AGENT_LABELS[step.agent] || step.agent, status: outcome.label, state: outcome.state };
    }) : [];
  const pending = Array.isArray(payload.pending_actions) ? payload.pending_actions
    .filter((action) => action && typeof action.action_id === 'string' && typeof action.reason === 'string')
    .map((action) => ({ id: action.action_id, tool: String(action.tool_name || ''), reason: action.reason, expiresAt: action.expires_at || null })) : [];
  return {
    answer: payload.answer.trim(),
    conversationId: typeof payload.conversation_id === 'string' && payload.conversation_id ? payload.conversation_id : null,
    steps,
    pending,
    healthConcern: payload.health_concern === true,
    // A planning step may have saved a new active plan or a draft.
    changedPlans: steps.some((step) => step.agent === 'planning' && step.state === 'ok'),
  };
}

/**
 * Split an answer into paragraphs and strip the Markdown bold markers the API adds.
 * Notes naming a pending action ID are left out: the approval box shows them.
 */
export function answerParagraphs(answer, pendingIds = []) {
  return String(answer).split(/\n{2,}/)
    .map((part) => part.replace(/\*\*(.+?)\*\*/g, '$1').trim())
    .filter((part) => part && !(part.startsWith('Note:') && pendingIds.some((id) => id && part.includes(id))));
}
