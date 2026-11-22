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

const ACTION_LABELS = {
  activate_plan: 'Make a draft plan your active program',
  organize_schedule: 'Move workouts in your week',
  add_workouts_to_calendar: 'Add workouts to your calendar',
  update_upcoming_workout: 'Change an upcoming workout',
  save_workout_feedback: 'Save workout feedback',
};

/** A readable name for the tool an action would run. */
export function actionLabel(tool) {
  return ACTION_LABELS[tool] || 'Apply a change';
}

/**
 * Keep the actions from GET /agent/actions that can still be approved:
 * status pending and not yet expired, newest first. Malformed rows are dropped.
 */
export function readPendingActions(payload, now = Date.now()) {
  if (!Array.isArray(payload)) throw new Error('RepWise returned an unexpected list of actions.');
  return payload
    .filter((action) => action && typeof action.id === 'string' && action.id && typeof action.reason === 'string' && action.status === 'pending')
    .map((action) => ({
      id: action.id,
      tool: String(action.tool_name || ''),
      label: actionLabel(action.tool_name),
      reason: action.reason,
      createdAt: Date.parse(action.created_at) || 0,
      expiresAt: Date.parse(action.expires_at) || 0,
    }))
    .filter((action) => action.expiresAt > now)
    .sort((a, b) => b.createdAt - a.createdAt);
}

/** Path of the approve or decline route for one action. */
export function decisionPath(userId, actionId, decision) {
  if (decision !== 'approve' && decision !== 'decline') throw new Error('Unknown decision.');
  return `/students/${encodeURIComponent(userId)}/agent/actions/${encodeURIComponent(actionId)}/${decision}`;
}

/** Turn an ActionDecision into a short, readable outcome. */
export function readDecision(payload, decision, label = 'this change') {
  const action = payload?.action;
  if (!action || typeof action.status !== 'string') throw new Error('RepWise returned an unexpected decision.');
  if (decision === 'decline') return { ok: true, message: `Declined: ${label}. Nothing was changed.` };
  if (payload.executed === true && action.status === 'executed') {
    const extra = action.tool_name === 'activate_plan' ? ' Your active program was updated.' : '';
    return { ok: true, message: `Done: ${label}.${extra}` };
  }
  // The tool ran but refused (for example, the draft no longer exists).
  const reason = typeof action.result?.message === 'string' && action.result.message.length <= 200 ? ` ${action.result.message}` : '';
  return { ok: false, message: `RepWise couldn’t apply this change.${reason}` };
}

/** Explain why the server refused a decision, from its HTTP status. */
export function decisionErrorMessage(status) {
  if (status === 404) return 'This action no longer exists. The list was refreshed.';
  if (status === 409) return 'This action was already decided. The list was refreshed.';
  if (status === 410) return 'This action has expired. Ask RepWise again if you still want it.';
  if (status === 401) return 'Your session has expired. Please sign in again.';
  if (!status) return 'The connection was interrupted, so the change may or may not have been applied. Your programs and the list were refreshed.';
  return 'RepWise couldn’t record your decision. Please try again in a moment.';
}

/** "Expires in 12 min" for soon, otherwise "Expires at 23:40". */
export function expiryText(expiresAt, now = Date.now()) {
  const minutes = Math.ceil((expiresAt - now) / 60000);
  if (minutes <= 1) return 'Expires in less than a minute';
  if (minutes < 60) return `Expires in ${minutes} min`;
  return `Expires at ${new Date(expiresAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;
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
