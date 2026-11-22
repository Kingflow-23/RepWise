import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  actionLabel, agentRequestBody, answerParagraphs, decisionErrorMessage, decisionPath, expiryText,
  readAgentResponse, readDecision, readPendingActions,
} from '../src/agent-data.js';

test('decision paths are encoded and limited to approve or decline', () => {
  assert.equal(decisionPath('user 1', 'a/b', 'approve'), '/students/user%201/agent/actions/a%2Fb/approve');
  assert.equal(decisionPath('u', 'a', 'decline'), '/students/u/agent/actions/a/decline');
  assert.throws(() => decisionPath('u', 'a', 'delete'), /Unknown decision/);
});

test('decisions are explained, including a change that could not be applied', () => {
  assert.deepEqual(readDecision({ executed: true, action: { status: 'executed', tool_name: 'activate_plan' } }, 'approve', 'Make a draft plan your active program'),
    { ok: true, message: 'Done: Make a draft plan your active program. Your active program was updated.' });
  assert.deepEqual(readDecision({ executed: false, action: { status: 'failed', result: { ok: false, message: 'Saved plan not found.' } } }, 'approve'),
    { ok: false, message: 'RepWise couldn’t apply this change. Saved plan not found.' });
  assert.equal(readDecision({ executed: false, action: { status: 'failed', result: { message: 'x'.repeat(300) } } }, 'approve').message, 'RepWise couldn’t apply this change.');
  assert.match(readDecision({ executed: false, action: { status: 'declined' } }, 'decline', 'Move workouts').message, /^Declined: Move workouts\. Nothing was changed\.$/);
  assert.throws(() => readDecision({}, 'approve'), /unexpected decision/);
});

test('refused decisions get a message per HTTP status', () => {
  assert.match(decisionErrorMessage(404), /no longer exists/);
  assert.match(decisionErrorMessage(409), /already decided/);
  assert.match(decisionErrorMessage(410), /expired/);
  assert.match(decisionErrorMessage(0), /may or may not have been applied/);
  assert.match(decisionErrorMessage(500), /couldn’t record your decision/);
});

const NOW = Date.parse('2026-11-23T10:00:00Z');
const action = (fields) => ({ id: 'a', tool_name: 'activate_plan', reason: 'Activate the draft.', status: 'pending', created_at: '2026-11-23T09:50:00Z', expires_at: '2026-11-23T10:20:00Z', ...fields });

test('only pending, unexpired, well-formed actions are kept, newest first', () => {
  const list = readPendingActions([
    action({ id: 'old', created_at: '2026-11-23T09:40:00Z' }),
    action({ id: 'new', created_at: '2026-11-23T09:55:00Z', tool_name: 'organize_schedule' }),
    action({ id: 'expired', expires_at: '2026-11-23T09:59:00Z' }),
    action({ id: 'done', status: 'executed' }),
    action({ id: '', reason: 'no id' }),
    { id: 'no-reason', status: 'pending' },
    null,
  ], NOW);
  assert.deepEqual(list.map((item) => item.id), ['new', 'old']);
  assert.equal(list[0].label, 'Move workouts in your week');
});

test('an unexpected actions payload is an error, an empty list is not', () => {
  assert.throws(() => readPendingActions({ detail: 'x' }), /unexpected list/);
  assert.deepEqual(readPendingActions([], NOW), []);
});

test('action labels and expiry text are readable', () => {
  assert.equal(actionLabel('add_workouts_to_calendar'), 'Add workouts to your calendar');
  assert.equal(actionLabel('something_new'), 'Apply a change');
  assert.equal(expiryText(NOW + 20 * 60000, NOW), 'Expires in 20 min');
  assert.equal(expiryText(NOW + 30000, NOW), 'Expires in less than a minute');
  assert.match(expiryText(NOW + 3 * 3600000, NOW), /^Expires at /);
});
import { request } from '../src/api.js';

test('a request body trims the message and only sends accepted fields', () => {
  assert.deepEqual(agentRequestBody('  Find a gym near me  ', null, 'Europe/Paris'), { message: 'Find a gym near me', timezone: 'Europe/Paris' });
  assert.deepEqual(agentRequestBody('Continue', 'conv-1', null), { message: 'Continue', conversation_id: 'conv-1' });
  assert.equal(agentRequestBody('Hi', 'x'.repeat(101), null).conversation_id, undefined);
});

test('empty or overlong messages are refused before any request', () => {
  assert.throws(() => agentRequestBody('   '), /Write a message/);
  assert.throws(() => agentRequestBody(undefined), /Write a message/);
  assert.throws(() => agentRequestBody('x'.repeat(1001)), /1000 characters/);
  assert.equal(agentRequestBody('x'.repeat(1000), null, null).message.length, 1000);
});

test('a response keeps the answer, the steps, pending actions and the conversation', () => {
  const result = readAgentResponse({
    answer: 'Here is your plan.\n\n**Note:** Waiting for your approval.',
    conversation_id: 'conv-7',
    steps: [{ agent: 'planning', task: 'Create', status: 'completed' }, { agent: 'location', task: 'Find', status: 'failed' }, { agent: 'schedule', task: 'Move', status: 'not_run' }],
    pending_actions: [{ action_id: 'a-1', tool_name: 'activate_plan', reason: 'Activate the new plan', expires_at: '2026-11-22T12:00:00Z' }, { reason: 'no id' }],
    health_concern: true,
  });
  assert.equal(result.conversationId, 'conv-7');
  assert.deepEqual(result.steps.map((step) => [step.label, step.state]), [['Workout planning', 'ok'], ['Location', 'failed'], ['Schedule', 'skipped']]);
  assert.deepEqual(result.pending, [{ id: 'a-1', tool: 'activate_plan', reason: 'Activate the new plan', expiresAt: '2026-11-22T12:00:00Z' }]);
  assert.equal(result.healthConcern, true);
  assert.equal(result.changedPlans, true);
});

test('plans are refreshed only after a completed planning step', () => {
  const base = { answer: 'Done.' };
  assert.equal(readAgentResponse({ ...base, steps: [{ agent: 'planning', status: 'failed' }] }).changedPlans, false);
  assert.equal(readAgentResponse({ ...base, steps: [{ agent: 'location', status: 'completed' }] }).changedPlans, false);
  assert.equal(readAgentResponse(base).changedPlans, false);
});

test('an answer without text is an error', () => {
  for (const payload of [null, {}, { answer: '  ' }, { answer: 42 }]) assert.throws(() => readAgentResponse(payload), /incomplete answer/);
});

test('answers become plain paragraphs without Markdown markers', () => {
  assert.deepEqual(answerParagraphs('First.\n\n**Note:** <b>kept as text</b>\n\n\n'), ['First.', 'Note: <b>kept as text</b>']);
});

test('approval notes are not repeated next to the approval box', () => {
  const answer = 'Saved a draft.\n\n**Note:** Waiting for your approval: activate it. (action a-1, expires soon)\n\n**Note:** General fitness guidance only.';
  assert.deepEqual(answerParagraphs(answer, ['a-1']), ['Saved a draft.', 'Note: General fitness guidance only.']);
  assert.equal(answerParagraphs(answer).length, 3);
});

test('assistant failures get assistant messages, not profile messages', async (t) => {
  t.mock.method(globalThis, 'fetch', async () => Response.json({ detail: [] }, { status: 422 }));
  await assert.rejects(request('/students/a/agent/requests', { method: 'POST', body: {} }), /1 and 1000 characters/);
  t.mock.method(globalThis, 'fetch', async () => Response.json({ detail: 'boom' }, { status: 500 }));
  await assert.rejects(request('/students/a/agent/requests', { method: 'POST', body: {} }), /assistant is unavailable/);
  t.mock.method(globalThis, 'fetch', async () => { throw new DOMException('timeout', 'TimeoutError'); });
  await assert.rejects(request('/students/a/agent/requests', { method: 'POST', body: {} }), /may still finish this request/);
});

test('stopping the wait is reported as an abort, not as a failure message', async (t) => {
  t.mock.method(globalThis, 'fetch', async () => { throw new DOMException('stopped', 'AbortError'); });
  await assert.rejects(request('/students/a/agent/requests', { method: 'POST', body: {} }), (error) => error.name === 'AbortError');
});
