import { test } from 'node:test';
import assert from 'node:assert/strict';
import { agentRequestBody, answerParagraphs, readAgentResponse } from '../src/agent-data.js';
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
