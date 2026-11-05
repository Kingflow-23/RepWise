import { test } from 'node:test';
import assert from 'node:assert/strict';
import { planOptions, generatedRecord } from '../src/generation-data.js';
import { request } from '../src/api.js';

const profile = { available_days: ['Monday', 'Friday'] };
test('generation options trim text and send only fields accepted by PlanOptions', () => {
  assert.deepEqual(planOptions({ focus: ' Build leg strength ', count: '2', constraints: ' No jumping\r\n\n Use dumbbells ', refinement: ' Easier than last time ' }, profile), {
    focus: 'Build leg strength', workouts_per_week: 2, extra_constraints: ['No jumping', 'Use dumbbells'], refinement_instructions: 'Easier than last time',
  });
  assert.equal(planOptions({ focus: 'Build strength', count: 1 }, profile).refinement_instructions, null);
});

test('invalid requests cannot bypass profile or weekly schedule limits', () => {
  for (const count of [0, 3, 1.5, 'invalid']) assert.throws(() => planOptions({ focus: 'Build strength', count }, profile), /workout count/);
  assert.throws(() => planOptions({ focus: 'Build strength', count: 1 }, null), /profile/);
  assert.throws(() => planOptions({ focus: '  x ', count: 1 }, profile), /3–300/);
  assert.throws(() => planOptions({ focus: 'Build strength', count: 1, constraints: Array(11).fill('constraint').join('\n') }, profile), /10 constraints/);
});

test('only a completed, saved, readable generation can replace the displayed plan', () => {
  assert.throws(() => generatedRecord({ status: 'out_of_scope' }), /outside workout planning/);
  assert.throws(() => generatedRecord({ status: 'completed', plan: {} }), /wasn’t confirmed/);
  const result = { status: 'completed', plan_id: 'saved-id', plan: { title: 'Strength', summary: 'A week', safety_note: 'Stop if painful.', weeks: [{ workouts: [{ exercises: [{}] }] }] } };
  assert.equal(generatedRecord(result).id, 'saved-id');
  assert.equal(generatedRecord(result).plan, result.plan);
});

test('generation failure outcomes produce useful messages without raw diagnostics', async (t) => {
  t.mock.method(globalThis, 'fetch', async () => Response.json({ detail: { status: 'retrieval_error', message: 'private internal path' } }, { status: 503 }));
  await assert.rejects(request('/students/account/plans', { method: 'POST' }), (error) => error.outcome === 'retrieval_error' && /knowledge base/.test(error.message) && !/private/.test(error.message));
});

test('an interrupted generation explains that the server may still save a plan', async (t) => {
  t.mock.method(globalThis, 'fetch', async () => { throw new DOMException('timeout', 'TimeoutError'); });
  await assert.rejects(request('/students/account/plans', { method: 'POST' }), /Refresh your saved plan before trying again/);
});
