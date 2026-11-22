import { test } from 'node:test';
import assert from 'node:assert/strict';
import { planOptions, generatedRecord } from '../src/generation-data.js';
import { request, generationFailureMessage } from '../src/api.js';

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
  const result = { status: 'completed', plan_id: 'saved-id', plan_status: 'active', plan: { title: 'Strength', summary: 'A week', safety_note: 'Stop if painful.', weeks: [{ workouts: [{ exercises: [{}] }] }] } };
  assert.equal(generatedRecord(result).id, 'saved-id');
  assert.equal(generatedRecord(result).plan, result.plan);
  assert.throws(() => generatedRecord({ ...result, plan_status: 'draft' }), /not active/);
});

test('generation failure outcomes produce useful messages without raw diagnostics', async (t) => {
  t.mock.method(globalThis, 'fetch', async () => Response.json({ detail: { status: 'retrieval_error', message: 'private internal path' } }, { status: 503 }));
  await assert.rejects(request('/students/account/plans', { method: 'POST' }), (error) => error.outcome === 'retrieval_error' && /knowledge base/.test(error.message) && !/private/.test(error.message));
});

test('each generation failure status gets its own message and HTTP code is kept', async (t) => {
  const cases = [
    [503, 'provider_unavailable', 'Ollama is not reachable. Start it with `ollama serve`.', /Ollama\) isn’t reachable/],
    [503, 'model_not_found', 'The configured model qwen3.5:9b is not installed in Ollama.', /\(qwen3\.5:9b\) isn’t installed, so trying again won’t help/],
    [504, 'generation_timeout', 'The model did not finish within 300 s.', /took too long/],
    [502, 'generation_error', 'The model did not return a usable plan after 3 attempt(s). Last error: x', /couldn’t be read/],
    [502, 'plan_not_compliant', 'The generated plan did not meet your requirements after 3 attempt(s): a workout is longer than the requested session duration', /\(a workout is longer than the requested session duration\)/],
  ];
  for (const [httpStatus, status, message, expected] of cases) {
    t.mock.method(globalThis, 'fetch', async () => Response.json({ detail: { status, message } }, { status: httpStatus }));
    await assert.rejects(request('/students/account/plans', { method: 'POST' }), (error) => error.outcome === status
      && error.status === httpStatus && expected.test(error.message) && /request and saved plans are unchanged/.test(error.message));
  }
});

test('server text that does not match the expected shape is never shown', () => {
  assert.doesNotMatch(generationFailureMessage('model_not_found', 'C:\\private\\path <script>'), /private|script/);
  assert.doesNotMatch(generationFailureMessage('plan_not_compliant', 'after 1 attempt(s): Traceback C:\\x\\y.py line 3'), /Traceback|\.py/);
  assert.match(generationFailureMessage('plan_not_compliant', undefined), /broke one of your requirements\./);
  assert.equal(generationFailureMessage('unknown_status', 'anything'), null);
});

test('an interrupted generation explains that the server may still save a plan', async (t) => {
  t.mock.method(globalThis, 'fetch', async () => { throw new DOMException('timeout', 'TimeoutError'); });
  await assert.rejects(request('/students/account/plans', { method: 'POST' }), /Refresh your saved plan before trying again/);
});
