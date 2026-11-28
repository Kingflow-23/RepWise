import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  performanceBody, performanceErrorMessage, performanceLine, performancePath, readHistory, readPerformance,
} from '../src/performance-data.js';
import { request } from '../src/api.js';

test('the performance path is encoded', () => {
  assert.equal(performancePath('u 1', 'p/1', 1, 0), '/students/u%201/plans/p%2F1/weeks/1/workouts/0/performance');
});

test('empty rows are skipped and filled rows become records', () => {
  assert.deepEqual(performanceBody([
    { name: 'Squat', sets: '3', reps: ' 10, 10, 8 ', weight: '42,5' },
    { name: 'Push-up', sets: '', reps: '', weight: '' },
    { name: 'Plank', sets: '2', reps: '30 s', weight: '' },
  ]), { exercises: [
    { exercise_name: 'Squat', sets_completed: 3, reps_per_set: '10, 10, 8', weight_kg: 42.5 },
    { exercise_name: 'Plank', sets_completed: 2, reps_per_set: '30 s', weight_kg: null },
  ] });
});

test('incomplete or out-of-range rows are refused with the exercise name', () => {
  assert.throws(() => performanceBody([{ name: 'Squat', sets: '', reps: '10', weight: '' }]), /Squat: enter the number of sets/);
  assert.throws(() => performanceBody([{ name: 'Squat', sets: '3', reps: '', weight: '' }]), /Squat: enter your reps/);
  assert.throws(() => performanceBody([{ name: 'Squat', sets: '2.5', reps: '10', weight: '' }]), /number of sets/);
  assert.throws(() => performanceBody([{ name: 'Squat', sets: '3', reps: '10', weight: '1001' }]), /between 0 and 1000 kg/);
  assert.throws(() => performanceBody([{ name: 'Squat', sets: '', reps: '', weight: '' }]), /at least one exercise/);
});

test('saved records are indexed by name for prefilling', () => {
  const saved = readPerformance([{ exercise_name: 'Bodyweight squat', sets_completed: 3, reps_per_set: '10', weight_kg: null }, null]);
  assert.equal(saved.get('bodyweight squat').sets_completed, 3);
  assert.throws(() => readPerformance({}), /unexpected/);
});

test('the training log shows status and readable performance lines', () => {
  assert.equal(performanceLine({ exercise_name: 'Squat', sets_completed: 1, reps_per_set: '12', weight_kg: 40 }), 'Squat · 1 set · 12 · 40 kg');
  const history = readHistory({
    summary: { workouts_logged: 2 },
    workouts: [
      { week_number: 1, day: 'Monday', focus: 'Legs', completion_status: 'partially_completed', performance: [{ exercise_name: 'Squat', sets_completed: 3, reps_per_set: '10', weight_kg: null }] },
      { week_number: 1, day: 'Friday', focus: 'Upper', completion_status: null, performance: [] },
    ],
  });
  assert.deepEqual(history.workouts.map((entry) => [entry.title, entry.status, entry.lines]), [
    ['Week 1 · Monday · Legs', 'Partly done', ['Squat · 3 sets · 10']],
    ['Week 1 · Friday · Upper', 'No feedback yet', []],
  ]);
  assert.throws(() => readHistory({ workouts: [] }), /unexpected training history/);
});

test('a refused save shows the exercise list from the API, nothing else raw', async (t) => {
  t.mock.method(globalThis, 'fetch', async () => Response.json({ detail: { code: 'exercise_not_found', message: "'Deadlift' is not in this workout. It has: Squat. Nothing was saved." } }, { status: 422 }));
  await assert.rejects(request('/students/u/plans/p/weeks/1/workouts/0/performance', { method: 'POST', body: {} }), (error) => {
    assert.match(performanceErrorMessage(error.status, error.detail), /It has: Squat\. Nothing was saved\./);
    return true;
  });
  assert.match(performanceErrorMessage(422, { code: 'invalid_request', message: 'raw' }), /Check the numbers/);
  assert.match(performanceErrorMessage(404, null), /no longer in your active program/);
  assert.match(performanceErrorMessage(503, null), /couldn’t be saved/);
});
