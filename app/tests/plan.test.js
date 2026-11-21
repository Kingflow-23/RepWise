import { test } from 'node:test';
import assert from 'node:assert/strict';
import { activePlan, draftPlans, savedDate, sourceLink } from '../src/plan-data.js';

test('only an explicitly active record can become the current program', () => {
  const record = { id: 'current', status: 'active', plan: { title: 'Leg strength', summary: 'A saved plan', safety_note: 'Stop if painful.', weeks: [{ workouts: [{ exercises: [{ name: 'Squat' }] }] }] } };
  assert.equal(activePlan(record), record);
  assert.equal(activePlan(null), null);
  for (const status of ['draft', 'archived', undefined]) assert.throws(() => activePlan({ ...record, status }), /not active/);
  assert.throws(() => activePlan({ id: 'broken', plan: { weeks: [] } }), /could not be read/);
  const draft = { ...record, status: 'draft' };
  assert.deepEqual(draftPlans([draft]), [draft]);
  assert.deepEqual(draftPlans([]), []);
  assert.throws(() => draftPlans([record]), /not a draft/);
  assert.throws(() => draftPlans(null), /invalid/);
});

test('source links allow only absolute HTTP or HTTPS addresses', () => {
  assert.equal(sourceLink('https://example.com/watch?t=30'), 'https://example.com/watch?t=30');
  for (const url of ['javascript:alert(1)', 'data:text/html,test', '/relative', 'file:///secret', undefined]) assert.equal(sourceLink(url), null);
});

test('saved dates handle UTC timestamps and omit invalid or missing timestamps', () => {
  assert.equal(savedDate('2026-11-05 12:00:00'), savedDate('2026-11-05T12:00:00Z'));
  assert.equal(savedDate('2026-11-05T14:00:00+02:00'), savedDate('2026-11-05T12:00:00Z'));
  assert.equal(savedDate('invalid'), '');
  assert.equal(savedDate(null), '');
});
