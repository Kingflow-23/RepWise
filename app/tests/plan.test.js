import { test } from 'node:test';
import assert from 'node:assert/strict';
import { latestPlan, savedDate, sourceLink } from '../src/plan-data.js';

test('the server-ordered newest record is used, with an explicit empty state', () => {
  const record = { id: 'newest', plan: { title: 'Leg strength', summary: 'A saved plan', safety_note: 'Stop if painful.', weeks: [{ workouts: [{ exercises: [{ name: 'Squat' }] }] }] } };
  assert.equal(latestPlan([record]), record);
  assert.equal(latestPlan([]), null);
  assert.throws(() => latestPlan({ detail: 'error' }), /invalid/);
  assert.throws(() => latestPlan([{ id: 'broken', plan: { weeks: [] } }]), /could not be read/);
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
