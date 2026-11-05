import { test } from 'node:test';
import assert from 'node:assert/strict';
import { request } from '../src/api.js';

test('profile requests send the bearer token and JSON to the saved-profile endpoint', async (t) => {
  const profile = { goal: 'Build strength', available_days: ['Monday'] };
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    assert.equal(url, '/api/v1/students/account-123/profile');
    assert.equal(options.method, 'PUT');
    assert.equal(options.headers.Authorization, 'Bearer session-token');
    assert.equal(options.credentials, 'omit');
    assert.deepEqual(JSON.parse(options.body), profile);
    return new Response(JSON.stringify(profile), { status: 200 });
  });
  assert.deepEqual(await request('/students/account-123/profile', { method: 'PUT', token: 'session-token', body: profile }), profile);
});

test('a missing profile stays distinguishable from a failed load', async (t) => {
  t.mock.method(globalThis, 'fetch', async () => new Response('{}', { status: 404 }));
  await assert.rejects(request('/students/account-123/profile'), (error) => error.status === 404);
});

test('profile validation failures have profile wording and do not expose server diagnostics', async (t) => {
  t.mock.method(globalThis, 'fetch', async () => new Response('{"detail":"internal diagnostics"}', { status: 422 }));
  await assert.rejects(request('/students/account-123/profile'), (error) => error.status === 422 && /profile fields/.test(error.message) && !/password|diagnostics/.test(error.message));
});
