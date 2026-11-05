import { test, beforeEach, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { authenticatedRequest, onSessionChange, restoreSession, signIn, signOut, register } from '../src/auth.js';

const originalFetch = globalThis.fetch;
function token(expires = Date.now() / 1000 + 3600) {
  return `header.${Buffer.from(JSON.stringify({ exp: expires })).toString('base64url')}.signature`;
}
const identity = { user_id: 'account-1', full_name: 'Alex Test' };
let store;
beforeEach(() => {
  store = new Map();
  globalThis.sessionStorage = {
    getItem: (key) => store.get(key) || null,
    setItem: (key, value) => store.set(key, value),
    removeItem: (key) => store.delete(key),
  };
});
afterEach(() => { signOut(); globalThis.fetch = originalFetch; });

test('registration immediately accepts the returned session without a second login', async (context) => {
  let current;
  let calls = 0;
  context.after(onSessionChange((value) => { current = value; }));
  globalThis.fetch = async (url, options) => {
    calls += 1;
    assert.equal(url, '/api/v1/auth/register');
    assert.deepEqual(JSON.parse(options.body), { full_name: 'Alex Test', password: 'new password' });
    return Response.json({ ...identity, access_token: token() });
  };
  await register(' Alex Test ', 'new password');
  assert.equal(calls, 1);
  assert.equal(current.user_id, identity.user_id);
  assert(!store.get('repwise.session').includes('new password'));
});

test('duplicate registration reports the conflict without creating a session', async () => {
  globalThis.fetch = async () => Response.json({}, { status: 409 });
  await assert.rejects(register('Alex Test', 'new password'), /already registered/);
  assert.equal(store.size, 0);
});

test('sign-in posts JSON and stores a session without the password; sign-out clears it', async (context) => {
  let current;
  context.after(onSessionChange((value) => { current = value; }));
  globalThis.fetch = async (url, options) => {
    assert.equal(url, '/api/v1/auth/login');
    assert.equal(options.method, 'POST');
    assert.deepEqual(JSON.parse(options.body), { full_name: 'Alex Test', password: ' secret password ' });
    return Response.json({ ...identity, access_token: token() });
  };
  await signIn(' Alex Test ', ' secret password ');
  assert.equal(current.user_id, 'account-1');
  assert(!store.get('repwise.session').includes('secret password'));
  signOut();
  assert.equal(current, null);
  assert.equal(store.size, 0);
});

test('restoration replaces stored identity with verified backend identity', async (context) => {
  let current;
  context.after(onSessionChange((value) => { current = value; }));
  const savedToken = token();
  store.set('repwise.session', JSON.stringify({ access_token: savedToken, user_id: 'wrong', full_name: 'Untrusted name' }));
  globalThis.fetch = async (url, options) => {
    assert.equal(url, '/api/v1/auth/me');
    assert.equal(options.headers.Authorization, `Bearer ${savedToken}`);
    return Response.json(identity);
  };
  await restoreSession();
  assert.equal(current.full_name, 'Alex Test');
  assert.equal(JSON.parse(store.get('repwise.session')).user_id, 'account-1');
});

test('expired stored tokens are removed without an API call', async (context) => {
  let notice;
  context.after(onSessionChange((value, message) => { notice = message; }));
  store.set('repwise.session', JSON.stringify({ access_token: token(Date.now() / 1000 - 60) }));
  globalThis.fetch = () => assert.fail('Expired token must not be restored');
  await restoreSession();
  assert.match(notice, /expired/);
  assert.equal(store.size, 0);
});

test('a rejected restored session is cleared', async () => {
  store.set('repwise.session', JSON.stringify({ access_token: token() }));
  globalThis.fetch = async () => Response.json({ detail: 'Invalid token' }, { status: 401 });
  await restoreSession();
  assert.equal(store.size, 0);
});

test('incorrect credentials never create a session', async () => {
  globalThis.fetch = async () => Response.json({ detail: 'Invalid name or password' }, { status: 401 });
  await assert.rejects(signIn('Alex Test', 'wrong password'), /don’t match/);
  assert.equal(store.size, 0);
});

test('authenticated requests clear a session rejected by the backend', async () => {
  globalThis.fetch = async () => Response.json({ ...identity, access_token: token() });
  await signIn('Alex Test', 'correct password');
  globalThis.fetch = async () => Response.json({ detail: 'Expired' }, { status: 401 });
  await assert.rejects(authenticatedRequest('/auth/me'));
  assert.equal(store.size, 0);
});

test('incomplete account responses cannot create a session', async () => {
  globalThis.fetch = async () => Response.json({ ...identity });
  await assert.rejects(signIn('Alex Test', 'correct password'), /incomplete/);
  assert.equal(store.size, 0);
});

test('storage restrictions still allow an in-memory sign-in', async () => {
  globalThis.sessionStorage = { getItem() { throw new Error('blocked'); }, setItem() { throw new Error('blocked'); }, removeItem() { throw new Error('blocked'); } };
  globalThis.fetch = async () => Response.json({ ...identity, access_token: token() });
  await signIn('Alex Test', 'correct password');
  globalThis.fetch = async (url, options) => {
    assert.match(options.headers.Authorization, /^Bearer /);
    return Response.json(identity);
  };
  assert.deepEqual(await authenticatedRequest('/auth/me'), identity);
});
