import { ApiError, request } from './api.js';

const STORAGE_KEY = 'repwise.session';
let session = null;
let expirationTimer;
let revision = 0;
const listeners = new Set();

function publish(message = '') {
  listeners.forEach((listener) => listener(session ? { ...session } : null, message));
}

function persist(value) {
  // Tab-scoped persistence only; password values are never stored.
  try {
    if (value) sessionStorage.setItem(STORAGE_KEY, JSON.stringify(value));
    else sessionStorage.removeItem(STORAGE_KEY);
  } catch { /* In restricted browsers, the current in-memory session still works. */ }
}

function expiresAt(token) {
  // This schedules the UI timeout only. FastAPI verifies the signature and claims.
  try {
    const encoded = token.split('.')[1].replace(/-/g, '+').replace(/_/g, '/');
    const payload = JSON.parse(atob(encoded));
    return Number.isFinite(payload.exp) ? payload.exp * 1000 : 0;
  } catch { return 0; }
}

function accept(value) {
  if (typeof value.access_token !== 'string' || !value.access_token
      || typeof value.user_id !== 'string' || !value.user_id
      || typeof value.full_name !== 'string' || !value.full_name.trim()) {
    throw new ApiError('RepWise returned an incomplete account response. Please try again.');
  }
  const delay = expiresAt(value.access_token) - Date.now();
  if (delay <= 0) throw new ApiError('Your session has expired. Please sign in again.', 401);
  clearTimeout(expirationTimer);
  session = { access_token: value.access_token, user_id: value.user_id, full_name: value.full_name };
  persist(session);
  expirationTimer = setTimeout(() => signOut('Your session has expired. Please sign in again.'), Math.min(delay, 2 ** 31 - 1));
  publish();
}

export function onSessionChange(listener) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function signOut(message = '') {
  revision += 1;
  clearTimeout(expirationTimer);
  session = null;
  persist(null);
  publish(message);
}

export async function signIn(fullName, password) {
  const current = ++revision;
  const value = await request('/auth/login', { method: 'POST', body: { full_name: fullName.trim(), password } });
  if (revision === current) accept(value);
}

export async function restoreSession() {
  let stored;
  try { stored = JSON.parse(sessionStorage.getItem(STORAGE_KEY)); } catch { persist(null); }
  if (!stored?.access_token) { persist(null); return; }
  if (expiresAt(stored.access_token) <= Date.now()) {
    signOut('Your session has expired. Please sign in again.');
    return;
  }
  const current = ++revision;
  try {
    const identity = await request('/auth/me', { token: stored.access_token });
    // Identity is always refreshed from the authenticated backend, not trusted storage.
    if (revision === current) accept({ ...identity, access_token: stored.access_token });
  } catch (error) {
    if (revision === current) {
      signOut(error.status === 401 ? 'Your session has expired. Please sign in again.'
        : 'We couldn’t restore your session. Please sign in again.');
    }
  }
}

export async function authenticatedRequest(path, options = {}) {
  if (!session) throw new ApiError('Please sign in to continue.', 401);
  const token = session.access_token;
  try { return await request(path, { ...options, token }); }
  catch (error) {
    if (error.status === 401 && session?.access_token === token) signOut('Your session has expired. Please sign in again.');
    throw error;
  }
}
