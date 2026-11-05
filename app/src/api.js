/** The browser uses same-origin /api routes; Vite forwards them to FastAPI locally. */
export class ApiError extends Error {
  constructor(message, status = 0) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

export async function request(path, { method = 'GET', body, token, signal } = {}) {
  let response;
  try {
    response = await fetch(`/api/v1${path}`, {
      method, signal: signal || AbortSignal.timeout(15000), credentials: 'omit', cache: 'no-store',
      headers: {
        ...(body === undefined ? {} : { 'Content-Type': 'application/json' }),
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      ...(body === undefined ? {} : { body: JSON.stringify(body) }),
    });
  } catch (error) {
    if (error.name === 'AbortError') throw error;
    throw new ApiError('We couldn’t reach RepWise. Please try again in a moment.');
  }
  let payload;
  try { payload = await response.json(); } catch { payload = null; }
  if (!response.ok) {
    const message = response.status === 401
      ? 'That name and password don’t match. Please try again.'
      : response.status === 422
        ? 'Check your account name and password, then try again.'
        : 'Sign-in is unavailable right now. Please try again in a moment.';
    throw new ApiError(message, response.status);
  }
  if (!payload || typeof payload !== 'object') throw new ApiError('RepWise returned an unexpected response. Please try again.');
  return payload;
}
