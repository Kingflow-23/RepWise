/** The browser uses same-origin /api routes; Vite forwards them to FastAPI locally. */
export class ApiError extends Error {
  constructor(message, status = 0, outcome = null) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.outcome = outcome;
  }
}

const UNCHANGED = 'Your request and saved plans are unchanged.';

/**
 * Explain a failed generation from its backend status. Server text is never shown
 * as is: only a model name or a validation rule that passes a strict pattern.
 */
export function generationFailureMessage(outcome, serverMessage = '') {
  const text = typeof serverMessage === 'string' ? serverMessage : '';
  switch (outcome) {
    case 'retrieval_error':
      return `The training knowledge base is unavailable. Please try again once it is connected. ${UNCHANGED}`;
    case 'insufficient_evidence':
      return `There isn’t enough supporting evidence for this request. Try a different training focus. ${UNCHANGED}`;
    case 'provider_unavailable':
      return `The AI model service (Ollama) isn’t reachable. Start it, then try again. ${UNCHANGED}`;
    case 'model_not_found': {
      const name = text.match(/model ([\w.:/-]{1,80}) is not installed/)?.[1];
      return `The configured AI model${name ? ` (${name})` : ''} isn’t installed, so trying again won’t help yet. Install it with Ollama or change the configured model. ${UNCHANGED}`;
    }
    case 'generation_timeout':
      return `The AI model took too long to answer. Try again later, or with fewer constraints. ${UNCHANGED}`;
    case 'generation_error':
      return `The model’s answer couldn’t be read, even after corrections. Try again or simplify your request. ${UNCHANGED}`;
    case 'plan_not_compliant': {
      const rule = text.match(/attempt\(s\): ([a-z0-9 ,'-]{3,160})$/i)?.[1];
      return `The model kept producing a plan that broke one of your requirements${rule ? ` (${rule})` : ''}. Try again or loosen a constraint. ${UNCHANGED}`;
    }
    default:
      return null;
  }
}

export async function request(path, { method = 'GET', body, token, signal, timeout = 15000 } = {}) {
  const generation = method === 'POST' && /^\/students\/[^/]+\/plans$/.test(path);
  let response;
  try {
    response = await fetch(`/api/v1${path}`, {
      method, signal: signal || AbortSignal.timeout(timeout), credentials: 'omit', cache: 'no-store',
      headers: {
        ...(body === undefined ? {} : { 'Content-Type': 'application/json' }),
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      ...(body === undefined ? {} : { body: JSON.stringify(body) }),
    });
  } catch (error) {
    if (error.name === 'AbortError') throw error;
    if (generation) throw new ApiError('The connection was interrupted or timed out. Your plan may still be saved. Refresh your saved plan before trying again.');
    throw new ApiError('We couldn’t reach RepWise. Please try again in a moment.');
  }
  let payload;
  let decoded = false;
  try { payload = await response.json(); decoded = true; } catch { payload = null; }
  if (!response.ok) {
    if (generation) {
      const outcome = payload?.detail?.status;
      const message = response.status === 401 ? 'Your session has expired. Please sign in again.'
        : response.status === 404 ? 'Save your training profile before generating a plan.'
          : generationFailureMessage(outcome, payload?.detail?.message) || (response.status === 422 ? 'Check your request and make sure the workout count fits your available days.'
            : 'Plan generation is unavailable right now. Please try again in a moment.');
      throw new ApiError(message, response.status, outcome);
    }
    const login = path === '/auth/login';
    const registration = path === '/auth/register';
    const message = (login || registration) && response.status === 404
      ? 'The account service isn’t connected. Restart the frontend server, then try again.'
      : registration && response.status === 409
        ? 'That account name is already registered. Sign in or choose another name.'
      : registration && response.status === 422
        ? 'Use a name with 3–100 characters and a password with 8–200 characters.'
      : registration
        ? 'Account creation is unavailable right now. Please try again in a moment.'
      : !login
      ? (response.status === 401 ? 'Your session has expired. Please sign in again.'
        : response.status === 422 ? 'Check your profile fields and try again.'
          : 'We couldn’t complete this request. Please try again in a moment.')
      : response.status === 401
      ? 'That name and password don’t match. Please try again.'
      : response.status === 422
        ? 'Check your account name and password, then try again.'
        : 'Sign-in is unavailable right now. Please try again in a moment.';
    throw new ApiError(message, response.status);
  }
  if (payload === null && method === 'GET' && path.endsWith('/plans/current') && response.status === 200
      && response.headers.get('content-type')?.includes('application/json')) {
    // Only a successfully decoded JSON null means there is no active plan.
    if (decoded) return null;
  }
  if (!payload || typeof payload !== 'object') throw new ApiError('RepWise returned an unexpected response. Please try again.');
  return payload;
}
