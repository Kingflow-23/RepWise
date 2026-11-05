import { onSessionChange, restoreSession, signIn, signOut } from './auth.js';
import { icon } from './icons.js';
import './login.css';

export function initializeLogin() {
  const form = document.getElementById('login-form');
  const name = document.getElementById('login-name');
  const password = document.getElementById('login-password');
  const submit = document.getElementById('login-submit');
  const error = document.getElementById('login-error');
  const status = document.getElementById('login-status');
  const panel = document.getElementById('login-panel');
  const menu = document.getElementById('account-menu');
  const toggle = document.getElementById('toggle-password');

  function busy(active, checking = false) {
    form.setAttribute('aria-busy', String(active));
    form.querySelectorAll('input, button').forEach((input) => { input.disabled = active; });
    submit.querySelector('span').textContent = active ? (checking ? 'Checking your session…' : 'Signing you in…') : 'Sign in to your studio';
    status.hidden = !active;
    status.textContent = active ? (checking ? 'Checking your saved session.' : 'Connecting to your account.') : '';
  }

  function showError(message) {
    error.textContent = message;
    error.hidden = !message;
    if (message) error.focus({ preventScroll: true });
  }

  function hidePassword() {
    password.type = 'password';
    toggle.setAttribute('aria-label', 'Show password');
    toggle.setAttribute('aria-pressed', 'false');
    toggle.innerHTML = icon('eye');
  }

  onSessionChange((session, message) => {
    const signedIn = Boolean(session);
    document.body.dataset.auth = signedIn ? 'signed-in' : 'signed-out';
    panel.hidden = signedIn;
    document.getElementById('training-placeholder').hidden = !signedIn;
    menu.hidden = !signedIn;
    menu.open = false;
    document.getElementById('show-login').hidden = signedIn;
    form.reset();
    hidePassword();
    showError(message);
    if (signedIn) {
      const words = session.full_name.trim().split(/\s+/);
      const initials = (words[0][0] + (words.length > 1 ? words.at(-1)[0] : '')).toUpperCase();
      document.getElementById('account-initials').textContent = initials;
      document.getElementById('account-short-name').textContent = words[0];
      document.getElementById('account-full-name').textContent = session.full_name;
      document.querySelector('#training-placeholder .connection-note strong').textContent = 'Your account is connected';
      document.querySelector('#training-placeholder .connection-note p').textContent = 'Your saved workouts and training profile will be connected in the next step.';
      menu.querySelector('summary').focus({ preventScroll: true });
    } else {
      ['account-initials', 'account-short-name', 'account-full-name'].forEach((id) => { document.getElementById(id).textContent = ''; });
      if (!message) name.focus({ preventScroll: true });
    }
  });

  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    if (form.getAttribute('aria-busy') === 'true') return;
    name.value = name.value.trim();
    if (!form.reportValidity()) return;
    const accountName = name.value;
    const secret = password.value;
    showError('');
    busy(true);
    try { await signIn(accountName, secret); }
    catch (failure) { showError(failure.message); }
    finally { busy(false); }
  });
  form.addEventListener('input', () => showError(''));
  toggle.addEventListener('click', () => {
    const show = password.type === 'password';
    password.type = show ? 'text' : 'password';
    toggle.setAttribute('aria-label', show ? 'Hide password' : 'Show password');
    toggle.setAttribute('aria-pressed', String(show));
    toggle.innerHTML = icon(show ? 'eye-off' : 'eye');
  });
  document.getElementById('show-login').addEventListener('click', () => {
    panel.scrollIntoView({ behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth', block: 'center' });
    name.focus({ preventScroll: true });
  });
  document.getElementById('sign-out').addEventListener('click', () => signOut());
  document.addEventListener('click', (event) => { if (!menu.contains(event.target)) menu.open = false; });
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && menu.open) { menu.open = false; menu.querySelector('summary').focus(); }
  });

  document.body.dataset.auth = 'signed-out';
  busy(true, true);
  restoreSession().finally(() => busy(false));
}
