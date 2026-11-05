import { onSessionChange, restoreSession, signIn, signOut, register } from './auth.js';
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
  const confirm = document.getElementById('register-confirm');
  const switchMode = document.getElementById('switch-auth-mode');
  let registering = false;

  function mode(create) {
    registering = create;
    form.action = create ? '/api/v1/auth/register' : '/api/v1/auth/login';
    document.getElementById('login-title').innerHTML = create ? 'Your studio.<br />Starts here.' : 'Welcome<br />back.';
    document.getElementById('auth-description').textContent = create ? 'Create your RepWise account. You’ll be signed in as soon as it’s ready.' : 'Your training starts with you. Sign in to your RepWise account.';
    document.getElementById('confirm-field').hidden = !create;
    confirm.required = create;
    confirm.disabled = !create;
    confirm.value = ''; confirm.setCustomValidity('');
    password.autocomplete = create ? 'new-password' : 'current-password';
    switchMode.textContent = create ? 'Already have an account? Sign in' : 'New here? Create an account';
    hidePassword(); showError(''); busy(false);
  }

  function busy(active, checking = false) {
    form.setAttribute('aria-busy', String(active));
    form.querySelectorAll('input, button').forEach((input) => { input.disabled = active; });
    confirm.disabled = active || !registering;
    submit.querySelector('span').textContent = active ? (checking ? 'Checking your session…' : registering ? 'Creating your account…' : 'Signing you in…') : registering ? 'Create account and sign in' : 'Sign in to your studio';
    status.hidden = !active;
    status.textContent = active ? (checking ? 'Checking your saved session.' : registering ? 'Creating your account.' : 'Connecting to your account.') : '';
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
    mode(false);
    hidePassword();
    showError(message);
    if (signedIn) {
      const words = session.full_name.trim().split(/\s+/);
      const initials = (words[0][0] + (words.length > 1 ? words.at(-1)[0] : '')).toUpperCase();
      document.getElementById('account-initials').textContent = initials;
      document.getElementById('account-short-name').textContent = words[0];
      document.getElementById('account-full-name').textContent = session.full_name;
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
    confirm.setCustomValidity(registering && confirm.value !== password.value ? 'Passwords must match.' : '');
    if (!form.reportValidity()) return;
    const accountName = name.value;
    const secret = password.value;
    showError('');
    busy(true);
    try { await (registering ? register(accountName, secret) : signIn(accountName, secret)); }
    catch (failure) { showError(failure.message); }
    finally { busy(false); }
  });
  form.addEventListener('input', () => { confirm.setCustomValidity(''); showError(''); });
  switchMode.addEventListener('click', () => { if (form.getAttribute('aria-busy') !== 'true') { mode(!registering); name.focus(); } });
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
