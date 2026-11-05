import { authenticatedRequest, onSessionChange } from './auth.js';
import { planOptions, generatedRecord } from './generation-data.js';
import './generation.css';

export function initializeGeneration(plans) {
  const panel = document.getElementById('generation-panel');
  const form = document.getElementById('generation-form');
  const fields = document.getElementById('generation-fields');
  const submit = document.getElementById('generate-plan');
  const focus = document.getElementById('generation-focus');
  const count = document.getElementById('generation-count');
  const constraints = document.getElementById('generation-constraints');
  const refinement = document.getElementById('generation-refinement');
  const error = document.getElementById('generation-error');
  const status = document.getElementById('generation-status');
  const profileStatus = document.getElementById('generation-profile-status');
  const setup = document.getElementById('generation-setup');
  let account = null;
  let profile = null;
  let profileState = 'loading';
  let busy = false;
  let revision = 0;
  let pending;
  let timer;
  let timeout;
  let countEdited = false;

  function showError(message) {
    error.textContent = message; error.hidden = !message;
    if (message) error.focus({ preventScroll: true });
  }
  function updateProfile() {
    const ready = profileState === 'ready' && Boolean(profile?.available_days?.length);
    fields.disabled = busy || !account || !ready;
    submit.disabled = fields.disabled;
    setup.hidden = !account || ready || profileState === 'loading';
    setup.disabled = busy;
    setup.textContent = profileState === 'error' ? 'Retry loading profile' : 'Set up your profile';
    profileStatus.textContent = ready
      ? `Using your saved profile: ${profile.available_days.join(', ')} · ${profile.session_minutes} minutes per session.`
      : profileState === 'error' ? 'Your training profile couldn’t load. Retry before generating a plan.'
        : profileState === 'missing' ? 'Save your training profile first, then tell us what you want to work on.'
          : 'Loading your training profile.';
    if (ready && !busy) {
      const max = Math.min(7, profile.available_days.length);
      count.max = String(max);
      count.value = String(countEdited ? Math.max(1, Math.min(max, Number(count.value) || 1)) : Math.min(3, max));
      document.getElementById('generation-count-hint').textContent = max === 1 ? 'Your profile has one available day.' : `Choose 1–${max}, based on your available days.`;
    }
  }
  function setBusy(value) {
    busy = value; form.setAttribute('aria-busy', String(value));
    submit.querySelector('span').textContent = value ? 'Generating your plan…' : 'Generate my plan';
    updateProfile();
  }
  function stop() {
    clearInterval(timer); clearTimeout(timeout); pending?.abort(); pending = null;
  }
  document.addEventListener('repwise:profile', (event) => {
    profileState = event.detail.status; profile = event.detail.profile;
    updateProfile();
  });
  count.addEventListener('input', () => { countEdited = true; });
  form.addEventListener('input', () => { showError(''); if (!busy) { status.hidden = true; status.textContent = ''; } });
  setup.addEventListener('click', () => document.getElementById('open-profile').click());
  document.getElementById('start-plan').addEventListener('click', () => {
    panel.scrollIntoView({ behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth', block: 'start' });
    if (!fields.disabled) focus.focus({ preventScroll: true });
    else if (!setup.hidden) setup.focus({ preventScroll: true });
  });

  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    if (!account || busy || fields.disabled) return;
    focus.value = focus.value.trim();
    if (!form.reportValidity()) return;
    let body;
    try { body = planOptions({ focus: focus.value, count: count.value, constraints: constraints.value, refinement: refinement.value }, profile); }
    catch (failure) { showError(failure.message); return; }
    const current = ++revision;
    const userId = account.user_id;
    pending = new AbortController();
    const controller = pending;
    timeout = setTimeout(() => controller.abort(new DOMException('Generation timed out', 'TimeoutError')), 900000);
    showError(''); setBusy(true); status.hidden = false;
    const started = Date.now();
    const progress = () => {
      const seconds = Math.floor((Date.now() - started) / 1000);
      status.textContent = `Generating and checking your plan. This can take a few minutes. ${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')} elapsed.`;
    };
    progress(); timer = setInterval(progress, 1000);
    try {
      const result = await authenticatedRequest(`/students/${encodeURIComponent(userId)}/plans`, { method: 'POST', body, signal: controller.signal });
      if (current !== revision) return;
      const record = generatedRecord(result);
      plans.showSavedPlan(record);
      status.textContent = 'Your new plan is saved. Your workouts are ready below.';
    } catch (failure) {
      if (current !== revision) return;
      status.hidden = true; status.textContent = '';
      if (failure.status === 404) { profile = null; profileState = 'missing'; }
      showError(failure.message);
    } finally {
      if (current === revision) { stop(); setBusy(false); }
    }
  });

  onSessionChange((session) => {
    revision += 1; stop(); account = session;
    // Profile changes are published separately by the authenticated profile loader.
    if (!session) { profile = null; profileState = 'signed-out'; }
    form.reset(); countEdited = false; form.querySelector('details').open = false;
    error.textContent = ''; error.hidden = true; status.textContent = ''; status.hidden = true;
    panel.hidden = !session; setBusy(false);
  });
}
