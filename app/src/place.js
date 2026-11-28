import { authenticatedRequest, onSessionChange } from './auth.js';
import { browserPosition, geolocationErrorMessage, locationErrorMessage, readSavedLocation, savedLocationBody } from './location-data.js';

/** The optional saved location: read, save (explicit opt-in) and forget. */
export function initializePlace() {
  const details = document.getElementById('assistant-place');
  const summary = document.getElementById('assistant-place-summary');
  const current = document.getElementById('assistant-place-current');
  const form = document.getElementById('assistant-place-form');
  const label = document.getElementById('assistant-place-label');
  const save = document.getElementById('assistant-place-save');
  const here = document.getElementById('assistant-place-here');
  const forget = document.getElementById('assistant-place-forget');
  const error = document.getElementById('assistant-place-error');
  const status = document.getElementById('assistant-place-status');
  let account = null;
  let revision = 0;
  let busy = false;

  const path = () => `/students/${encodeURIComponent(account.user_id)}/location`;

  function showError(message) {
    error.textContent = message; error.hidden = !message;
    if (message) error.focus({ preventScroll: true });
  }
  function setBusy(value) {
    busy = value;
    [save, here, forget, label].forEach((control) => { control.disabled = value; });
  }
  function show(saved) {
    summary.textContent = saved ? saved.label : 'Not saved';
    current.hidden = !saved;
    current.textContent = saved ? `Saved: ${saved.label}. RepWise uses it when you don’t share your position or name a place.` : '';
    forget.hidden = !saved;
    save.textContent = saved ? 'Replace with this place' : 'Save this place';
  }

  async function load() {
    const mine = ++revision;
    try {
      const payload = await authenticatedRequest(path());
      if (mine === revision) show(readSavedLocation(payload));
    } catch (failure) {
      if (mine !== revision) return;
      show(null);
      // 404 simply means nothing is saved.
      if (failure.status !== 404) status.textContent = 'Your saved location couldn’t load.';
    }
  }

  async function store(body, done) {
    const mine = ++revision;
    setBusy(true); showError(''); status.textContent = 'Saving…';
    try {
      const saved = readSavedLocation(await authenticatedRequest(path(), { method: 'PUT', body, timeout: 30000 }));
      if (mine !== revision) return;
      show(saved); label.value = '';
      status.textContent = done;
    } catch (failure) {
      if (mine !== revision) return;
      status.textContent = '';
      showError(locationErrorMessage(failure.status));
    } finally {
      if (mine === revision) setBusy(false);
    }
  }

  form.addEventListener('submit', (event) => {
    event.preventDefault();
    if (!account || busy) return;
    let body;
    try { body = savedLocationBody(label.value); } catch (failure) { showError(failure.message); return; }
    store(body, 'Location saved. You can forget it at any time.');
  });

  here.addEventListener('click', async () => {
    if (!account || busy) return;
    const mine = revision;
    setBusy(true); showError(''); status.textContent = 'Finding your position…';
    let body;
    try { body = savedLocationBody(label.value.trim() || 'My saved position', await browserPosition()); }
    catch (failure) {
      if (mine !== revision) return;
      setBusy(false); status.textContent = '';
      showError(failure instanceof Error ? failure.message : geolocationErrorMessage(failure, 'type a place name instead'));
      return;
    }
    if (mine !== revision) return;
    store(body, 'Your current position was saved (rounded to about 100 m). You can forget it at any time.');
  });

  forget.addEventListener('click', async () => {
    if (!account || busy) return;
    const mine = ++revision;
    setBusy(true); showError(''); status.textContent = 'Forgetting…';
    try {
      await authenticatedRequest(path(), { method: 'DELETE' });
      if (mine !== revision) return;
      show(null); status.textContent = 'Your saved location was deleted.';
    } catch (failure) {
      if (mine !== revision) return;
      status.textContent = '';
      if (failure.status === 404) { show(null); status.textContent = 'No location was saved.'; }
      else showError('Your location couldn’t be deleted. Please try again in a moment.');
    } finally {
      if (mine === revision) setBusy(false);
    }
  });
  label.addEventListener('input', () => showError(''));

  onSessionChange((session) => {
    revision += 1; account = session;
    form.reset(); showError(''); status.textContent = ''; setBusy(false);
    show(null); details.open = false;
    if (session) load();
  });
}
