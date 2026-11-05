import { authenticatedRequest, onSessionChange } from './auth.js';
import './profile.css';

const DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];

export function initializeProfile() {
  const dialog = document.getElementById('profile-dialog');
  const form = document.getElementById('profile-form');
  const open = document.getElementById('open-profile');
  const menu = document.getElementById('menu-profile');
  const error = document.getElementById('profile-error');
  const summary = document.getElementById('profile-summary');
  const notice = document.getElementById('profile-notice');
  const state = document.getElementById('profile-state');
  const description = document.getElementById('profile-description');
  const next = document.getElementById('profile-next');
  const back = document.getElementById('profile-back');
  const save = document.getElementById('profile-save');
  const fields = Object.fromEntries(['experience', 'goal', 'minutes', 'equipment', 'preferences'].map((key) => [key, document.getElementById(`profile-${key}`)]));
  let session = null;
  let generation = 0;
  let saved = null;
  let step = 0;
  let loadingFailed = false;
  let saving = false;
  let returnFocus;

  function publish(status, profile = null) {
    document.dispatchEvent(new CustomEvent('repwise:profile', { detail: { status, profile } }));
  }

  function addDay(day, custom = false) {
    const label = document.createElement('label');
    const input = document.createElement('input');
    input.type = 'checkbox'; input.name = 'available_days'; input.value = day;
    if (custom) label.dataset.custom = 'true';
    const text = document.createElement('span'); text.textContent = custom ? day : day.slice(0, 3);
    input.setAttribute('aria-label', day);
    label.append(input, text);
    document.getElementById('profile-days').append(label);
  }
  DAYS.forEach((day) => addDay(day));

  function showError(message) {
    error.textContent = message; error.hidden = !message;
    if (message) error.focus({ preventScroll: true });
  }

  function showStep(index, focus = true) {
    step = index;
    form.querySelectorAll('.profile-step').forEach((panel, i) => {
      panel.hidden = i !== index;
      // Hidden steps must not block native validation or receive keyboard focus.
      panel.disabled = i !== index || saving;
    });
    dialog.querySelectorAll('.profile-progress li').forEach((item, i) => {
      if (i === index) item.setAttribute('aria-current', 'step');
      else item.removeAttribute('aria-current');
    });
    back.hidden = index === 0; next.hidden = index === 2; save.hidden = index !== 2;
    if (focus) form.querySelector(`.profile-step[data-step="${index}"] input, .profile-step[data-step="${index}"] select, .profile-step[data-step="${index}"] textarea`)?.focus();
  }

  function fill(profile) {
    form.reset();
    form.querySelectorAll('#profile-days [data-custom]').forEach((label) => label.remove());
    (profile?.available_days ?? []).filter((day) => !DAYS.includes(day)).forEach((day) => addDay(day, true));
    fields.experience.querySelectorAll('[data-custom]').forEach((option) => option.remove());
    // Older API clients accept free-form experience; preserve their saved value.
    if (profile && ![...fields.experience.options].some((option) => option.value === profile.experience)) {
      const option = new Option(profile.experience, profile.experience);
      option.dataset.custom = 'true'; fields.experience.add(option);
    }
    fields.experience.value = profile?.experience ?? 'beginner';
    fields.goal.value = profile?.goal ?? '';
    fields.minutes.value = profile?.session_minutes ?? 45;
    fields.equipment.value = profile?.equipment.join(', ') ?? '';
    fields.preferences.value = profile?.preferences ?? '';
    form.querySelectorAll('[name="available_days"]').forEach((input) => { input.checked = profile?.available_days.includes(input.value) ?? false; });
    showStep(0, false); showError('');
  }

  function render() {
    summary.replaceChildren(); summary.hidden = !saved;
    if (saved) {
      const rows = [['Goal', saved.goal], ['Experience', saved.experience], ['Schedule', `${saved.available_days.join(', ')} · ${saved.session_minutes} min`], ['Equipment', saved.equipment.join(', ') || 'No equipment'], ['Preferences', saved.preferences || 'No preferences added']];
      rows.forEach(([label, value]) => {
        const term = document.createElement('dt'); term.textContent = label;
        const detail = document.createElement('dd'); detail.textContent = value;
        summary.append(term, detail);
      });
    }
    state.textContent = saved ? 'Your profile is ready' : 'Let’s start with you';
    description.textContent = saved ? 'These details are saved to your account. Update them whenever your routine changes.' : 'Set your experience, schedule, and equipment before creating a plan.';
    open.textContent = saved ? 'Edit training profile' : 'Set up your profile';
    open.disabled = false; menu.disabled = false;
    publish(saved ? 'ready' : 'missing', saved);
  }

  async function load() {
    const current = ++generation;
    const path = `/students/${encodeURIComponent(session.user_id)}/profile`;
    loadingFailed = false; open.disabled = true; menu.disabled = true;
    open.textContent = 'Loading profile…'; state.textContent = 'Loading your profile';
    description.textContent = 'Getting your saved training details.';
    notice.hidden = true;
    publish('loading');
    try {
      const profile = await authenticatedRequest(path);
      if (current !== generation) return;
      saved = profile;
    } catch (failure) {
      if (current !== generation) return;
      if (failure.status === 404) saved = null;
      else {
        loadingFailed = true; state.textContent = 'Your profile couldn’t load';
        description.textContent = 'Try again to load your saved details before making changes.';
        open.textContent = 'Retry loading profile'; open.disabled = false;
        publish('error');
        return;
      }
    }
    if (current !== generation) return;
    fill(saved); render();
  }

  function openEditor(event) {
    if (!session || saving) return;
    if (loadingFailed) { load(); return; }
    if (open.disabled) return;
    returnFocus = event.currentTarget === menu ? document.querySelector('#account-menu summary') : open;
    document.getElementById('account-menu').open = false;
    showError(''); notice.hidden = true;
    dialog.showModal(); showStep(step);
  }
  open.addEventListener('click', openEditor);
  menu.addEventListener('click', openEditor);
  dialog.addEventListener('close', () => { if (session) returnFocus?.focus({ preventScroll: true }); });

  function valid(index) {
    fields.goal.value = fields.goal.value.trim();
    if (index === 1 && !form.querySelector('[name="available_days"]:checked')) {
      showError('Choose at least one day you can train.'); return false;
    }
    for (const input of form.querySelectorAll(`.profile-step[data-step="${index}"] input, .profile-step[data-step="${index}"] select, .profile-step[data-step="${index}"] textarea`)) {
      if (!input.reportValidity()) return false;
    }
    showError(''); return true;
  }
  next.addEventListener('click', () => { if (!saving && valid(step)) showStep(Math.min(2, step + 1)); });
  back.addEventListener('click', () => { if (!saving) { showError(''); showStep(Math.max(0, step - 1)); } });
  form.addEventListener('input', () => showError(''));

  function setSaving(active) {
    saving = active; form.setAttribute('aria-busy', String(active));
    form.querySelectorAll('button').forEach((button) => { button.disabled = active; });
    document.getElementById('profile-saving').hidden = !active;
    showStep(step, false);
  }

  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    if (!session || saving) return;
    if (step < 2) { if (valid(step)) showStep(step + 1); return; }
    if (!valid(step)) return;
    const current = generation;
    const payload = {
      experience: fields.experience.value, goal: fields.goal.value.trim(),
      available_days: [...form.querySelectorAll('[name="available_days"]:checked')].map((input) => input.value),
      session_minutes: Number(fields.minutes.value),
      equipment: fields.equipment.value.split(',').map((item) => item.trim()).filter(Boolean),
      preferences: fields.preferences.value.trim() || null,
    };
    setSaving(true);
    try {
      const profile = await authenticatedRequest(`/students/${encodeURIComponent(session.user_id)}/profile`, { method: 'PUT', body: payload });
      if (current !== generation) return;
      saved = profile; fill(saved); render(); dialog.close();
      notice.textContent = 'Profile saved.'; notice.hidden = false;
    } catch (failure) {
      if (current === generation) showError(failure.message);
    } finally { if (current === generation) setSaving(false); }
  });

  onSessionChange((account) => {
    generation += 1; session = account; saved = null; loadingFailed = false;
    dialog.close(); setSaving(false); fill(null);
    summary.replaceChildren(); summary.hidden = true; notice.textContent = ''; notice.hidden = true;
    menu.disabled = true; open.disabled = true;
    publish(account ? 'loading' : 'signed-out');
    if (account) load();
  });
}
