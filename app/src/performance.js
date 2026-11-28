import { authenticatedRequest, onSessionChange } from './auth.js';
import { performanceBody, performanceErrorMessage, performancePath, readHistory, readPerformance } from './performance-data.js';

const SAVED_EVENT = 'repwise:performance-saved';

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text != null) node.textContent = String(text);
  if (className) node.className = className;
  return node;
}

let fieldCount = 0;
function field(labelText, attributes) {
  const id = `performance-field-${++fieldCount}`;
  const wrapper = element('div', null, 'performance-field');
  const label = element('label', labelText); label.htmlFor = id;
  const input = document.createElement('input');
  Object.assign(input, attributes); input.id = id;
  wrapper.append(label, input);
  return { wrapper, input };
}

/**
 * The "Log what you did" form for one workout of the active program.
 * It prefills from the server and saves through POST .../performance.
 */
export function performancePanel({ userId, planId, weekNumber, workoutIndex, exercises }) {
  const panel = element('section', null, 'performance-panel');
  const heading = element('h4', 'Log what you did');
  const intro = element('p', 'Optional. Enter the sets, reps and weight you actually did. Leave an exercise empty to skip it.', 'performance-intro');
  const form = document.createElement('form');
  const rows = exercises.map((exercise) => {
    const row = element('fieldset', null, 'performance-row');
    row.append(element('legend', exercise.name));
    const sets = field('Sets', { type: 'number', min: '0', max: '50', step: '1', inputMode: 'numeric' });
    const reps = field('Reps per set', { type: 'text', maxLength: 100, placeholder: exercise.repetitions || '10, 10, 8' });
    const weight = field('Weight (kg)', { type: 'text', inputMode: 'decimal', placeholder: 'Optional' });
    row.append(sets.wrapper, reps.wrapper, weight.wrapper);
    form.append(row);
    return { name: exercise.name, sets: sets.input, reps: reps.input, weight: weight.input };
  });
  const error = element('p', null, 'auth-error'); error.hidden = true; error.setAttribute('role', 'alert'); error.tabIndex = -1;
  const status = element('p', null, 'performance-status'); status.setAttribute('role', 'status');
  const save = element('button', 'Save my performance', 'assistant-approve'); save.type = 'submit';
  form.append(error, status, save);
  panel.append(heading, intro, form);

  const path = performancePath(userId, planId, weekNumber, workoutIndex);
  let busy = false;
  const setBusy = (value) => { busy = value; save.disabled = value; rows.forEach((row) => { row.sets.disabled = value; row.reps.disabled = value; row.weight.disabled = value; }); };

  // Prefill what was already saved for this workout.
  setBusy(true); status.textContent = 'Loading what you logged…';
  authenticatedRequest(path).then((payload) => {
    const saved = readPerformance(payload);
    rows.forEach((row) => {
      const record = saved.get(row.name.toLowerCase());
      if (!record) return;
      row.sets.value = String(record.sets_completed);
      row.reps.value = record.reps_per_set;
      row.weight.value = record.weight_kg == null ? '' : String(record.weight_kg);
    });
    status.textContent = saved.size ? 'Saved values are shown. Saving again replaces them.' : '';
  }).catch(() => { status.textContent = 'What you logged before couldn’t load. You can still save new values.'; })
    .finally(() => setBusy(false));

  form.addEventListener('input', () => { error.hidden = true; error.textContent = ''; });
  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    if (busy) return;
    let body;
    try {
      body = performanceBody(rows.map((row) => ({ name: row.name, sets: row.sets.value, reps: row.reps.value, weight: row.weight.value })));
    } catch (failure) { error.textContent = failure.message; error.hidden = false; error.focus({ preventScroll: true }); return; }
    setBusy(true); status.textContent = 'Saving…';
    try {
      const result = await authenticatedRequest(path, { method: 'POST', body });
      const count = Array.isArray(result?.records) ? result.records.length : body.exercises.length;
      status.textContent = `Saved ${count} ${count === 1 ? 'exercise' : 'exercises'}. Your training log is updated.`;
      document.dispatchEvent(new CustomEvent(SAVED_EVENT));
    } catch (failure) {
      status.textContent = '';
      error.textContent = performanceErrorMessage(failure.status, failure.detail);
      error.hidden = false; error.focus({ preventScroll: true });
    } finally { setBusy(false); }
  });
  return panel;
}

/** The "Your training log" section: workouts with feedback or logged performance. */
export function initializeTrainingLog() {
  const section = document.getElementById('training-log');
  const list = document.getElementById('training-log-list');
  const status = document.getElementById('training-log-status');
  let account = null;
  let revision = 0;

  async function load() {
    if (!account) return;
    const current = ++revision;
    status.textContent = 'Loading your training log…';
    try {
      const history = readHistory(await authenticatedRequest(`/students/${encodeURIComponent(account.user_id)}/history`));
      if (current !== revision) return;
      list.replaceChildren();
      history.workouts.forEach((entry) => {
        const item = element('li', null, 'training-log-item');
        item.append(element('strong', entry.title), element('span', entry.status, 'training-log-status'));
        if (entry.lines.length) {
          const lines = element('ul');
          entry.lines.forEach((line) => lines.append(element('li', line)));
          item.append(lines);
        }
        list.append(item);
      });
      status.textContent = history.workouts.length ? '' : 'Nothing logged yet. Open a workout in your active program to log what you did.';
    } catch (failure) {
      if (current !== revision) return;
      // 404 means no profile yet: nothing to show.
      status.textContent = failure.status === 404 ? 'Save your training profile to start a training log.' : 'Your training log couldn’t load. Refresh your programs to try again.';
    }
  }

  document.addEventListener(SAVED_EVENT, load);
  document.addEventListener('repwise:plans-loaded', load);
  onSessionChange((session) => {
    revision += 1; account = session;
    list.replaceChildren(); status.textContent = '';
    section.hidden = !session;
    if (session) load();
  });
}
