import { authenticatedRequest, onSessionChange } from './auth.js';
import { latestPlan, savedDate, sourceLink } from './plan-data.js';
import './plan.css';

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text != null) node.textContent = String(text);
  if (className) node.className = className;
  return node;
}

export function initializePlan() {
  const card = document.getElementById('latest-plan');
  const viewer = document.getElementById('plan-viewer');
  const title = document.getElementById('latest-plan-title');
  const description = document.getElementById('latest-plan-description');
  const meta = document.getElementById('latest-plan-meta');
  const view = document.getElementById('view-latest-plan');
  const refresh = document.getElementById('reload-plan');
  const navigation = document.getElementById('plan-workouts');
  const detail = document.getElementById('workout-detail');
  let account = null;
  let revision = 0;

  function clear() {
    viewer.hidden = true; view.hidden = true; meta.hidden = true;
    meta.replaceChildren(); navigation.replaceChildren(); detail.replaceChildren();
    ['plan-title', 'plan-summary', 'plan-saved-at', 'plan-safety-note'].forEach((id) => { document.getElementById(id).textContent = ''; });
    document.querySelector('#plan-assumptions ul').replaceChildren();
    document.getElementById('plan-source-list').replaceChildren();
  }

  function sourcesPanel(record) {
    const list = document.getElementById('plan-source-list');
    (record.sources || []).forEach((source) => {
      const item = element('section', null, 'source-item');
      const url = sourceLink(source.start_url) || sourceLink(source.url);
      const heading = element('h4', `${source.source_id} · ${source.title}`);
      if (url) {
        const link = element('a', 'Open source ↗'); link.href = url;
        link.target = '_blank'; link.rel = 'noopener noreferrer';
        item.append(heading, link);
      } else item.append(heading);
      item.append(element('p', source.evidence_text || source.excerpt || 'No excerpt was saved.'));
      list.append(item);
    });
    if (!list.childElementCount) list.append(element('p', 'No sources were saved with this plan.'));
  }

  function workout(record, week, entry, index, button) {
    navigation.querySelectorAll('button').forEach((item) => {
      if (item === button) item.setAttribute('aria-current', 'true');
      else item.removeAttribute('aria-current');
    });
    detail.replaceChildren();
    detail.append(element('p', `WEEK ${week.number} · ${entry.day}`, 'eyebrow'));
    const heading = element('h3', entry.focus); heading.id = 'workout-title';
    detail.append(heading, element('p', `${entry.estimated_minutes} min · ${entry.exercises.length} ${entry.exercises.length === 1 ? 'exercise' : 'exercises'}`, 'workout-duration'));
    const feedback = (record.workout_feedback || []).find((item) => item.week_number === week.number && item.workout_index === index);
    if (feedback) {
      const log = element('div', null, 'workout-log');
      log.append(element('strong', `Recorded: ${feedback.completion_status.replaceAll('_', ' ')}`), element('p', `Difficulty: ${feedback.difficulty.replaceAll('_', ' ')}`));
      if (feedback.comment) log.append(element('p', feedback.comment));
      detail.append(log);
    }
    const exercises = element('ol', null, 'exercise-list');
    entry.exercises.forEach((exercise) => {
      const item = element('li');
      item.append(element('h4', exercise.name));
      const prescription = element('dl', null, 'exercise-prescription');
      [['Sets', exercise.sets], ['Reps', exercise.repetitions], ['Rest', `${exercise.rest_seconds}s`], ['Estimate', `${exercise.estimated_minutes} min`]].forEach(([label, value]) => prescription.append(element('dt', label), element('dd', value)));
      item.append(prescription, element('p', exercise.coaching_note, 'coaching-note'));
      if (exercise.equipment?.length) item.append(element('p', `Equipment: ${exercise.equipment.join(', ')}`, 'exercise-equipment'));
      if (exercise.source_ids?.length || exercise.supporting_evidence?.length) {
        const evidence = element('details', null, 'exercise-evidence');
        evidence.append(element('summary', `Supporting evidence${exercise.source_ids?.length ? ` · ${exercise.source_ids.join(', ')}` : ''}`));
        (exercise.source_ids || []).forEach((id) => {
          const source = (record.sources || []).find((entry) => entry.source_id === id);
          const url = source && (sourceLink(source.start_url) || sourceLink(source.url));
          if (url) {
            const link = element('a', `${id} · ${source.title}`); link.href = url; link.target = '_blank'; link.rel = 'noopener noreferrer'; evidence.append(link);
          } else evidence.append(element('p', `${id}: no source link saved.`));
        });
        (exercise.supporting_evidence || []).forEach((quote) => evidence.append(element('blockquote', `${quote.source_id}: ${quote.quote}`)));
        item.append(evidence);
      }
      exercises.append(item);
    });
    detail.append(exercises);
  }

  function render(record) {
    const { plan } = record;
    title.textContent = plan.title;
    description.textContent = plan.summary;
    const count = plan.weeks.reduce((total, week) => total + week.workouts.length, 0);
    meta.append(element('span', `${plan.weeks.length} ${plan.weeks.length === 1 ? 'week' : 'weeks'}`), element('span', `${count} ${count === 1 ? 'workout' : 'workouts'}`));
    meta.hidden = false; view.hidden = false; viewer.hidden = false;
    document.getElementById('plan-title').textContent = plan.title;
    document.getElementById('plan-summary').textContent = plan.summary;
    const date = savedDate(record.created_at);
    document.getElementById('plan-saved-at').textContent = date ? `Saved ${date}` : '';
    document.getElementById('plan-safety-note').textContent = plan.safety_note;
    const assumptions = document.getElementById('plan-assumptions');
    assumptions.hidden = !plan.assumptions?.length;
    (plan.assumptions || []).forEach((text) => assumptions.querySelector('ul').append(element('li', text)));
    sourcesPanel(record);
    plan.weeks.forEach((week) => {
      const group = element('div', null, 'workout-week');
      group.append(element('h3', `Week ${week.number}`), element('p', week.objective, 'week-objective'));
      week.workouts.forEach((entry, index) => {
        const button = element('button'); button.type = 'button';
        button.append(element('span', entry.day, 'workout-day'), element('strong', entry.focus), element('span', `${entry.estimated_minutes} min`));
        button.addEventListener('click', () => workout(record, week, entry, index, button));
        group.append(button);
      });
      navigation.append(group);
    });
    navigation.querySelector('button').click();
  }

  async function load() {
    if (!account) return;
    const current = ++revision;
    clear(); card.setAttribute('aria-busy', 'true'); refresh.disabled = true;
    title.textContent = 'Loading your plan'; description.textContent = 'Getting your saved workouts.';
    try {
      const records = await authenticatedRequest(`/students/${encodeURIComponent(account.user_id)}/plans?limit=1`);
      if (current !== revision) return;
      const record = latestPlan(records);
      if (record) render(record);
      else { title.textContent = 'Your next chapter starts here.'; description.textContent = 'You don’t have a saved plan yet. Plans created with RepWise will appear here.'; }
      refresh.textContent = 'Refresh saved plan';
    } catch (failure) {
      if (current !== revision) return;
      clear();
      title.textContent = 'Your plan couldn’t load'; description.textContent = failure.message;
      refresh.textContent = 'Retry loading plan';
    } finally {
      if (current === revision) { card.setAttribute('aria-busy', 'false'); refresh.disabled = false; }
    }
  }
  refresh.addEventListener('click', load);
  view.addEventListener('click', () => {
    viewer.scrollIntoView({ behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth', block: 'start' });
    document.getElementById('plan-title').focus({ preventScroll: true });
  });
  onSessionChange((session) => {
    revision += 1; account = session; clear(); card.hidden = !session;
    title.textContent = ''; description.textContent = '';
    document.querySelectorAll('#plan-viewer details').forEach((item) => { item.open = false; });
    if (session) load();
  });
  return {
    showSavedPlan(record) {
      if (!account) return;
      const validated = latestPlan([record]);
      revision += 1; clear(); render(validated);
      card.setAttribute('aria-busy', 'false'); refresh.disabled = false;
      refresh.textContent = 'Refresh saved plan';
      view.click();
    },
  };
}
