import { authenticatedRequest, onSessionChange } from './auth.js';
import { activePlan, draftPlans, savedDate, sourceLink } from './plan-data.js';
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
  const drafts = document.getElementById('draft-plans');
  const draftList = document.getElementById('draft-plan-list');
  const draftStatus = document.getElementById('draft-plan-status');
  let account = null;
  let currentPlan = null;
  let revision = 0;

  function clearViewer() {
    viewer.hidden = true;
    navigation.replaceChildren(); detail.replaceChildren();
    ['plan-title', 'plan-summary', 'plan-saved-at', 'plan-safety-note'].forEach((id) => { document.getElementById(id).textContent = ''; });
    document.querySelector('#plan-assumptions ul').replaceChildren();
    document.getElementById('plan-source-list').replaceChildren();
    document.getElementById('plan-viewer-status').textContent = '';
  }

  function clear() {
    clearViewer(); view.hidden = true; meta.hidden = true;
    meta.replaceChildren(); currentPlan = null;
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
    detail.append(element('p', `WEEK · ${entry.day}`, 'eyebrow'));
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
      exercises.append(item);
    });
    detail.append(exercises);
  }

  function render(record, preview = false) {
    clearViewer();
    const { plan } = record;
    if (!preview) {
      currentPlan = record;
      title.textContent = plan.title;
      description.textContent = plan.summary;
      const count = plan.weeks.reduce((total, week) => total + week.workouts.length, 0);
      meta.append(element('span', `${plan.weeks.length} ${plan.weeks.length === 1 ? 'week' : 'weeks'}`), element('span', `${count} ${count === 1 ? 'workout' : 'workouts'}`));
      meta.hidden = false; view.hidden = false; viewer.hidden = false;
    }
    viewer.hidden = false;
    viewer.dataset.planStatus = record.status;
    document.getElementById('plan-viewer-status').textContent = preview ? 'DRAFT PREVIEW · NOT ACTIVE' : 'YOUR ACTIVE PROGRAM';
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
      group.append(element('h3', 'Week'), element('p', week.objective, 'week-objective'));
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

  function showViewer() {
    viewer.scrollIntoView({ behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth', block: 'start' });
    document.getElementById('plan-title').focus({ preventScroll: true });
  }

  async function loadDrafts(userId, current) {
    draftList.replaceChildren(); draftStatus.textContent = 'Loading drafts…';
    try {
      const records = draftPlans(await authenticatedRequest(`/students/${encodeURIComponent(userId)}/plans?status=draft&limit=20`));
      if (current !== revision) return;
      draftStatus.textContent = records.length ? 'Drafts do not replace your active program. Activation requires the approval workflow.' : 'No draft plans.';
      records.forEach((record) => {
        const item = element('article', null, 'draft-plan-item');
        item.append(element('span', 'Draft · not active', 'draft-badge'), element('h3', record.plan.title), element('p', record.plan.summary));
        const date = savedDate(record.created_at);
        if (date) item.append(element('p', `Saved ${date}`, 'draft-date'));
        const button = element('button', 'Preview draft', 'plan-create-link'); button.type = 'button';
        button.addEventListener('click', () => { render(record, true); showViewer(); });
        item.append(button); draftList.append(item);
      });
    } catch {
      if (current === revision) draftStatus.textContent = 'Drafts couldn’t load. Use Refresh programs to try again.';
    }
  }

  async function load() {
    if (!account) return;
    const current = ++revision;
    clear(); card.setAttribute('aria-busy', 'true'); refresh.disabled = true;
    const draftRequest = loadDrafts(account.user_id, current);
    title.textContent = 'Loading your active program'; description.textContent = 'Getting your saved workouts.';
    try {
      const payload = await authenticatedRequest(`/students/${encodeURIComponent(account.user_id)}/plans/current`);
      if (current !== revision) return;
      const record = activePlan(payload);
      if (record) render(record);
      else { title.textContent = 'No active program yet'; description.textContent = 'Create a program to get started. Draft plans appear separately below and are not active.'; }
      refresh.textContent = 'Refresh programs';
    } catch (failure) {
      if (current !== revision) return;
      clear();
      title.textContent = 'Your plan couldn’t load'; description.textContent = failure.message;
      refresh.textContent = 'Retry loading plan';
    } finally {
      if (current === revision) { card.setAttribute('aria-busy', 'false'); refresh.disabled = false; }
    }
    await draftRequest;
  }
  refresh.addEventListener('click', load);
  view.addEventListener('click', () => {
    if (currentPlan) { meta.replaceChildren(); render(currentPlan); showViewer(); }
  });
  onSessionChange((session) => {
    revision += 1; account = session; clear(); card.hidden = !session;
    drafts.hidden = !session; draftList.replaceChildren(); draftStatus.textContent = '';
    title.textContent = ''; description.textContent = '';
    document.querySelectorAll('#plan-viewer details').forEach((item) => { item.open = false; });
    if (session) load();
  });
  return {
    showSavedPlan(record) {
      if (!account) return;
      const validated = activePlan(record);
      revision += 1; clear(); render(validated);
      card.setAttribute('aria-busy', 'false'); refresh.disabled = false;
      refresh.textContent = 'Refresh programs';
      loadDrafts(account.user_id, revision);
      view.click();
    },
  };
}
