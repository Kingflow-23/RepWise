import { authenticatedRequest, onSessionChange } from './auth.js';
import {
  agentRequestBody, answerParagraphs, decisionErrorMessage, decisionPath, expiryText,
  readAgentResponse, readDecision, readPendingActions,
} from './agent-data.js';
import { browserPosition, geolocationErrorMessage, sharedLocation } from './location-data.js';
import './assistant.css';

// Longer than the server's 420 s request budget, so the server always answers first.
const AGENT_TIMEOUT_MS = 480000;

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text != null) node.textContent = String(text);
  if (className) node.className = className;
  return node;
}

export function initializeAssistant(plans) {
  const panel = document.getElementById('assistant-panel');
  const form = document.getElementById('assistant-form');
  const input = document.getElementById('assistant-message');
  const log = document.getElementById('assistant-log');
  const error = document.getElementById('assistant-error');
  const status = document.getElementById('assistant-status');
  const send = document.getElementById('assistant-send');
  const stopButton = document.getElementById('assistant-stop');
  const newButton = document.getElementById('assistant-new');
  const share = document.getElementById('assistant-share-location');
  const queue = document.getElementById('assistant-queue');
  const queueList = document.getElementById('assistant-queue-list');
  const queueStatus = document.getElementById('assistant-queue-status');
  const queueRefresh = document.getElementById('assistant-queue-refresh');
  let account = null;
  let conversationId = null;
  let actions = [];
  let queueRevision = 0;
  let queueTimer;
  let deciding = null;
  const queueResult = document.getElementById('assistant-queue-result');
  let busy = false;
  let revision = 0;
  let pending = null;
  let timer;
  let timeout;

  function showError(message) {
    error.textContent = message; error.hidden = !message;
    if (message) error.focus({ preventScroll: true });
  }
  function setBusy(value) {
    busy = value;
    form.setAttribute('aria-busy', String(value));
    input.disabled = value; send.disabled = value; share.disabled = value;
    send.querySelector('span').textContent = value ? 'Working on it…' : 'Send';
    stopButton.hidden = !value;
    newButton.hidden = value || !log.childElementCount;
    if (!value) { status.hidden = true; status.textContent = ''; }
  }
  function stop() {
    clearInterval(timer); clearTimeout(timeout);
    pending?.abort(); pending = null;
  }

  // --- Actions waiting for approval (stored on the server, so they survive a reload).
  function renderQueue() {
    const now = Date.now();
    actions = actions.filter((action) => action.expiresAt > now);
    queueList.replaceChildren();
    actions.forEach((action) => {
      const item = element('li', null, 'assistant-queue-item');
      item.dataset.actionId = action.id;
      item.append(element('strong', action.label), element('p', action.reason), element('span', expiryText(action.expiresAt, now), 'assistant-queue-expiry'));
      const buttons = element('div', null, 'assistant-queue-buttons');
      const approve = element('button', deciding === action.id ? 'Applying…' : 'Approve', 'assistant-approve');
      const decline = element('button', 'Decline', 'text-button');
      [approve, decline].forEach((button) => { button.type = 'button'; button.disabled = Boolean(deciding); });
      approve.setAttribute('aria-label', `Approve: ${action.label}`);
      decline.setAttribute('aria-label', `Decline: ${action.label}`);
      approve.addEventListener('click', () => decide(action, 'approve'));
      decline.addEventListener('click', () => decide(action, 'decline'));
      buttons.append(approve, decline);
      item.append(buttons);
      queueList.append(item);
    });
    if (!queueStatus.dataset.error) {
      queueStatus.textContent = actions.length
        ? 'Nothing changes until you approve. Each action runs exactly as described.'
        : 'No actions are waiting for your approval.';
    }
    queue.hidden = !account;
  }
  async function loadQueue() {
    if (!account) return;
    const current = ++queueRevision;
    const userId = account.user_id;
    queueRefresh.disabled = true; queue.hidden = false;
    if (!actions.length && !queueStatus.dataset.error) queueStatus.textContent = 'Checking for actions waiting for your approval…';
    try {
      const payload = await authenticatedRequest(`/students/${encodeURIComponent(userId)}/agent/actions`);
      if (current !== queueRevision) return;
      actions = readPendingActions(payload);
      delete queueStatus.dataset.error; queueStatus.textContent = '';
    } catch {
      if (current !== queueRevision) return;
      // Keep what was shown; say it may be out of date.
      queueStatus.dataset.error = 'true';
      queueStatus.textContent = 'Actions waiting for approval couldn’t load. Use Refresh to try again.';
    } finally {
      if (current === queueRevision) { queueRefresh.disabled = false; renderQueue(); }
    }
  }
  // Periodic update: change the expiry text in place so focus stays on the
  // button a student is about to press; rebuild only when an action expired.
  function tickQueue() {
    const now = Date.now();
    if (actions.some((action) => action.expiresAt <= now)) { renderQueue(); return; }
    queueList.querySelectorAll('.assistant-queue-item').forEach((item) => {
      const action = actions.find((entry) => entry.id === item.dataset.actionId);
      if (action) item.querySelector('.assistant-queue-expiry').textContent = expiryText(action.expiresAt, now);
    });
  }
  function showResult(message, ok) {
    queueResult.textContent = message; queueResult.hidden = !message;
    queueResult.className = ok ? 'decision-ok' : 'decision-failed';
  }
  async function decide(action, decision) {
    if (!account || deciding) return;
    const userId = account.user_id;
    const session = account;
    deciding = action.id; showResult('', true); renderQueue();
    let refreshPlans = decision === 'approve';
    try {
      const payload = await authenticatedRequest(decisionPath(userId, action.id, decision), { method: 'POST', timeout: 120000 });
      if (account !== session) return;
      const outcome = readDecision(payload, decision, action.label);
      showResult(outcome.message, outcome.ok);
    } catch (failure) {
      if (account !== session) return;
      // A dropped connection may hide an approval that did run: refresh either way.
      refreshPlans = refreshPlans || !failure.status;
      showResult(decisionErrorMessage(failure.status), false);
    } finally {
      if (account === session) {
        if (refreshPlans) plans.refresh();
        // Keep the buttons disabled until the refreshed list replaces the old one.
        await loadQueue();
        deciding = null; renderQueue();
      }
    }
  }
  function clearQueue() {
    deciding = null; showResult('', true);
    queueRevision += 1; clearInterval(queueTimer);
    actions = []; queueList.replaceChildren();
    queueStatus.textContent = ''; delete queueStatus.dataset.error; queue.hidden = true;
  }
  queueRefresh.addEventListener('click', loadQueue);

  function addStudentMessage(text) {
    const item = element('li', null, 'assistant-turn student');
    item.append(element('span', 'You', 'assistant-author'), element('p', text));
    log.append(item);
  }
  function addAnswer(result) {
    const item = element('li', null, 'assistant-turn repwise');
    item.append(element('span', 'RepWise', 'assistant-author'));
    // Plain text only: the answer is never inserted as HTML.
    answerParagraphs(result.answer, result.pending.map((action) => action.id)).forEach((paragraph) => item.append(element('p', paragraph)));
    if (result.steps.length) {
      const steps = element('ul', null, 'assistant-steps');
      steps.setAttribute('aria-label', 'Steps RepWise ran');
      result.steps.forEach((step) => steps.append(element('li', `${step.label} · ${step.status}`, `assistant-step ${step.state}`)));
      item.append(steps);
    }
    if (result.pending.length) {
      const box = element('div', null, 'assistant-pending');
      box.append(element('strong', result.pending.length === 1 ? 'Waiting for your approval' : `${result.pending.length} actions waiting for your approval`));
      box.append(element('p', 'Nothing has changed yet. Approve or decline them under Waiting for your approval.'));
      const list = element('ul');
      result.pending.forEach((action) => list.append(element('li', action.reason)));
      box.append(list);
      item.append(box);
    }
    log.append(item);
  }
  function addNotice(text) {
    const item = element('li', null, 'assistant-turn notice');
    item.append(element('p', text));
    log.append(item);
  }

  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    if (!account || busy) return;
    let body;
    try { body = agentRequestBody(input.value, conversationId); }
    catch (failure) { showError(failure.message); return; }
    const current = ++revision;
    const userId = account.user_id;
    if (share.checked) {
      // Asked only now, for this message; the position is never saved.
      busy = true; send.disabled = true; status.hidden = false;
      status.textContent = 'Finding your position…';
      try { body.location = sharedLocation(await browserPosition()); }
      catch (failure) {
        if (current !== revision) return;
        busy = false; send.disabled = false; status.hidden = true;
        showError(failure instanceof Error ? failure.message : geolocationErrorMessage(failure));
        return;
      }
      if (current !== revision) return;
    }
    pending = new AbortController();
    const controller = pending;
    timeout = setTimeout(() => controller.abort(new DOMException('Assistant timed out', 'TimeoutError')), AGENT_TIMEOUT_MS);
    showError('');
    addStudentMessage(body.message);
    setBusy(true); status.hidden = false;
    const started = Date.now();
    const progress = () => {
      const seconds = Math.floor((Date.now() - started) / 1000);
      status.textContent = `RepWise is working on your request. This can take a minute or two. ${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')} elapsed.`;
    };
    progress(); timer = setInterval(progress, 1000);
    try {
      const payload = await authenticatedRequest(`/students/${encodeURIComponent(userId)}/agent/requests`, { method: 'POST', body, signal: controller.signal });
      if (current !== revision) return;
      const result = readAgentResponse(payload);
      conversationId = result.conversationId || conversationId;
      addAnswer(result);
      input.value = '';
      share.checked = false; // sharing is per message, never sticky
      if (result.changedPlans) plans.refresh();
      loadQueue();
    } catch (failure) {
      if (current !== revision) return;
      if (failure.name === 'AbortError') {
        addNotice('You stopped waiting. RepWise may still finish this request on the server. Refresh your programs and the actions waiting for approval before asking again.');
      } else {
        // The typed message stays in the box so it can be sent again.
        showError(failure.message);
      }
    } finally {
      if (current === revision) { stop(); setBusy(false); }
    }
  });

  stopButton.addEventListener('click', () => { pending?.abort(); });
  newButton.addEventListener('click', () => {
    if (busy) return;
    conversationId = null; log.replaceChildren(); showError('');
    newButton.hidden = true; input.focus();
  });
  input.addEventListener('input', () => showError(''));

  onSessionChange((session) => {
    // Never carry a conversation or a pending request across accounts.
    revision += 1; stop(); account = session;
    conversationId = null; log.replaceChildren(); form.reset();
    showError(''); setBusy(false);
    clearQueue();
    panel.hidden = !session;
    if (session) {
      loadQueue();
      // Drop expired actions and update the "expires in" text.
      queueTimer = setInterval(tickQueue, 30000);
    }
  });
}
