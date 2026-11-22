import { authenticatedRequest, onSessionChange } from './auth.js';
import { agentRequestBody, answerParagraphs, readAgentResponse } from './agent-data.js';
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
  let account = null;
  let conversationId = null;
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
    input.disabled = value; send.disabled = value;
    send.querySelector('span').textContent = value ? 'Working on it…' : 'Send';
    stopButton.hidden = !value;
    newButton.hidden = value || !log.childElementCount;
    if (!value) { status.hidden = true; status.textContent = ''; }
  }
  function stop() {
    clearInterval(timer); clearTimeout(timeout);
    pending?.abort(); pending = null;
  }

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
      box.append(element('p', 'Nothing has changed yet. Approving actions will be available here soon.'));
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
      if (result.changedPlans) plans.refresh();
    } catch (failure) {
      if (current !== revision) return;
      if (failure.name === 'AbortError') {
        addNotice('You stopped waiting. RepWise may still finish this request on the server; refresh your programs before asking again.');
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
    panel.hidden = !session;
  });
}
