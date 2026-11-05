import './styles.css';
import { icon, mountIcons } from './icons.js';

mountIcons();
const lastDialogTrigger = new Map();
document.querySelectorAll('[data-open]').forEach((button) => {
  button.addEventListener('click', () => {
    const dialog = document.getElementById(button.dataset.open);
    lastDialogTrigger.set(dialog, button);
    dialog.showModal();
  });
});
document.querySelectorAll('dialog').forEach((dialog) => {
  dialog.querySelectorAll('[data-close]').forEach((button) => button.addEventListener('click', () => dialog.close()));
  dialog.addEventListener('click', (event) => {
    if (event.target !== dialog) return;
    const bounds = dialog.getBoundingClientRect();
    if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) dialog.close();
  });
  dialog.addEventListener('close', () => lastDialogTrigger.get(dialog)?.focus());
});

const scene = document.getElementById('scene');
const message = document.getElementById('scene-message');
const status = document.getElementById('render-status');
function unavailable(text) {
  scene.classList.add('failed');
  scene.setAttribute('aria-busy', 'false');
  message.hidden = false;
  message.querySelector('span:last-child').textContent = text;
  status.textContent = '3D view unavailable';
  document.getElementById('scene-hint').hidden = true;
  document.querySelectorAll('.scene-toolbar button, #explore-studio').forEach((button) => { button.disabled = true; });
}

async function initializeStudio() {
  try {
    const { createStudio } = await import('./studio.js');
    const studio = await createStudio(scene);
    scene.setAttribute('aria-busy', 'false');
    message.hidden = true;
    status.textContent = 'Interactive studio ready';
    document.querySelectorAll('.scene-toolbar button, #explore-studio').forEach((button) => { button.disabled = false; });
    document.querySelectorAll('[data-view]').forEach((button) => {
      button.addEventListener('click', () => studio.setView(button.dataset.view));
    });
    document.getElementById('reset-view').addEventListener('click', () => studio.reset());
    const rotation = document.getElementById('toggle-motion');
    function reflectMotion(active) {
      rotation.setAttribute('aria-pressed', String(active));
      rotation.querySelector('[data-icon]').innerHTML = icon(active ? 'pause' : 'play');
      document.getElementById('motion-label').textContent = active ? 'Pause rotation' : 'Auto rotate';
    }
    studio.onMotionChange = reflectMotion;
    reflectMotion(studio.rotating);
    studio.onViewChange = (view) => document.querySelectorAll('[data-view]').forEach((button) => button.setAttribute('aria-pressed', String(button.dataset.view === view)));
    rotation.addEventListener('click', () => studio.setMotion(!studio.rotating));
    document.getElementById('explore-studio').addEventListener('click', () => {
      studio.reset();
      studio.canvas.focus({ preventScroll: true });
      scene.scrollIntoView({ behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth', block: 'center' });
    });
    document.querySelectorAll('[data-lighting]').forEach((button) => button.addEventListener('click', () => {
      studio.setLighting(button.dataset.lighting);
      document.querySelectorAll('[data-lighting]').forEach((item) => item.setAttribute('aria-pressed', String(item === button)));
    }));
    studio.onContextLost = () => unavailable('The 3D view was interrupted. Reload the page to restore the studio.');
    window.addEventListener('pagehide', () => studio.dispose(), { once: true });
  } catch (error) {
    console.warn('The RepWise studio could not initialize:', error.message);
    unavailable('This device couldn’t open the 3D view. You can still explore the layout and studio settings.');
  }
}

initializeStudio();
