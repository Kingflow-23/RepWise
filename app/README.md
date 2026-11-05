# RepWise studio frontend

Step one of the web frontend: HTML, CSS and JavaScript, with Three.js for a
real 3D scene and Vite for local development and production builds.

From `app`, using Node.js 20.19 or newer:

```powershell
npm install
npm run dev
```

Open the local address printed by Vite, normally http://127.0.0.1:5173.

```powershell
npm run build
npm run preview
```

The production build is written to `dist/`. The model and JavaScript are
served locally; no external model/CDN/font requests are needed at runtime.
The Three.js renderer is loaded separately so the interface appears before
the model finishes loading.

Implemented controls: mouse orbit and zoom, front/back views, reset, auto
rotation, keyboard rotation/zoom, warm/neutral lighting, and accessible preview
and settings dialogs. On phones, one finger scrolls the page and two fingers
control the model; front/back buttons are also available. Reduced-motion
preferences pause rotation and remove camera transitions. Rendering pauses
when hidden or off screen and stops when the camera is idle. A failed WebGL
context or missing asset leaves the layout and dialogs usable.

This preview makes **no API calls** and displays **no invented training data**.
Login, saved workouts, sources, feedback and the assistant will be connected in
later steps. Muscle targeting, calendar dates and set-by-set persistence are
not simulated. The previous Streamlit app remains in `../genai_app/`.

The model's source and licence are recorded in `public/models/README.md`.
