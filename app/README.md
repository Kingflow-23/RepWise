# RepWise studio frontend

The web frontend uses HTML, CSS and JavaScript, with Three.js for a
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

## Account sign-in

Start the backend from the repository root in a separate terminal:

```powershell
python -m uvicorn src.api.main:app --reload
```

Use the full name and password of an existing RepWise account, or choose
**New here? Create an account**. Registration checks password confirmation and
uses `/api/v1/auth/register`; its returned token signs the new user in
immediately so they can set up their profile.

The browser sends `/api` requests through Vite's same-origin proxy to
`REPWISE_API_URL` in the root `.env`, defaulting to `http://127.0.0.1:8000`.
Restart Vite after changing that setting. The same proxy is configured for
`npm run preview`. A production deployment must route `/api` to FastAPI on
the same origin; Vite's development proxy is not part of the built files.
No backend secret is exposed through `VITE_*` settings or the browser bundle.

The login form uses `/api/v1/auth/login`. Tab-scoped session storage retains
the bearer token and public account identity across reloads, never passwords.
Reloading verifies the token through `/api/v1/auth/me` before showing a saved
identity. Expired or rejected sessions return to login. Sign-out clears local
session state; it does not revoke the backend JWT, which expires normally.
If session storage is unavailable, sign-in works in memory for the current page.

## Training profile

After sign-in, the studio loads the account's profile through the existing
authenticated `GET /api/v1/students/{student_id}/profile` route. A missing
profile opens the way to setup; a connection error offers a retry instead.
Choose **Set up your profile** for a three-step form: experience and goal,
available days and session duration, then equipment and optional preferences.
Saving uses the existing `PUT` route and stores the profile in the backend
database. Reloading restores it; **Edit training profile** or **Training profile**
in the account menu reopens the editor. Closing without saving keeps the draft
while signed in on the current page. Sign-out clears the profile and draft
from the interface. Profiles are not stored in browser storage.

Saved workouts, sources, feedback and the assistant will be connected in later
steps. The preview displays no invented training data. Muscle targeting,
calendar dates and set-by-set persistence are not simulated. The previous
Streamlit app remains in `../genai_app/`.

The model's source and licence are recorded in `public/models/README.md`.
