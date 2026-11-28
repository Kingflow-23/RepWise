/** Pure helpers for locations: shared once with a message, or saved by choice. */

const AREA = 'metropolitan France';

/** Round to about 100 m: enough for nearby gyms, less precise than GPS. */
function coarse(value) {
  return Math.round(value * 1000) / 1000;
}

function validCoordinates(latitude, longitude) {
  return Number.isFinite(latitude) && Number.isFinite(longitude)
    && latitude >= -90 && latitude <= 90 && longitude >= -180 && longitude <= 180;
}

/**
 * Build the `location` field of one assistant request from browser coordinates.
 * It is sent with that message only and never stored by RepWise.
 */
export function sharedLocation(coords) {
  const latitude = Number(coords?.latitude);
  const longitude = Number(coords?.longitude);
  if (!validCoordinates(latitude, longitude)) throw new Error('Your browser returned an unusable position.');
  return { label: 'Current location', latitude: coarse(latitude), longitude: coarse(longitude) };
}

/** Ask the browser for its position once (the browser shows its own permission prompt). */
export function browserPosition(geolocation = globalThis.navigator?.geolocation) {
  return new Promise((resolve, reject) => {
    if (!geolocation) { reject({ code: 2 }); return; }
    geolocation.getCurrentPosition((position) => resolve(position.coords), reject, { enableHighAccuracy: false, timeout: 15000, maximumAge: 300000 });
  });
}

/** Explain a failed browser position request (GeolocationPositionError codes). */
export function geolocationErrorMessage(error, alternative = 'untick “Share my location”') {
  const or = alternative ? `, or ${alternative}` : '';
  if (error?.code === 1) return `Location access was refused in your browser. Allow it${or}.`;
  if (error?.code === 3) return `Your position took too long to find. Try again${or}.`;
  return `Your position couldn’t be found. Try again${or}.`;
}

/**
 * Body of PUT /location: a place name to look up, or a name with coordinates.
 * Throws a readable error when the name is too short or too long.
 */
export function savedLocationBody(label, coords = null) {
  const text = typeof label === 'string' ? label.trim().replace(/\s+/g, ' ') : '';
  if (text.length < 2 || text.length > 120) throw new Error('Use a place name of 2–120 characters.');
  if (!coords) return { label: text };
  const { latitude, longitude } = sharedLocation(coords);
  return { label: text, latitude, longitude };
}

/** Keep the saved location from GET or PUT /location, or null when unusable. */
export function readSavedLocation(payload) {
  if (!payload || typeof payload.label !== 'string' || !validCoordinates(Number(payload.latitude), Number(payload.longitude))) return null;
  return { label: payload.label, consentedAt: Date.parse(payload.consented_at) || null };
}

/** Explain why saving a location was refused, from the HTTP status. */
export function locationErrorMessage(status) {
  if (status === 404) return `That place wasn’t found. Try a town, a postcode with the town, or an address in ${AREA}.`;
  if (status === 422) return `Use a town, a postcode with the town, or an address in ${AREA}. Countries, regions and places abroad can’t be used.`;
  if (status === 503) return 'The map service is unavailable right now. Please try again later.';
  if (status === 401) return 'Your session has expired. Please sign in again.';
  return 'Your location couldn’t be saved. Please try again in a moment.';
}
