import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  browserPosition, geolocationErrorMessage, locationErrorMessage, readSavedLocation, savedLocationBody, sharedLocation,
} from '../src/location-data.js';
import { request } from '../src/api.js';

test('a shared position is rounded to about 100 m and labelled', () => {
  assert.deepEqual(sharedLocation({ latitude: 45.764043, longitude: 4.835659 }), { label: 'Current location', latitude: 45.764, longitude: 4.836 });
  for (const coords of [null, {}, { latitude: 91, longitude: 0 }, { latitude: 'x', longitude: 2 }]) assert.throws(() => sharedLocation(coords), /unusable position/);
});

test('the browser position is asked once, without high accuracy', async () => {
  let options;
  const geolocation = { getCurrentPosition: (ok, _fail, opts) => { options = opts; ok({ coords: { latitude: 48.85, longitude: 2.35 } }); } };
  assert.deepEqual(await browserPosition(geolocation), { latitude: 48.85, longitude: 2.35 });
  assert.equal(options.enableHighAccuracy, false);
  await assert.rejects(browserPosition(undefined), (error) => error.code === 2);
});

test('position errors explain what to do', () => {
  assert.match(geolocationErrorMessage({ code: 1 }), /refused.*untick “Share my location”/);
  assert.match(geolocationErrorMessage({ code: 3 }, 'type a place name instead'), /too long.*type a place name instead\.$/);
  assert.equal(geolocationErrorMessage({ code: 2 }, ''), 'Your position couldn’t be found. Try again.');
});

test('a saved location body is a cleaned name, with rounded coordinates when given', () => {
  assert.deepEqual(savedLocationBody('  Lyon   69003 '), { label: 'Lyon 69003' });
  assert.deepEqual(savedLocationBody('Home', { latitude: 48.85661, longitude: 2.35222 }), { label: 'Home', latitude: 48.857, longitude: 2.352 });
  assert.throws(() => savedLocationBody('L'), /2–120/);
  assert.throws(() => savedLocationBody('x'.repeat(121)), /2–120/);
});

test('saved locations are read safely and refusals explained', () => {
  assert.deepEqual(readSavedLocation({ label: 'Lyon', latitude: 45.76, longitude: 4.83, consented_at: '2026-11-23T10:00:00Z' }), { label: 'Lyon', consentedAt: Date.parse('2026-11-23T10:00:00Z') });
  assert.equal(readSavedLocation({ label: 'Lyon' }), null);
  assert.equal(readSavedLocation(null), null);
  assert.match(locationErrorMessage(404), /wasn’t found/);
  assert.match(locationErrorMessage(422), /places abroad/);
  assert.match(locationErrorMessage(503), /map service/);
});

test('a 204 response (forgetting the location) is a success', async (t) => {
  t.mock.method(globalThis, 'fetch', async () => new Response(null, { status: 204 }));
  assert.equal(await request('/students/a/location', { method: 'DELETE' }), null);
});
