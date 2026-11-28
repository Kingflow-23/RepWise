/** Pure helpers for logging sets, reps and weights, and for the training log. */

export function performancePath(userId, planId, weekNumber, workoutIndex) {
  return `/students/${encodeURIComponent(userId)}/plans/${encodeURIComponent(planId)}/weeks/${encodeURIComponent(weekNumber)}/workouts/${encodeURIComponent(workoutIndex)}/performance`;
}

/**
 * Build the POST body from form rows ({name, sets, reps, weight} as typed).
 * Rows left completely empty are skipped; a partly filled row must be complete.
 * Throws a readable error naming the exercise.
 */
export function performanceBody(rows) {
  const exercises = [];
  for (const row of rows) {
    const sets = String(row.sets ?? '').trim();
    const reps = String(row.reps ?? '').trim();
    const weight = String(row.weight ?? '').trim().replace(',', '.');
    if (!sets && !reps && !weight) continue;
    const name = String(row.name || '').trim();
    const setCount = Number(sets);
    if (!sets || !Number.isInteger(setCount) || setCount < 0 || setCount > 50) throw new Error(`${name}: enter the number of sets you did (0–50).`);
    if (!reps || reps.length > 100) throw new Error(`${name}: enter your reps, for example “10, 10, 8”.`);
    let weightKg = null;
    if (weight) {
      weightKg = Number(weight);
      if (!Number.isFinite(weightKg) || weightKg < 0 || weightKg > 1000) throw new Error(`${name}: the weight must be between 0 and 1000 kg, or empty.`);
    }
    exercises.push({ exercise_name: name, sets_completed: setCount, reps_per_set: reps, weight_kg: weightKg });
  }
  if (!exercises.length) throw new Error('Fill in at least one exercise before saving.');
  return { exercises };
}

/** Index saved records by lower-case exercise name, to prefill the form. */
export function readPerformance(payload) {
  if (!Array.isArray(payload)) throw new Error('RepWise returned an unexpected performance record.');
  const byName = new Map();
  payload.forEach((item) => {
    if (item && typeof item.exercise_name === 'string') byName.set(item.exercise_name.toLowerCase(), item);
  });
  return byName;
}

/** One readable line per exercise: "Bodyweight squat · 3 sets · 10, 10, 8 · 40 kg". */
export function performanceLine(record) {
  const parts = [record.exercise_name, `${record.sets_completed} ${record.sets_completed === 1 ? 'set' : 'sets'}`, record.reps_per_set];
  if (record.weight_kg != null) parts.push(`${record.weight_kg} kg`);
  return parts.join(' · ');
}

const STATUS_LABELS = { completed: 'Completed', partially_completed: 'Partly done', not_completed: 'Not done' };

/** Keep what the training log shows from GET /history, newest plan first. */
export function readHistory(payload) {
  if (!payload || !Array.isArray(payload.workouts) || !payload.summary) throw new Error('RepWise returned an unexpected training history.');
  const workouts = payload.workouts
    .filter((entry) => entry && typeof entry.day === 'string' && typeof entry.focus === 'string')
    .map((entry) => ({
      title: `Week ${entry.week_number} · ${entry.day} · ${entry.focus}`,
      status: STATUS_LABELS[entry.completion_status] || 'No feedback yet',
      lines: (entry.performance || []).map(performanceLine),
    }));
  return { workouts, logged: Number(payload.summary.workouts_logged) || 0 };
}

/** Explain a refused save, from the HTTP status and the API detail. */
export function performanceErrorMessage(status, detail) {
  // The exercise check lists the workout's real exercise names: useful and safe.
  if (status === 422 && detail?.code === 'exercise_not_found' && typeof detail.message === 'string' && detail.message.length <= 400) {
    return detail.message;
  }
  if (status === 422) return 'Check the numbers you entered, then try again.';
  if (status === 404) return 'This workout is no longer in your active program. Refresh your programs.';
  if (status === 401) return 'Your session has expired. Please sign in again.';
  return 'Your performance couldn’t be saved. Please try again in a moment.';
}
