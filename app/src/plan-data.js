export function latestPlan(records) {
  if (!Array.isArray(records)) throw new Error('The saved-plan response is invalid. Please try again.');
  if (!records.length) return null;
  const record = records[0];
  const plan = record?.plan;
  if (!record?.id || !plan || typeof plan.title !== 'string' || typeof plan.summary !== 'string'
      || typeof plan.safety_note !== 'string' || !Array.isArray(plan.weeks) || !plan.weeks.length
      || plan.weeks.some((week) => !Array.isArray(week.workouts) || !week.workouts.length
        || week.workouts.some((workout) => !Array.isArray(workout.exercises) || !workout.exercises.length))) {
    throw new Error('This saved plan could not be read. Please try loading it again.');
  }
  return record;
}

export function sourceLink(value) {
  try {
    const url = new URL(value);
    return ['http:', 'https:'].includes(url.protocol) ? url.href : null;
  } catch { return null; }
}

export function savedDate(value) {
  if (!value) return '';
  // SQLite timestamps are UTC even when they omit the timezone suffix.
  const normalized = /(?:Z|[+-]\d{2}:?\d{2})$/i.test(value) ? value : `${value.replace(' ', 'T')}Z`;
  const date = new Date(normalized);
  return Number.isNaN(date.getTime()) ? '' : date.toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' });
}
