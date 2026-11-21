export function savedPlan(record) {
  const plan = record?.plan;
  if (!record?.id || !plan || typeof plan.title !== 'string' || typeof plan.summary !== 'string'
      || typeof plan.safety_note !== 'string' || !Array.isArray(plan.weeks) || !plan.weeks.length
      || plan.weeks.some((week) => !Array.isArray(week.workouts) || !week.workouts.length
        || week.workouts.some((workout) => !Array.isArray(workout.exercises) || !workout.exercises.length))) {
    throw new Error('This saved plan could not be read. Please try loading it again.');
  }
  return record;
}

export function activePlan(record) {
  if (record === null) return null;
  const saved = savedPlan(record);
  if (saved.status !== 'active') throw new Error('This plan is not active. Refresh your active program.');
  return saved;
}

export function draftPlans(records) {
  if (!Array.isArray(records)) throw new Error('The draft-plan response is invalid. Please try again.');
  return records.map((record) => {
    const saved = savedPlan(record);
    if (saved.status !== 'draft') throw new Error('This response contains a plan that is not a draft.');
    return saved;
  });
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
