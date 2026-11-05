import { latestPlan } from './plan-data.js';

export function planOptions({ focus, count, constraints = '', refinement = '' }, profile) {
  const text = focus.trim();
  if (text.length < 3 || text.length > 300) throw new Error('Describe your training focus in 3–300 characters.');
  const available = profile?.available_days?.length || 0;
  if (!available) throw new Error('Save your training profile before generating a plan.');
  const workouts = Number(count);
  if (!Number.isInteger(workouts) || workouts < 1 || workouts > Math.min(7, available)) throw new Error('Choose a workout count that fits your available days.');
  const extra = constraints.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
  if (extra.length > 10) throw new Error('Add up to 10 constraints, one per line.');
  if (refinement.trim().length > 500) throw new Error('Keep changes to your latest plan within 500 characters.');
  return { focus: text, workouts_per_week: workouts, extra_constraints: extra, refinement_instructions: refinement.trim() || null };
}

export function generatedRecord(result) {
  if (result?.status !== 'completed' || typeof result.plan_id !== 'string' || !result.plan_id) {
    const messages = {
      out_of_scope: 'This request is outside workout planning. Try a training goal such as building strength or improving fitness.',
      insufficient_evidence: 'There isn’t enough supporting evidence for this request. Try a different training focus.',
      generation_error: 'The model couldn’t produce a valid plan. Try simplifying your request.',
    };
    throw new Error(messages[result?.status] || 'A saved plan wasn’t confirmed. Refresh your saved plan before trying again.');
  }
  return latestPlan([{ id: result.plan_id, plan: result.plan, sources: result.sources || [], workout_feedback: [] }]);
}
