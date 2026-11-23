# Streamlit user guide

This guide covers the Streamlit app in `genai_app/`. RepWise is a local learning
prototype for general fitness planning, not a medical or coaching service. Read
the [setup guide](setup.md) to install and start the API and app.

## Sign in and set up your profile

1. Start the API and Streamlit app, then open the Streamlit URL.
2. In the sidebar, create an account or sign in. Keep your password private.
3. Open **Profile**, choose your experience, goal, available training days,
   session length, equipment, and any preferences, then save.

The saved profile constrains generated plans. Choose only days you can actually
train; a plan can use no more workouts per week than the number of available
days.

## Create a weekly plan

In **Create a plan**, enter a focus and choose how many workouts to schedule
each week. By default, the app selects up to three workouts, limited by your
available days. For example, with four available days, the default is three;
choose four explicitly to request four workouts.

Plan generation creates a **one-week template** by default. When you ask the
assistant for a plan for this week, it considers only your remaining profile
weekdays and, when connected, checks that each has enough free calendar time.
If fewer weekdays are available than the requested workout count, it reduces
the count and reports that in the result. Without a connected calendar, busy
times cannot be verified. A new plan becomes active when you have no active plan;
when you already have one, the new plan is saved as a draft and activation
waits for your approval.

Plans show their source links and saved status in **Saved plans**. The newest
plan is not necessarily the active one; drafts wait for approval and archived
plans have been replaced.

Creating a plan from **Create a plan** is a direct action: the new plan becomes
active immediately and the previous active plan is archived. In contrast, a
replacement plan proposed by the Assistant is saved as a draft and waits for
your approval before activation.

## Log completed or missed workouts

Use the workout feedback controls in a saved plan to record whether you
completed, partly completed, or missed a workout, how difficult it felt, and an
optional comment. RepWise cannot detect that you trained: it only knows what
you log. Feedback is saved as workout history and may inform a later adaptation.
Possible pain or injury concerns stop automatic adaptation and may require
approval.

Calendar suggestions skip workouts that already have feedback recorded,
including workouts marked as missed. They also skip workouts already booked in
RepWise and workout dates that have passed. Logging a missed workout does not
automatically create a replacement event.

## Connect and book Google Calendar events

Calendar access is optional. Follow the [one-time Google Calendar setup](setup.md#connect-google-calendar),
then sign in and open **Location & calendar** to connect your Google account.
RepWise reads busy periods only; it does not read event titles or details.

When you ask the assistant to find or book training times:

- It checks free time on the workout's planned day and in your configured time
  zone. It does not move workouts automatically.
- By default, it uses the current week if at least one planned workout day is
  today or later. Past workout dates are skipped. If all planned weekdays are
  earlier than today's weekday, it looks at the next week. If today is a planned
  day but all candidate times have passed, no slot is offered for that workout.
- A slot must be in the future, fit the workout duration, avoid calendar
  conflicts, and start and finish between 06:00 and 22:00 local time.
- If you ask it to book without choosing exact times, RepWise selects preferred
  free daytime times for the unbooked, unlogged workouts. If any of those
  workouts has no free time on its planned day, no booking action is proposed;
  RepWise does not create a partial schedule.
- Nothing is added until you approve the booking. RepWise checks for conflicts
  again before creating events. A retry does not intentionally duplicate a
  booking.

If no calendar is connected, RepWise cannot check availability or create
events. If a workout has no free time on its planned day, it does not silently
move the workout.

The Schedule Agent can also propose moving workouts after a missed session.
Those moves change the plan only after approval; they do not themselves create
calendar events. Automatic slot search excludes feedback-logged workouts, so a
logged missed workout will not receive an automatic slot suggestion.

## Other sections

- **Assistant**: ask questions, request a plan or schedule help, and review the
  specialists' results.
- **Approvals**: review pending actions and approve or decline each one.
- **Progress**: review the current session's agent timing and tool metrics.
- **Location & calendar**: manage the optional saved location and Google
  Calendar connection.
- **Settings**: view system status and account controls.

Assistant conversation turns are kept in the running process; they are not a
permanent chat archive.

## Delete your account

Open **Settings**, expand **Delete account permanently**, and type `DELETE`
exactly to confirm. RepWise then deletes the account, profile, plans, workout
history, saved location, memory, pending approvals, and the locally stored
Google Calendar credentials. This cannot be undone.

Calendar events already created in Google Calendar are external to RepWise and
remain in that calendar; delete them there if you also want them removed.
Deleting the RepWise account signs you out of the app.

## Health and privacy limits

RepWise provides general fitness information, not diagnosis or treatment. It
does not verify that a cited passage supports every exercise or that a plan is
medically appropriate. For pain, injury, or a medical condition, consult a
qualified health professional.

The app is designed for local use. Account and training records are stored in
the configured SQLite database; Google Calendar credentials are stored locally
by the RepWise installation. See [API and database](api.md) for the data model
and [security limits](api.md#security-limits) before exposing the service.
