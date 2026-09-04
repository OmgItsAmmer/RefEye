# RefEye — Operator Guide

RefEye watches the match, and when you press a key it tells you which frames
are most likely the moment of ball contact. You decide which one is right.

---

## Starting up

Double-click **RefEye.exe**.

The window opens immediately and video starts playing. The AI loads in the
background and takes a few seconds — the badge in the bottom-right corner
tells you where it is:

| Badge | Meaning |
|---|---|
| **AI loading** | Still starting. Video works; analysis is not ready yet. |
| **AI ready** | Everything is running normally. |
| **AI degraded** | Running, but not on the preferred model. Results are still produced — hover the badge for the reason. |
| **AI unavailable** | Analysis is not possible. Video still plays. Check the log in `logs/`. |

You never need to restart RefEye because the AI failed. Video and analysis are
independent.

---

## The main screen

**Left — Live preview.** The match, playing continuously. It keeps running
while analysis happens; you are never locked out.

**Right — Analysis.** Idle until you trigger it, then the review panel.

**Bottom right — Diagnostics.** Frame rate, buffer state, timings. Useful when
something feels wrong. Press **Hide** to collapse it.

---

## Analysing a moment

When you see something you want the exact frame for, press **F8**.

You do not need to press it at the precise instant. RefEye keeps the last
several seconds of video in memory and analyses that whole window, so pressing
it a second or two late is fine.

While it works, two cards swap back and forth — the AI is deciding. This takes
roughly a second or two.

---

## Reviewing candidates

You get up to five candidates, best first.

**The picture** is the frame RefEye thinks is the contact.

**Below it:**
- the action type — Pass, Shot, Cross, Header, Goalkeeper contact
- `Candidate 2 of 5` — where you are in the list
- `Frame 1732 · +2 from AI frame · 18/31` — which frame you are on, how far you
  have stepped from the AI's pick, and your position in the available frames
- a confidence line, plus any caveats (for example "ball position estimated")

### Moving around

Two separate controls, because they answer two different questions.

| Question | Control | Keys |
|---|---|---|
| *Is this the right moment?* | Candidate ‹ › | **↑** / **↓** |
| *Is this the exact frame?* | Frame ‹ › | **←** / **→** |
| *Take me back to the best one* | Best | **Home** |

You can also click any row in **Alternatives** to jump straight to it. The
selected row is highlighted.

### Confirming

When the frame is right, press **Enter** or click **Confirm frame**. RefEye
saves that exact frame as a JPEG into the `exports/` folder (next to the
executable) and the status bar confirms the filename.

If none of the candidates look right, press **Retry analysis** — or just press
**F8** again.

---

## Keyboard reference

| Key | Action |
|---|---|
| **F8** | Analyse the recent play |
| **←** / **→** | Previous / next frame |
| **↑** / **↓** | Previous / next candidate |
| **Home** | Jump to the best candidate |
| **Enter** | Confirm the current frame |

Every key can be changed — see Settings below.

**One limitation to be aware of:** these keys work while RefEye is the active
window. They do not currently work from inside another application.

---

## Settings

Everything adjustable lives in `config/default.yaml` next to the executable.
Open it in Notepad. The values most likely to matter:

```yaml
video:
  local_file:
    path: "./clips/match.mp4"   # the video to play

buffer:
  recent_window_seconds: 20     # how far back F8 looks

shortcuts:
  analyze: "F8"                 # change any key here

persistence:
  exports_directory: "./exports"   # where confirmed frames are saved
```

Save the file and restart RefEye.

If RefEye will not start after an edit, it will tell you exactly which setting
is wrong — a typo is reported rather than silently ignored.

---

## When something goes wrong

**"No video buffered yet"** — playback has not started. Wait a moment.

**"AI analysis could not complete"** — the analysis failed. The video keeps
playing. Press F8 again; if it keeps happening, send us the newest file from
the `logs/` folder.

**"No ball-contact actions were found"** — RefEye analysed the window and found
nothing. Not an error. Common when the play is a long build-up with no clear
contact, or the ball was not visible.

**Video will not open** — the panel shows the reason. Usually the path in
`config/default.yaml` is wrong.

### Sending a report

The `logs/` folder holds one file per session. Send the newest one. It records
what happened, including a **request id** for every analysis, so we can trace
exactly which run went wrong.

Logs contain no video and no personal data — only timings, frame numbers and
model names.

---

## What RefEye is and is not

RefEye **proposes**; you **decide**. It is an assistant, not a referee.

It does not make offside calls, does not decide whether a foul occurred, and
does not make any decision on its own. Its job is to save you from scrubbing
frame by frame to find a contact — the final judgement stays yours.
