# App Theme

Dark, professional visual system for the desktop app. Colors are drawn from
football (pitch green, ball white) but desaturated and used as accents on a
neutral dark base — not as dominant surface colors.

Reference implementation location: `apps/desktop/ui/theme` (centralized QSS,
per M1.4). No inline/ad-hoc styling on individual widgets.

---

## 1. Design Principles

* **Neutral base, colored accents.** Backgrounds and panels stay charcoal/slate.
  Color is reserved for meaning: state, action, status — never decoration.
* **One accent color dominates.** The muted pitch-green is the single primary
  accent. Info/warning/error exist but are used sparingly and only for their
  semantic purpose.
* **No saturated primaries.** Avoid pure red/green/yellow — they read as game
  UI. Every hue below is pulled down in saturation and lightness from its
  "obvious" sports equivalent.
* **Off-white, not white.** Pure white text on a dark background is harsh.
  Text colors are warm-neutral grays.
* **Single-hue for data, not traffic-light coding.** Confidence scores,
  candidate rankings, etc. use shades of one hue (the accent), not a
  red→yellow→green gradient.

---

## 2. Color Tokens

### Surfaces

| Token                 | Hex         | Usage                             |
| --------------------- | ----------- | --------------------------------- |
| `--bg-base`         | `#12161C` | App window background             |
| `--bg-panel`        | `#1D222B` | Cards, panels, side bars          |
| `--bg-panel-raised` | `#242A34` | Hover state / elevated panel      |
| `--border`          | `#2A303B` | Default hairline border           |
| `--border-strong`   | `#3A4150` | Emphasized divider, focus outline |

### Text

| Token                | Hex         | Usage                        |
| -------------------- | ----------- | ---------------------------- |
| `--text-primary`   | `#E8E9ED` | Primary body/label text      |
| `--text-secondary` | `#D5D8DE` | Secondary text               |
| `--text-muted`     | `#8B92A0` | Captions, disabled, metadata |
| `--text-disabled`  | `#4B525E` | Disabled control labels      |

### Accent — Pitch Green (primary)

| Token                     | Hex                       | Usage                                          |
| ------------------------- | ------------------------- | ---------------------------------------------- |
| `--accent`              | `#4A8C72`               | Primary buttons, active states, links          |
| `--accent-hover`        | `#5A9C82`               | Hover state on accent elements                 |
| `--accent-bg`           | `rgba(74,140,114,0.15)` | Accent tint background (badges, selected rows) |
| `--accent-text-on-fill` | `#0D1A15`               | Text/icon color on solid accent fill           |
| `--accent-text`         | `#6FBFA0`               | Accent-colored text on dark background         |

### Semantic — Info / Warning / Danger

| Token           | Hex                      | Usage                                    |
| --------------- | ------------------------ | ---------------------------------------- |
| `--info`      | `#5B7FA6`              | Informational highlights, neutral status |
| `--warning`   | `#C99A4A`              | Non-critical warnings (e.g. degraded AI) |
| `--danger`    | `#B5544B`              | Errors, failed requests                  |
| `--danger-bg` | `rgba(181,84,75,0.15)` | Error banner background                  |

### Data / Ranking (single-hue ramp, accent-based)

Use for candidate confidence, ranking order — not red/yellow/green.

| Token                 | Hex         | Usage          |
| --------------------- | ----------- | -------------- |
| `--rank-1`(highest) | `#4A8C72` | Best candidate |
| `--rank-2`          | `#3E7862` | 2nd            |
| `--rank-3`          | `#325F4F` | 3rd            |
| `--rank-4`(lowest)  | `#26493C` | 4th+           |

---

## 3. Typography

* Font family: system default (Segoe UI on Windows) — no custom font unless
  branding requires it.
* Two weights only: regular (400) and medium (500). Avoid bold (600+) —
  reads heavy against the dark surface.
* Sentence case throughout. No ALL CAPS, no Title Case except proper nouns.

| Role                 | Size | Weight | Color              |
| -------------------- | ---- | ------ | ------------------ |
| Panel heading        | 16px | 500    | `--text-primary` |
| Body / labels        | 13px | 400    | `--text-primary` |
| Secondary / metadata | 12px | 400    | `--text-muted`   |
| Button label         | 13px | 500    | context-dependent  |

---

## 4. Component Guidelines

### Buttons

* **Primary** (e.g. "Confirm frame"): solid `--accent` fill, `--accent-text-on-fill`
  text, no border. One primary button per view max.
* **Secondary** (e.g. "Prev/Next frame"): transparent fill, `0.5px solid --border`,
  `--text-primary` label. Hover → `--bg-panel-raised`.
* Avoid disabling buttons where possible; if disabled, drop opacity of label
  to `--text-disabled` rather than graying the whole control.

### Status badges (candidate labels, model status, etc.)

* Background: role-appropriate `-bg` tint (e.g. `--accent-bg`)
* Text: role-appropriate `-text` color (e.g. `--accent-text`)
* Never plain-color text on a tinted background — always pull from the same
  hue family.

### Panels / Cards

* `--bg-panel` background, `0.5px solid --border`, `12px` corner radius,
  `16px` internal padding.
* No drop shadows. Flat surfaces only — depth comes from color contrast
  between `--bg-base` and `--bg-panel`, not shadows.

### Diagnostics / secondary panels

* Slightly recessed: `--bg-base` background instead of `--bg-panel`, to
  visually de-emphasize relative to the main review panel.

---

## 5. States

| State                    | Treatment                                                                      |
| ------------------------ | ------------------------------------------------------------------------------ |
| Hover (buttons/rows)     | Background steps to `--bg-panel-raised`or `--accent-hover`                 |
| Active/pressed           | Scale 0.98 or darken fill by one step                                          |
| Selected (candidate row) | `--accent-bg`background,`--border-strong`left border (2px, square corners) |
| Disabled                 | Label →`--text-disabled`, no background change                              |
| Focus (keyboard)         | `1px solid --accent`outline, no glow/blur                                    |

---

## 6. Loaders

Two loader types. Which one appears is fully deterministic — driven by the
`AnalysisRequest` state, not by "feel" or per-screen choice.

### Primary — Card swap (red / yellow)

* A red card and a yellow card, both on screen simultaneously, with a subtle
  3D tilt (`perspective` + `rotateY`), continuously swapping horizontal
  position in a loop.
* Reserved for the single highest-stakes wait in the app: the AI is deciding.
* Colors stay true to their referee-card identity here — this is the one
  deliberate exception to the desaturated palette, because the metaphor
  ("waiting on a decision") only reads if the cards are recognizable.

**Trigger (deterministic):** shown whenever an `AnalysisRequest` is in
`QUEUED` or `RUNNING` state — i.e. the operator pressed the hotkey and the
pipeline is producing candidates. Hidden the instant the request reaches
`COMPLETED` or `FAILED`.

### Secondary — Football spin

* A ball icon (pentagon/hexagon panel pattern) in continuous rotation.
* Neutral, low-emphasis — communicates "something is loading" without
  implying a decision is being made.

**Trigger (deterministic):** shown for any blocking wait that is *not* tied
to an `AnalysisRequest` — app startup / model warm-up, opening or buffering a
video file, retry-on-failure reconnect attempts, any other infrastructure/IO
wait.

### Rule of thumb

> If the wait is tied to an `AnalysisRequest`, use the card swap.
> If it's infrastructure/IO, use the football spin.

Never show both at once — they represent mutually exclusive states of the app.

### Implementation — Card swap

```css
.loader-cards {
  position: relative;
  width: 90px;
  height: 90px;
  perspective: 600px;
}

.loader-cards .card {
  position: absolute;
  top: 50%;
  left: 50%;
  width: 34px;
  height: 48px;
  border-radius: 4px;
  margin: -24px 0 0 -17px;
  transform-style: preserve-3d;
  animation-duration: 1.6s;
  animation-timing-function: ease-in-out;
  animation-iteration-count: infinite;
}

.loader-cards .card--yellow {
  background: #C99A4A;
  animation-name: card-swap-right;
}

.loader-cards .card--red {
  background: #B5544B;
  animation-name: card-swap-left;
  animation-delay: 0s; /* both cards run in lockstep, no offset */
}

@keyframes card-swap-right {
  0%   { transform: translateX(-26px) rotateY(25deg); }
  50%  { transform: translateX(26px)  rotateY(-25deg); }
  100% { transform: translateX(-26px) rotateY(25deg); }
}

@keyframes card-swap-left {
  0%   { transform: translateX(26px)  rotateY(-25deg); }
  50%  { transform: translateX(-26px) rotateY(25deg); }
  100% { transform: translateX(26px)  rotateY(-25deg); }
}
```

Markup:

```html
<div class="loader-cards">
  <div class="card card--yellow"></div>
  <div class="card card--red"></div>
</div>
```

Notes:

* Both cards animate on the same 1.6s loop, exactly out of phase, so they
  continuously cross paths at center.
* `rotateY` peaks at the midpoint crossing to sell the 3D pass-through.
* Z-order (`z-index`) is intentionally left unset/equal — the slight overlap
  at crossing is fine and reads as depth, not a bug.

### Implementation — Football spin

```css
.loader-ball {
  width: 44px;
  height: 44px;
  border-radius: 50%;
  background: #D5D8DE;
  animation: ball-spin 1.2s linear infinite;
}

@keyframes ball-spin {
  from { transform: rotate(0deg); }
  to   { transform: rotate(360deg); }
}
```

Markup (pentagon pattern as inline SVG so it rotates as one unit with the ball):

```html
<div class="loader-ball">
  <svg viewBox="0 0 100 100" width="44" height="44">
    <polygon points="50,20 62,32 58,48 42,48 38,32" fill="#1D222B"/>
    <line x1="50" y1="20" x2="50" y2="5"  stroke="#1D222B" stroke-width="3"/>
    <line x1="62" y1="32" x2="80" y2="25" stroke="#1D222B" stroke-width="3"/>
    <line x1="58" y1="48" x2="70" y2="65" stroke="#1D222B" stroke-width="3"/>
    <line x1="42" y1="48" x2="30" y2="65" stroke="#1D222B" stroke-width="3"/>
    <line x1="38" y1="32" x2="20" y2="25" stroke="#1D222B" stroke-width="3"/>
  </svg>
</div>
```

Notes:

* Linear timing, constant speed — this loader should feel mechanical/neutral,
  not bouncy or eased (that emphasis is reserved for the card swap).
* Single continuous rotation, no direction change, no pause.

---

## 7. What to Avoid

* Pure white (`#FFFFFF`) or pure black (`#000000`) anywhere.
* Saturated grass-green (`#2ECC71`-class) or ball-white as large fill areas.
* Red/yellow/green traffic-light coding for anything other than genuine
  error/warning/success states.
* Gradients, glows, neon borders, drop shadows — flat surfaces only.
* More than one accent-filled primary action visible at once.
