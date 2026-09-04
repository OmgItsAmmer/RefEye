# App Theme

Monochrome, technical, "operator console" visual system for the desktop app —
cloned from the reference implementation at `sample_ui/` (a Tauri/React video
management console). The design language: near-black surfaces, a single
off-white primary used for every interactive/active state, razor-thin
borders, sharp geometric corners, and uppercase micro-labels set in a
monospace face wherever the value is technical/data-like. Color is almost
never decorative — status dots and the alert/success hues are the only place
saturated color appears at all.

Reference implementation location: `apps/desktop/ui/theme` (centralized QSS +
tokens, per M1.4). No inline/ad-hoc styling on individual widgets. Font files
are vendored in `apps/desktop/ui/theme/fonts/` and loaded at startup via
`QFontDatabase.addApplicationFont` — the app must render identically with no
network access and no fonts pre-installed on the machine.

---

## 1. Design Principles

* **Monochrome first.** One primary color — an off-white, `#FAFAFA`, never
  pure `#FFFFFF` — carries every active/selected/primary-action state. There
  is no colored "brand accent." This reads as a premium, zero-distraction
  console rather than a themed consumer app.
* **True near-black, not charcoal.** Surfaces are genuinely dark
  (`#121212` base, `#1A1A1A` panel) — darker and flatter than the previous
  pitch-green iteration of this theme. Depth comes from a two-step surface
  ramp plus hairline borders, not from lightness gradients.
* **Color is reserved for two states only.** Red (`#FF3366`) means alert/
  recording/error. Green (`#00FF66`) means active/success/streaming — used
  exclusively as a small status dot, never as a fill, button, or large
  surface. Nothing else on screen is allowed to borrow these hues.
* **Uppercase + wide tracking for labels, sentence case for content.**
  Section headers, button labels, status text, and metadata captions are
  uppercase with wide letter-spacing, set small (9–11px). Actual content
  (camera names, headings, body copy) stays sentence case at a normal
  weight. Mixing the two signals what's chrome vs. what's data.
* **Two-typeface system.** `Outfit` (a geometric humanist sans) is the UI
  voice — headings, labels, buttons, body text. `Space Mono` is the data
  voice — anything technical or numeric: timestamps, identifiers, source
  paths, on-video HUD text, counters. A value that looks "measured" should be
  in Space Mono; a value that's prose should be in Outfit.
* **Sharp, not rounded.** Corners are small (2–8px) or square. Nothing in
  this system uses a large soft border-radius. The only fully round shapes
  are status dots and count badges.
* **Flat surfaces, glow only on the video canvas.** UI chrome (panels,
  buttons, sidebars) never has a drop shadow or glow. Soft colored glow is
  reserved for things actually drawn on top of live video (zone outlines,
  line-crossing strokes) — it reads as "projected onto the feed," not as a
  UI embellishment.

---

## 2. Color Tokens

### Surfaces

| Token           | Hex       | Usage                                     |
| --------------- | --------- | ------------------------------------------ |
| `--bg-base`     | `#121212` | App window background                      |
| `--bg-surface`  | `#1A1A1A` | Panels, cards, sidebar, top bar             |
| `--bg-surface-2`| `#222222` | Hover/raised state on a surface             |
| `--border`      | `#2A2A2A` | Default hairline border                     |
| `--border-strong`| `#3A3A3A`| Emphasized divider / focus ring fallback    |

### Text

| Token             | Hex       | Usage                                  |
| ------------------ | --------- | --------------------------------------- |
| `--text-primary`  | `#FAFAFA` | Same value as `--primary` — active/emphasized text |
| `--text-main`     | `#E0E0E0` | Default body/label text                 |
| `--text-muted`    | `#525252` | Captions, disabled, secondary metadata  |
| `--text-disabled` | `#3A3A3A` | Disabled control labels                 |

### Primary (monochrome accent)

| Token                  | Value                    | Usage                                             |
| ---------------------- | ------------------------ | -------------------------------------------------- |
| `--primary`           | `#FAFAFA`               | Active nav item, focused/primary button, selected state |
| `--primary-bg`        | `rgba(250,250,250,0.10)`| Tint background for selected rows / hover fill      |
| `--primary-border`    | `rgba(250,250,250,0.35)`| Border on an outlined-primary control               |
| `--primary-text-on-fill` | `#121212`             | Text on a solid-primary-filled surface (rare — most primary controls are outlined, not filled) |

### Semantic — used only for these two states, nowhere else

| Token           | Hex/Value                | Usage                                    |
| --------------- | -------------------------- | ------------------------------------------ |
| `--alert`      | `#FF3366`                 | Recording indicator, errors, intrusion/danger |
| `--alert-bg`   | `rgba(255,51,102,0.10)`   | Alert tint background                       |
| `--success`    | `#00FF66`                 | Streaming/online/confirmed status dot only  |

### Data ramp (single hue, never traffic-light)

Candidate ranking still uses one hue ramped by opacity against the primary —
consistent with the rest of the app's "no traffic-light coding" rule, just
now expressed in monochrome instead of green:

| Token          | Value                    | Usage          |
| -------------- | --------------------------- | -------------- |
| `--rank-1`    | `rgba(250,250,250,0.90)`   | Best candidate |
| `--rank-2`    | `rgba(250,250,250,0.65)`   | 2nd            |
| `--rank-3`    | `rgba(250,250,250,0.40)`   | 3rd            |
| `--rank-4`    | `rgba(250,250,250,0.22)`   | 4th+           |

---

## 3. Typography

* **UI face:** `Outfit` — weights 400 (Regular), 500 (Medium), 600 (SemiBold).
  Vendored as a variable font (`fonts/Outfit[wght].ttf`); the three weights
  above are the only ones used.
* **Data face:** `Space Mono` — Regular 400, Bold 700, Italic 400. Vendored
  as static TTFs. Used for: timestamps, frame/candidate counters, source
  paths and codec strings, on-video HUD text, diagnostic values.
* **Case rule:** UI chrome (section headers, button labels, status text,
  metadata captions) is uppercase with wide letter-spacing. Content (a
  camera name, a heading describing what the panel shows, body sentences,
  descriptions) stays sentence case. When in doubt: if it's a label
  *about* something, uppercase it; if it *is* the something, don't.
* **No text-transform in QSS** — Qt style sheets don't support
  `text-transform` or `letter-spacing`. Uppercase labels are uppercased in
  Python at the call site (a single `.upper()` on the string, never a
  stylesheet trick); letter-spacing is applied via
  `QFont.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, px)` on that
  widget's font.

| Role                     | Size | Weight | Face        | Case / tracking             | Color            |
| ------------------------- | ---- | ------ | ----------- | ---------------------------- | ------------------ |
| Screen/section title      | 15px | 600    | Outfit      | UPPERCASE, +1.5px tracking   | `--text-main`    |
| Panel content heading      | 18px | 600    | Outfit      | Sentence case                | `--text-main`    |
| Body / labels              | 13px | 400    | Outfit      | Sentence case                | `--text-main`    |
| Micro caption / meta label | 10px | 500    | Outfit      | UPPERCASE, +1px tracking     | `--text-muted`   |
| Data value (timestamp, id) | 11px | 400    | Space Mono  | As-is                        | `--text-main`    |
| Button label               | 10px | 600    | Outfit      | UPPERCASE, +1.5px tracking   | context-dependent |

---

## 4. Component Guidelines

### Buttons

* **Default control** — outlined, not filled: transparent background,
  `1px solid --border`, `--text-main` label, small sharp radius (`4px`).
  Hover → border and label both step to `--primary`, background tints to
  `--primary-bg`. This "outline that lights up on hover" is the single most
  characteristic control in the system — almost nothing is a solid fill.
* **Primary action** (e.g. "Confirm frame"): still outlined, using
  `--primary-border` at rest and `--primary` + filled `--primary-bg` on
  hover/pressed, `--text-primary` label. One primary action per view. A
  *filled* primary button (solid `--primary` background) is reserved for
  exactly one place: nowhere in the reference app, in fact — the sample
  never fills a button solid. Keep that restraint here too; "primary" means
  brighter outline + brighter label, not a filled block.
* **Danger control** (stop recording, retry after failure): `--alert`
  border and label at rest, `--alert-bg` fill on hover.
* Label text is always UPPERCASE, bold(ish) 600 weight, wide tracking, 10px.
* Disabled: label drops to `--text-disabled`, border drops to `--border`,
  cursor stays default. Never gray out the whole control block.

### Status dots (not badges — this system prefers a dot + label, not a pill)

* A `6px` (rendered slightly larger for legibility on desktop, `8px`)
  circular dot in the semantic color, followed by an uppercase Space-Mono-
  or-Outfit micro-label.
* `--success` dot = streaming/ready/confirmed. `--alert` dot = error/
  disconnected/recording. `--primary` dot (pulsing) = connecting/in-progress.
  `--text-muted`-colored dot = idle/neutral/no data.
* The recording dot pulses (expanding ring) — the one place in the whole
  system a status indicator animates on its own without user interaction.

### Count / unread badges (the one place a filled pill is correct)

* Small rounded-full pill, tinted background matching the semantic color at
  low opacity, semantic-colored bold numeral text. Used only for a count
  (unread alerts, queue depth) — never for a state label (states get a dot,
  not a pill).

### Panels / Cards

* `--bg-surface` background, `1px solid --border`, `8px` corner radius
  (sharper than the previous 12–14px), `16–24px` internal padding depending
  on density.
* No drop shadows, no gradients on chrome. Flat surfaces; depth is the
  `--bg-base` → `--bg-surface` → `--bg-surface-2` step, nothing else.
* A panel with a section header uses the micro-caption style (10px
  uppercase, `--text-muted`, wide tracking) for that header — not the 18px
  content-heading style, which is reserved for a heading that names actual
  content (e.g. a candidate's action type), not chrome.

### On-video overlays (badges drawn over the live feed)

* `--bg-surface` at ~80% opacity + a hairline `--border`, small sharp
  radius. This is the one place a "glassy" translucent surface is correct —
  everywhere else surfaces are opaque.
* Text inside an on-video badge uses Space Mono at 10–11px, since it's
  reporting a technical value (source name, frame id, codec).

### Diagnostics / secondary panels

* Recessed: `--bg-base` background instead of `--bg-surface`, same border
  and radius, to visually de-emphasize relative to the main panels.

---

## 5. States

| State                     | Treatment                                                                 |
| -------------------------- | --------------------------------------------------------------------------- |
| Hover (buttons)             | Border + label → `--primary`; background tints to `--primary-bg`         |
| Hover (rows / list items)   | Background steps to `--bg-surface-2`                                     |
| Active/pressed              | Background one step darker than hover; no scale transform                |
| Selected (nav item, row)    | `--primary-bg` background + `2px solid --primary` left border (square, not rounded) |
| Disabled                    | Label → `--text-disabled`, border → `--border`, no background change     |
| Focus (keyboard)            | `1px solid --primary` outline, no glow/blur                              |
| Recording / live-alert dot  | Pulses: scale 0.95→1.0 with an expanding, fading ring shadow, 2s loop     |

---

## 6. Motion

Real, deliberate motion — not decorative. Each transition exists to make a
state change legible, matching the reference app's actual animation
vocabulary:

| Motion               | Duration / easing        | Used for                                             |
| ---------------------- | --------------------------- | ------------------------------------------------------- |
| Fade + rise-in         | 400ms ease-out              | A panel/tile appearing (e.g. a candidate row being added), staggered ~100ms per item in a list |
| Scale-in               | 200ms ease-out              | A popover/flyout or a newly-selected card              |
| Slide-in from edge     | 200ms ease-out              | A side flyout (e.g. a configurator panel) entering      |
| Hover reveal           | 150–200ms ease, opacity+transform | Bottom hover-controls bar on a video tile, tooltips |
| Pulse (status dot)     | 2s ease, infinite            | Recording indicator only                                |
| Accordion expand       | 260ms ease (height/size)     | A collapsible section (diagnostics panel) opening       |

Qt has no CSS-transition equivalent in QSS, so every one of the above is a
real `QPropertyAnimation`/`QVariantAnimation` in code — see
`apps/desktop/ui/widgets/common.py` (`AnimatedButton`, `apply_elevation` is
retired — no more drop shadows — replaced by `PulsingDot`) and
`apps/desktop/ui/widgets/loaders.py`.

---

## 7. Loaders

The two-loader system (deterministic, driven by `AnalysisRequest` state, not
by "feel") is a domain feature of this app with no equivalent in the
reference UI — the reference app's only loading indicator is a plain
spinning ring. Rather than deleting a loading state that carries real
information (AI-deciding vs. plain infra-wait), both loaders are kept but
re-skinned in the new monochrome language: their surrounding chrome
(caption text, panel, borders) now follows the tokens above, while the two
loaders themselves keep their existing distinct identities:

* **Card swap** (`CardSwapLoader`) — reserved for "the AI is deciding"
  (`AnalysisRequest` `QUEUED`/running). Referee-card colors stay true
  (yellow/red) — the one deliberate exception to the monochrome rule, same
  reasoning as before: the metaphor only reads if the cards are
  recognizable.
* **Football spin** (`FootballSpinLoader`) — any infrastructure/IO wait.
  Restyled to sit on `--bg-surface` with a thin `--border` ring accent
  instead of the previous green glow, so it reads as part of this system
  rather than the old pitch-green one.

Trigger rules, animation timing, and "never show both at once" are
unchanged from the previous revision of this document.

---

## 8. Icons

The reference app uses Google's Material Symbols variable icon font
everywhere. Bundling and wiring up a full icon font is out of scope for this
pass (it would mean adding thousands of glyphs and a new
`QFontDatabase`-driven icon-widget layer with no icons currently drawn
anywhere in this codebase to replace). Controls keep this app's existing
text/glyph-based affordances (`‹ › ⏮` etc.) rendered in Outfit, sized and
spaced per the button rules above, rather than introducing an icon font
half-used across only some controls.

---

## 9. What to Avoid

* Pure white (`#FFFFFF`) or pure black (`#000000`) anywhere — use `#FAFAFA`
  / `#121212`.
* Any color as a large fill area other than the two near-black surface tones.
* A solid-filled button — every button in this system is outlined; only its
  border/label color and background tint change with emphasis/state.
* Drop shadows or glows on UI chrome (panels, buttons, sidebars). Glow is
  reserved for marks drawn on top of the live video.
* Rounded-pill shapes anywhere except status dots and count badges.
* Uppercase tracking on content text, or sentence case on chrome labels —
  the two must not swap roles.
* Red/green used for anything other than alert/success semantics.
