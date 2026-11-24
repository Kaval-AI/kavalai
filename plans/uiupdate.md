# Backoffice UI update: carrying over the kavalapp console design

## Goal

Give the backoffice (`frontend/`) the look, components and navigation model of
the kavalapp console (`~/projects/kavalapp-main/console`), so the design work
done there is reused rather than redone, and fix the backoffice's UX debts while
each page is touched.

The two apps are close relatives: the console started as a fork of this
`frontend/`, both are Angular 20 standalone apps with signals, both use the
Kaval.AI visual toolkit (Charcoal Blue `#273e47`, Lime Moss `#acc12f`, Rosy
Granite; Chakra Petch / Space Grotesk / Space Mono), and the backoffice already
ships the same fonts and `login-background.png`. What differs is that the
console grew a small component library (`src/app/ui`), a sidebar shell and
consistent page patterns, while the backoffice stayed on ad-hoc DaisyUI markup.

The backoffice is mainly used to **debug agent runs** — Pulse's in particular:
what was asked, which steps ran, what each model call was sent and answered.
The order of the work follows that: the pages read daily come before the
settings pages opened a few times a year.

## Done

Fixed before the redesign, as changes of their own (2026-11-24, uncommitted):

- **The database password no longer reaches project members.** Project
  endpoints answer through `public_project` (no `db_password`, a
  `has_password` flag). The password is write-only; an empty one keeps the
  stored password, but only for the same type, host, port, user and database,
  so editing the host cannot send it to another server. Administrators can ask
  for it with `POST /projects/{id}/reveal-password` (logged), which is what the
  Projects page's **Show password** beside the connection strings does.
- **Test connection** checks the typed settings before they are saved (owners;
  a new project's settings, administrators only) and shows the database's own
  error instead of `Http failure response … 400`.
- Tests: `tests/backoffice` (169 passing, both backoffice database backends),
  the Karma suite (212 passing). Documented in `docs/ui/index.rst`,
  `docs/versions.rst` and `CLAUDE.md`.

### Steps 1–8 and 10 (2026-11-24, uncommitted)

- **1** The coexistence check passed: Tailwind 4 compiles DaisyUI into its
  utilities layer, so element defaults in `@layer base` did not disturb the
  pages not yet ported.
- **2–3** Login page and shell (top bar, user menu, sidebar with the project
  switcher, Settings, New project, Profile); the hamburger menu, breadcrumbs,
  `sidebar-menu`, `dropdown-menu`, `user-info`, the `tests`, `metrics` and
  `theme` pages and `NavigationService` are deleted.
- **4–8** Every page is ported: Overview and Settings (replacing both project
  editors), Conversations and the conversation, the run with its timeline and
  steps, Model calls with the text view of prompts, Agents, Runs, RAG, Users
  and Profile. Backend: `GET`/`PATCH /user/me`, `POST /user/me/password`, and a
  409 for a duplicate email address.
- **10** DaisyUI, Angular Material, the CDK, mermaid, nxt-json-view, prismjs,
  marked, ngx-markdown, normalize.css and uuid are uninstalled; the element
  defaults are global; `CLAUDE.md`, `AGENTS.md`, `docs/ui/index.rst` (text)
  and `docs/versions.rst` describe the result.

Still open: step 9 (project in the URL), new screenshots for
`docs/ui/index.rst` (the images still show the old interface), a pager where an
endpoint returns `total`, and the Runs page's one request per session.

## Where the backoffice stood

- **Navigation** is a hamburger mega-dropdown that opens on focus with a
  `setTimeout`. A `sidebar-menu` component exists but is not mounted. The
  active-project selector lives only on the Projects page; elsewhere the header
  shows the project name as a breadcrumb.
- **Project settings** have two editors: an inline one on the Projects page
  and `project-edit-page`, reachable only as `/project-edit/new`, so its
  members section is dead code. Viewers see Edit; non-admins see New Project.
- **Users**: the admin forms have no password field, though the API accepts
  one; "User profile" loads `/users/all`, which is admin-only, so it is empty
  for everyone else; the avatar `<img>` breaks for users with no picture.
- **Consistency**: five ways to show an error (DaisyUI alert, cursor-positioned
  toast, error modal, `alert()`/`confirm()`, plain text), several silent
  `console.error` failures, four date formats, three duration formats, mixed
  icon sets (Font Awesome, inline heroicons, Unicode ▼/✕, an unused Material
  Icons font), hand-written CSS with hex colours in four components, modals
  without Esc or focus handling.
- **Dead weight**: `tests` and `metrics` are placeholder pages; `theme` is
  routed but unlinked; `@angular/material`, `mermaid`, `nxt-json-view`,
  `prismjs`, `marked`, `ngx-markdown` are installed and unused; fonts load
  twice.
- **Charts**: Chart.js line charts with duplicated setup code on the Projects
  and Agents pages; a hand-built timeline (run tasks) and SVG scatter plot
  (RAG).

## Ground rules

- **Visible from the first step.** No phase builds the library ahead of use:
  each step brings over the `ui/` pieces its page needs, with their specs, so
  every step ends in a page that looks different and can be judged.
- **Look and feel only.** A step changes how a page looks and behaves on the
  API it has today. Backend changes are separate items, done when a page cannot
  do without them, not bundled into the port. "Load more" stays where the API
  has no `total`, for instance.
- **Copy, then adapt.** A `ui/` component moves with its template, stylesheet
  and spec into `frontend/src/app/ui/<name>/`, behind the barrel `ui/index.ts`,
  and `ui/README.md` says when to use which piece.
- **No i18n.** The backoffice is English only and has no `@angular/localize`.
  Strip `i18n*` attributes and replace `$localize` with plain strings while
  copying — an `i18n` attribute without the package fails at run time, not at
  build.
- **Keep Tailwind, drop DaisyUI.** Tailwind's utilities do the layout in most
  existing templates and cost nothing to keep; DaisyUI's components compete
  with the `ui/` ones and go when the last page using them is ported.
- **Licence.** kavalapp is closed source; the backoffice is Apache-2.0. Moving
  the console's UI code here publishes it under Apache-2.0. Ported files carry
  the backoffice's licence header.

## 1. Components to carry over

### The `ui/` library

| Piece | What it gives the backoffice | First needed in step |
|---|---|---|
| `button[uiButton]`, `a[uiButton]` | primary/secondary/ghost/danger/icon, sm/md/lg, chamfered primary | 2 (login) |
| `ui-field` | label, control, hint, error | 2 (login) |
| `ui-alert` | inline message by tone, with a Retry slot | 2 (login) |
| `ui-loading` / `ui-spinner` | centred and inline spinners | 2 |
| `ui-nav-item` | sidebar row with active indicator | 3 (shell) |
| `ui-menu` / `[uiMenuItem]` | anchored dropdown sheet | 3 (user menu, project switcher) |
| `ui-avatar` | initials or round picture; fixes the broken image | 3 |
| `ui-badge` | chamfered mono labels by tone | 3 (`read-only` in the switcher) |
| `ui-page-header` | section tag, title, subtitle, `backLink`, `page-actions`; replaces breadcrumbs | 4 |
| `ui-card` | chamfered surface, lime-ruled title, `card-actions`, `card-footer`, `tone` | 4 |
| `ui-table` | mono headers on a lime rule, `dense`, `clickable`, selected row | 4 (conversations) |
| `ui-toolbar` | wrapping row of filters | 4 |
| `ui-empty-state` | icon, title, one line, call to action | 4 |
| `ui-stats` / `ui-stat` | figure tiles with a lime edge | 4 (run detail), 5 (agents) |
| `ui-info` | label/value row, `mono` | 4 (run metadata) |
| `ui-copy-button` | copy with confirmation | 4 (IDs, JSON) |
| `ui-status-badge` | badge worded by status; its map becomes run/task states | 4 |
| `ui-chip` | "what this page is about" pill | 4 (conversations narrowed to an agent) |
| `ui-pager` | Previous/Next with range | when an endpoint returns `total` |
| `ui-progress` | determinate bar | 5 (RAG similarity) |
| `app-bar-chart` | stacked bars per day, SVG | 5 (agents), 6 (project overview) |
| `app-box-plot` | per-day distribution, SVG | optional, see Charts |
| `ui-dialog` | native `<dialog>` with Esc and backdrop close | 6 (replaces modals, `alert()`, `confirm()`) |
| `ui-favicon` | — | not ported (websites only) |

### Page-level patterns, not the pages

The console's pages are about chatbots and websites, so they do not port as
pages. Their patterns do:

- **Login** (`pages/login`): charcoal textured ground, two columns — a short
  intro with the section tag and wedge list, and the chamfered panel — Google
  first, an "or with email" divider, `ui-field`s, inline `ui-alert`. Sign-up
  and password reset do not apply (accounts come from
  `python -m kavalai.backoffice.users`); the panel says to ask an
  administrator for an account. The intro's wording follows the docs' style
  rules (no marketing tone).
- **Shell** (`layout/shell.*`): sticky charcoal top bar, 256 px sidebar, user
  menu with avatar, name and sign-out. Invitations, the alerts bell, the
  language switcher and the help chat are not ported; a collapsed sidebar on
  narrow screens is enough, without the console's full mobile burger.
- **List with side detail** (`pages/conversations`): a clickable table, the
  selected row opening a detail panel beside it. Fits conversations and model
  calls, which the backoffice now prints as one tall card per item.
- **Members card** (`pages/account/members-card.*`): avatar, name and email,
  role as a select for owners and a badge for viewers, remove confirmed in
  place, "(you)" marker. Replaces both members UIs.
- **Similarity search** (`pages/index/similarity-search.*`): the shape of the
  RAG query page — a toolbar of inputs, a results table with a similarity bar.
- **Action bar** (`.card-actions-bar`): the row of secondary buttons closing a
  card — "Test connection", "Open tasks", "Open model calls".
- **Danger zone**: delete confirmed by typing the name, in a `danger` card at
  the bottom of the settings page.

## 2. Visual language

- **Tokens.** The console's `:root` block (palette, lime and slate shades,
  semantic roles, 4 px spacing scale, type scale, radii, three elevations,
  focus ring, motion, header and sidebar sizes) moves into `src/styles.css`.
  The DaisyUI theme variables are mapped onto the tokens for the transition,
  so pages not yet ported pick up the same colours.
- **Typography.** Chakra Petch for headings, buttons and figures; Space Grotesk
  for text; Space Mono for labels, table headers, IDs, JSON. Fonts are loaded
  once, from the local `@font-face` rules.
- **Shape.** The chamfered corner is the signature; fields keep a small radius.
  Elevation via `filter: drop-shadow` on chamfered surfaces, since `clip-path`
  clips `box-shadow`.
- **Buttons.** One `primary` per view; `secondary` outlined; `ghost` for card
  actions; `danger` only for destructive actions; `icon` with an `aria-label`.
  A busy button holds a `ui-spinner` and is disabled.
- **Icons.** Font Awesome only, from npm (`@fortawesome/fontawesome-free`) as
  in the console, not the CDN. Material Icons, inline heroicons and Unicode
  glyphs go.
- **Global primitives** from the console's `styles.css`: `.section-tag`,
  `ul.wedge-list`, `pre.snippet`, `.mono`, `.form-hint`, `.card-actions-bar`.
- **Formats.** `date: 'short'` in tables and `'medium'` in detail; one
  `formatDuration` (ms below a second, seconds with one decimal, then
  minutes and seconds); token counts through `number`; IDs in mono, shortened
  to 8 characters, the full value in `title` and behind a copy button.
- **Charts.**
  - Counts per day (sessions, runs, messages; input, output and embedding
    tokens) become `app-bar-chart` stacked bars.
  - Durations stay a per-day line for now: in local use a day holds a handful
    of runs, and a box plot of three values says less than the average. The
    Chart.js setup duplicated in the Projects and Agents pages becomes one
    module whose colours, fonts and grid come from the tokens.
  - `app-box-plot` is ported only when a page has the volume to make it
    readable (an agent with many runs a day).
  - The run timeline and the RAG scatter plot stay custom, restyled with the
    tokens (lime for model calls, slate for tools, raspberry for failures).
  - Chart.js is removed only if the last line chart gets an SVG replacement
    that keeps hover read-outs; the console's charts show values only in
    native `<title>` tooltips.

## 3. Sidebar navigation

Top bar: logo, a "Backoffice" product tag, the user menu at the right (avatar,
name, Profile, Sign out).

Sidebar, top to bottom:

1. **Project switcher** — the active project's name, its database
   (`localhost:6543 / pulse`) and a `read-only` badge, opening a `ui-menu` of
   the user's projects. The one place the project changes, on every page.
2. **Project** — Overview, Conversations, Runs (today's Workflows page: the
   recent runs of every agent as task lanes), Model calls, Agents, RAG.
3. **Settings** — the project's settings (members are a card there, not a
   sidebar item of their own).
4. **Administration** (admins) — Projects, Users.
5. **Footer** — Documentation, GitHub, PyPI.

Breadcrumbs and the `isProjectRoute` route data go; `ui-page-header` with
`backLink` gives detail pages their way back.

**Project in the URL — a separate, optional step.** Moving project pages
under `/projects/:projectId/…` would make links shareable and reload-proof
and remove the `userDetails` subscription every page repeats. It touches every
route, link and spec, and changes nothing visible, so it is not part of the
shell step; it is step 9, to be decided on its own.

## 4. UX: project settings, connections, consistency

### Project settings page

One page replacing the inline editor and `project-edit-page`, one card per
concern, each card saving itself:

- **General** — name, description.
- **Database** — type (PostgreSQL / SQLite file); host, port, user, password,
  database, schema and RAG schema, or the SQLite file path. A `.form-hint`
  says what schema and RAG schema are. The password field reads "Unchanged —
  type to replace" when one is stored, and the card explains that changing
  the server or user asks for it again. The card ends in an action bar,
  **Test connection** and **Save**; the test's result is a `ui-alert` with the
  database's message, and the button holds a spinner while it runs. (The
  backend already supports this.)
- **Connection strings** — SQLAlchemy, asyncpg, JDBC and `KAVALAI_DB_URI`,
  password masked, each with `ui-copy-button`; **Show password** for
  administrators, as today. For a SQLite project the labels name SQLite, not
  PostgreSQL.
- **Access** — the read-only switch, with one sentence on what it does.
- **Members** — the members-card pattern; Add offers only users who are not
  yet members; the last owner's role and remove controls are disabled with the
  reason shown, rather than discovered through an error.
- **Danger zone** — delete, confirmed by typing the name in a `ui-dialog`.

A new project is the same page with General, Database and Access, a single
Create button, and Test connection available before Create.

Permissions are reflected, not only enforced: viewers get the page read-only
with a note saying so; New project appears only for administrators.

### Users and profile

- The users table shows avatar, name, email, and `admin` and `password` /
  `Google only` badges.
- One user form (`ui-dialog`) for create and edit, with a password field
  ("leave empty to keep" when editing); the API already hashes `password`.
- A **Profile** page for every user — name, picture URL, change password —
  needs a self-service endpoint (`GET`/`PATCH /user/me`, a password change
  asking for the current password). That is a backend item for step 8.

### Consistency rules, written into `ui/README.md`

- A failed load shows `ui-alert` with Retry; a failed save shows `ui-alert` in
  the card that was saved. Errors read FastAPI's `detail` through one helper,
  `core/http-error.ts`, as in the console. No `console.error`-only failures,
  no `alert()` / `confirm()`.
- Toasts only confirm success ("Saved", "Copied"), in one corner, never at the
  cursor.
- Loading is `ui-loading`; an empty list is `ui-empty-state` naming the next
  step; a destructive action confirms in place or in `ui-dialog`.
- Lists are tables; a tall item opens in a side panel or a page of its own.
- JSON is shown by `json-tree`, restyled with the tokens and the mono font;
  a model call's request and response get a "Show as text" toggle, so a prompt
  reads as messages rather than escaped JSON.
- Every page has the same content width.

## 5. Frontend unit testing

### Where it stands

- Karma with Jasmine in headless Chrome; 29 spec files, 212 specs, all
  passing. CI runs them in the `frontend-tests` job of
  `.github/workflows/ci.yml` (`npm test -- --watch=false --browsers=ChromeHeadless`).
- Specs build `TestBed` with `jasmine.createSpyObj` services and assert mostly
  on component fields; few look at the rendered page. Fixtures are written
  inline in every spec; there are no shared helpers.
- Six specs cover components nobody mounts (`dropdown-menu`, `sidebar-menu`),
  two cover placeholder pages, and `theme-page` and `toast` have none.

### The conventions the port adopts

They are the console's, which kept one spec per component, service and route
table:

- **Every component, directive, service and the route table has a spec**,
  added in the same change as the code. The `ui/` pieces arrive with their
  console specs, adapted for the removed i18n.
- **Page specs test what the user sees.** They render the template and find
  elements by `data-testid` or role, rather than reading component fields: a
  spec says "the error alert shows the server's `detail` and a Retry button",
  not "`connectionStatus.message` is set".
- **Each data-loading view has four cases**: loading, error with Retry, empty,
  and loaded. Each form has: invalid input refused, saving (button busy),
  server refusal shown, success.
- **Permissions are tested per role.** Admin, owner and viewer each get a case
  wherever the page hides or disables controls.
- **Services use `HttpTestingController`** and assert the URL, method and
  body of every call, so a frontend–backend contract change fails a spec.
- **Shared fixtures in `src/testing.ts`**, as in the console: `settle(fixture)`
  (flush and re-render), builders `makeProject`, `makeUser`, `makeMember`,
  `makeSession`, `makeRun`, `makeTask`, `makeModelCall` that take overrides,
  and `signedInAs('admin' | 'owner' | 'viewer')` for the user service.
- **A port rewrites its page's spec in the same change**, and deleting a
  component deletes its spec. No spec is left skipped (`xit`, `xdescribe`).
- **The route table spec** lists every route, checks that each resolves to its
  page, and, if step 9 is taken, that each old route redirects.

### Coverage and running

- New and changed files reach 100% statement coverage, as `CLAUDE.md` asks of
  the backend: `npm test -- --watch=false --code-coverage`, report in
  `frontend/coverage/frontend/`. Once `src/app/ui/` is complete, a
  `karma-coverage` `check` threshold for that folder makes the rule enforced
  rather than remembered.
- One file: `npm test -- --watch=false --include src/app/ui/card/card.spec.ts`.
- Specs never call a backend; HTTP is `HttpTestingController` or a spy.

### Above unit tests

Karma cannot show what a page looks like. Each step ends with a screenshot pass
of the pages it touched — the Playwright script used for the "before" shots,
run against the dev server with a local backoffice — compared with the previous
pass. Turning that into a repeatable check (Playwright with the API mocked by
route interception, so it needs no backend) is a follow-up, and its place in
the repository is an open decision below.

## 6. Steps

Each step ends with: the Karma suite and `pytest tests/backoffice` passing, a
development build, the screenshot pass, and the step's pages checked by hand in
the browser. A step is **done** when its pages use only `ui/` pieces and the
shared error, loading and empty patterns, have no DaisyUI classes, and their
specs follow section 5. Sizes are rough: S under a day, M one to two days, L
three or more.

1. **Coexistence check (S).** Add the tokens and element defaults to
   `styles.css` inside `@layer base`, port `ui-card` and `uiButton`, and put
   one ported card on an unported page. Confirm that Tailwind utilities and
   DaisyUI components on the rest of the page are unchanged — which layers
   DaisyUI 5 uses decides whether the console's element rules override it.
   Only then build on the approach.
2. **Login page (S).** The console's layout with the backoffice's two sign-in
   methods; `ui-field`, `ui-alert`, `uiButton`, `ui-loading`; the
   `core/http-error.ts` helper; `src/testing.ts`.
3. **Shell (M).** Top bar, sidebar with the project switcher, user menu with
   the avatar fallback. Delete `header-dropdown`, `sidebar-menu`,
   `dropdown-menu`, the `tests` and `metrics` pages and their specs.
4. **Debugging pages (L).** Conversations (table with a side detail panel,
   filters in a toolbar), session detail (the transcript, grouped per run),
   run tasks (timeline restyled, task list), model calls (table with a detail
   panel, the text view of prompts). On the existing API.
5. **Agents, Runs, RAG (M).** Agents with bar charts and the workflow graph in
   a card; Runs restyled; the RAG query page in the similarity-search shape.
6. **Projects and settings (M).** The projects list for administrators, the
   project overview (stats, charts), the settings page of section 4, the
   members card; `ui-dialog` replaces the remaining modals.
7. **Users (S).** Users table, one user form with a password field.
8. **Profile (S, backend + frontend).** `GET`/`PATCH /user/me` with a
   password change, and the Profile page.
9. **Project in the URL (M, optional).** Routes under `/projects/:projectId`,
   redirects from the old ones, the route table spec extended.
10. **Clean-up (S).** Remove DaisyUI, the unused packages (`@angular/material`,
    `mermaid`, `nxt-json-view`, `prismjs`, `marked`, `ngx-markdown`, and
    `@angular/cdk` if nothing uses it), Material Icons, the Font Awesome CDN
    link and the duplicate font import; delete the `theme` page or turn it
    into a `ui/` gallery. Update `CLAUDE.md` and `AGENTS.md` (frontend
    conventions, the `ui/` library, the end of DaisyUI) and `docs/ui/index.rst`
    with new screenshots.

## Backend items still open

- `GET`/`PATCH /user/me` for the profile (step 8).
- `offset` and `total` on `/agents/sessions/{pid}` and
  `/projects/{pid}/llm-call-stats`, only if a pager is wanted instead of
  "Load more".
- One `/projects/{pid}/runs` listing for the Runs page, replacing its
  agents → sessions → details fan-out (one request per session), only if the
  page proves slow on a real project.

## Open decisions

- **DaisyUI's removal** changes the convention `CLAUDE.md` states ("use
  Tailwind CSS + DaisyUI"); the plan keeps Tailwind.
- **Project in the URL** (step 9): worth it, or not.
- **Screenshots.** `docs/ui/projectinfopage.png` is embedded by `README.md`
  and the docs index; retaking it changes what the README shows without
  editing the README.
- **Where a browser-level UI check lives**, if the screenshot pass is made
  repeatable: a Playwright test in `frontend/` (an npm dependency) or in the
  Python suite (Playwright is already in the dev environment), and whether CI
  runs it.
- **Dark mode.** The console has none; the tokens make it a later addition
  rather than part of this work.
