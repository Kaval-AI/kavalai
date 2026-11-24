# Backoffice component library (`ui/`)

Every page of the backoffice is built from these pieces, so buttons, cards,
tables and menus look and behave the same everywhere. They come from the
kavalapp console, which built them on the Kaval.AI visual toolkit. Import them
from the barrel:

```ts
import { UiButton, UiCard, UiPageHeader } from '../../ui';
```

Each piece lives in its own folder with its template, stylesheet and spec.
Values come from the design tokens in `src/styles.css` (palette, spacing, type
scale, elevation, focus ring).

| Selector                          | What it is |
| --------------------------------- | ---------- |
| `button[uiButton]`, `a[uiButton]` | Buttons: `primary`, `secondary` (default), `ghost`, `danger`, `icon`; `size="sm\|md\|lg"`, `block`. |
| `ui-spinner`, `ui-loading`        | An in-progress ring; a centred one, with a `label`, for a view that is still loading. |
| `ui-card`                         | A white chamfered surface with a lime-ruled `title`/`icon`; `titleLink` makes the heading a link; `card-actions` content sits in the heading, `card-footer` content closes the card; `fill` makes it as tall as its grid row; `tone="danger\|dark"`; `selected`. |
| `ui-page-header`                  | `tag`, `title`, `section`, `backLink`/`backLabel`; subtitle as content, `page-title` and `page-actions` slots. |
| `ui-stats`, `ui-stat`             | A grid of figures: `label`, `hint`, optional `link` (and its `queryParams`); the content is the value. |
| `ui-table`, `ui-pager`            | The scroll container and styling for a plain `<table>` (`dense`, `top`, `clickable`; rows with `selected` or `muted`); the range and Previous/Next under it. |
| `ui-badge`, `ui-status-badge`     | Chamfered mono labels by `tone`; the status one words and colours `running`, `completed`, `failed`, `ok` and `error`. |
| `ui-alert`                        | An inline message by `tone` (`error` is the default), with an `alert-actions` slot for Retry. |
| `ui-empty-state`                  | `icon`, `title`, one line of help as content, `empty-actions` for the call to action. |
| `ui-info`                         | A label/value row; `mono` for identifiers. |
| `ui-toolbar`                      | A wrapping row of filters and actions. |
| `ui-progress`                     | A determinate bar: `value`, `max`, `tone="warning"`. |
| `ui-field`                        | `label`, `for`, `hint`, `error` around a projected control. |
| `ui-avatar`                       | Initials in the charcoal box, or a `round` picture (people); `size="sm\|md\|lg"`. |
| `ui-chip`                         | The context a page is about (the agent a list is narrowed to); a link when `link` is set. |
| `ui-nav-item`                     | A sidebar destination with its active indicator; `exact`, `compact`. |
| `ui-menu`, `[uiMenuItem]`         | A dropdown sheet and its entries; `danger` for sign-out; `align="start"`. |
| `ui-copy-button`                  | Copies its `text` and confirms. |
| `ui-dialog`                       | A modal box (the native `<dialog>`), centred: `open`, `title`, `closed` on a dismissal; buttons go in `dialog-actions`; `size="md"` (a decision, the default), `lg` or `xl` (a document — JSON, a log). |
| `app-bar-chart`                   | A stacked bar per day (`bars`, `series`), drawn as SVG; `fillDays` gives quiet days their gap. |

Outside `ui/`:

- `core/http-error.ts` — `errorMessage(err, fallback)`, the one way to turn a
  failed request into text. It reads FastAPI's `detail`.
- `core/format.ts` — `formatDuration(seconds)` (ms, s, min s; a dash for
  none) and `shortId(id)`, the one way every page writes them.
- `core/daily-stats.ts` — `GET /agents/stats/{project}` as chart data:
  `activityBars` and `tokenBars` for `app-bar-chart` (with `ACTIVITY_SERIES`,
  `TOKEN_SERIES`), `callDurationLines` and `runDurationLines` for a Chart.js
  line chart with `durationChartOptions()`, and `hasActivity`.
- `src/testing.ts` — the spec helpers (see Testing).

Global primitives in `styles.css`: `.section-tag`, `.wedge-list`,
`pre.snippet`, `.mono`, `.form-hint`, `.card-actions-bar`, `.panel-cut`,
`.text-muted`. Layout uses Tailwind utilities (`flex`, `grid`, `gap-*`,
`grid-cols-*`); there is no component framework besides this library.

## A page

- The element defaults in `styles.css` style headings, paragraphs, links,
  labels, inputs, selects and textareas everywhere; a page does not restyle
  them.
- It opens with `ui-page-header`; a detail page gives it a `backLink`. There
  are no breadcrumbs.
- Its sections are `ui-card`s; lists are `ui-table`s.
- It uses only these pieces, Tailwind layout utilities and its own
  stylesheet, whose colours, fonts and spacing come from the tokens — no hex
  literals.
- Icons are Font Awesome (`fa-solid fa-…`, `fa-regular`, `fa-brands`) only.
- Dates: `date: 'short'` in a table, `date: 'medium'` on a detail. Token counts
  through `number`. IDs in `.mono`, shortened to their first 8 characters with
  the full value in `title` and a `ui-copy-button` where copying is useful.

## Consistency rules

- **Errors.** A failed load shows `ui-alert` with a Retry button in
  `alert-actions`; a failed save shows `ui-alert` in the card that was saved.
  The text comes from `errorMessage(err, '…')`. No failure is only logged with
  `console.error`; no `alert()` or `confirm()`.
- **Success** is confirmed by `ToastService.success` ("Saved", "Copied"),
  which stacks in the bottom-right corner, or by the page changing.
- **Loading** is `ui-loading` for a view and `ui-spinner` inside a busy
  button, which is also disabled.
- **Empty** lists are `ui-empty-state` naming the next step.
- **Destructive** actions use a `danger` button and confirm in place (a second
  click) or in `ui-dialog` when what goes with them needs saying.
- **Buttons.** One `primary` per view; everything else is `secondary` or
  `ghost`.
- **Permissions** are reflected, not only enforced: a control the user may not
  use is hidden or disabled with the reason shown.

## Testing

- Every component, directive, service and the route table has a spec, added
  in the same change as the code.
- A page spec renders the template and finds elements by `data-testid` or
  role; it asserts what the user sees, not component fields.
- A data-loading view has four cases: loading, error with Retry, empty,
  loaded. A form has: invalid input refused, saving (button busy), server
  refusal shown, success.
- Roles are tested where a page hides or disables controls: `signedInAs('admin'
  | 'owner' | 'viewer')` from `src/testing.ts`, with `makeProject({ role })`.
- Fixtures come from `src/testing.ts` (`makeUser`, `makeProject`, `makeMember`,
  `makeSession`, `makeRun`, `makeTask`, `makeMessage`, `makeModelCall`, each
  taking overrides) and `settle(fixture)` lets a fixture finish its work.
- Services are tested with `HttpTestingController`, asserting URL, method and
  body.
- New and changed files reach 100% statement coverage
  (`npm test -- --watch=false --code-coverage`).

## Where a style belongs

A component's stylesheet is encapsulated by Angular, so it reaches the
component's own markup and its host, and nothing else. Three kinds of rule
cannot live there, and they are what `ui.css` carries:

- `button/button.css` and `menu/menu-item.css`, because `UiButton` and
  `UiMenuItem` are directives on the native `<button>` and `<a>` (so they keep
  their type, disabled state, form submission and `routerLink`), and an Angular
  directive cannot carry a stylesheet.
- `rhythm.css`, the space one surface leaves before the next
  (`ui-card + ui-card`), which belongs to neither of the two.

To retint a component from the surface it sits on, set a custom property
rather than writing a selector that reaches into it: properties inherit
through the encapsulation boundary, selectors do not. `ui-field` reads
`--ui-field-label-color` and `--ui-field-hint-color` (the login panel sets both
for its charcoal ground), and `ui-stats[inset]` hands `--ui-stat-ground` and
`--ui-stat-value-size` down to its tiles.

A component that styles content a page projects into it declares
`encapsulation: ViewEncapsulation.None` and says why in a comment above the
line: `ui-card`, `ui-page-header`, `ui-table`, `ui-toolbar` and `ui-nav-item`.
Their selectors stay scoped to a `ui-` element, class or attribute so nothing
leaks.
