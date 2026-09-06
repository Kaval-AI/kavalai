# Kaval.AI chatbot widget

The production chat widget: a framework-free port of the chatbot on the
kaval.ai homepage (an Angular component there), so one implementation can
serve the website, client demos and any page that can load a `<script>`
tag. The in-repo `webwidget/` remains the separate developer playground for
running workflows in the browser; this widget is the one meant for
end-user pages.

Two files, no build step, no dependencies:

```html
<link rel="stylesheet" href="kaval-chatbot.css" />
<script src="kaval-chatbot.js"></script>
<script>
  KavalChatbot.mount({
    connector: KavalChatbot.agentConnector({ url: "/api/chat/stream_agent" }),
    title: "Kaval.AI",
    logo: "roundlogo.png",
    suggestions: ["What is Kaval.AI?"],
  });
</script>
```

## Preview

```bash
python -m http.server        # from the repo root
# open http://localhost:8000/chatbotwidget/preview.html
```

`preview.html` mounts the widget twice — floating with the kaval.ai theme
and inline with a client-style theme — against a scripted connector, so
look and streaming behaviour can be checked without a backend.

## Connectors

A connector is `async (text, onPartial) => { text, choices }` (or an object
with that `send` plus an optional `reset`). The widget retries transient
failures with exponential backoff; throw `KavalChatbot.UserError` for
messages that should reach the visitor verbatim (rate limits,
maintenance), anything else renders as a generic apology.

`KavalChatbot.agentConnector(options)` speaks the kavalai agent-server SSE
protocol (`create_agent_router`): it POSTs to the stream endpoint,
extracts the reply from partial frames as they arrive, and keeps one
conversation per tab via a sessionStorage `external_id` (`reset()` starts
a fresh one). Options: `url`, `inputKey`/`replyKey`/`choicesKey` for
workflows whose data types use different field names, `storageKey`,
`fetchImpl` for tests.

Any other backend fits behind the same shape — including the
`webwidget/` WebLLM bridge for a fully in-browser bot:

```js
const bridge = KavalPlayground.workflowBridge();
KavalChatbot.mount({
  connector: async (text, onPartial) => {
    const out = await bridge.send(text);
    if (out && out.error) throw new Error(out.error);
    return { text: typeof out === "string" ? out : out.reply, choices: [] };
  },
});
```

## Options

| Option | Meaning |
|--------|---------|
| `connector` | Required — see above |
| `mode` | `"floating"` (bubble bottom-right, default) or `"inline"` (fills `target`) |
| `target` | Element or selector; required for inline mode |
| `title`, `logo` | Header text and optional round logo image URL |
| `greeting` | Invitation bubble next to the launcher; `null` disables |
| `emptyMessage` | Text shown before the first message |
| `placeholder` | Input placeholder |
| `suggestions` | Choice chips offered before the first message |
| `theme` | `--kcb-*` tokens, camelCased: `{accent, accentContent, surface, surfaceAlt, border, text, linkHover, radius, radiusSmall, font, fontHeader, fontMono, shadow, z, topOffset}` |

`mount` returns `{root, open, close, toggle, send, reset, destroy, setStatus,
setTheme}`: `setStatus(text)` shows a short line under the title (model
loading progress, "Ready", a fallback notice; empty hides it) and
`setTheme(theme)` swaps the skin wholesale at runtime.

Theming also works from CSS alone — every color, font and radius is a
`--kcb-*` custom property on `.kcb-chatbot`, defaulting to the kaval.ai
palette. Pages with a fixed header set `--kcb-top-offset` to its height so
a maximized window stops below it.

## Safety properties

- Markdown parses to data and renders through DOM nodes — never
  `innerHTML` — so agent output cannot introduce markup.
- Links render as anchors only for http(s), mailto, site-relative or
  fragment targets; any other scheme stays plain text.
- Errors other than `UserError` reach the visitor as a generic apology,
  never as internal detail.

## Tests

```bash
node --test chatbotwidget/tests/kaval-chatbot.test.js
```

Covers the DOM-free logic (markdown, SSE parsing, the agent connector);
the widget's look and DOM behaviour are checked manually via
`preview.html`.
