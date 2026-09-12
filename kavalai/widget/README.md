# Kaval.AI chat widget

The production chat widget: a framework-free port of the chatbot on the
kaval.ai homepage (an Angular component there), so one implementation can
serve the website, client demos and any page that can load a `<script>`
tag. The in-repo `webwidget/` remains the separate developer playground for
running workflows in the browser; this widget is the one meant for
end-user pages.

Two files, no build step, no dependencies. They ship inside the `kavalai`
wheel (`kavalai.widget.widget_dir()` returns their directory), so a Python
host serves the version whose connector speaks its agent server's protocol.
The full reference — every option, the `texts` keys and their defaults,
events, storage modes, feedback, tokens and the stable class list — is
`docs/reference/widget.rst`.

```html
<link rel="stylesheet" href="kaval-chatbot.css" />
<script src="kaval-chatbot.js"></script>
<script>
  const widget = KavalChatbot.mount({
    connector: KavalChatbot.agentConnector({ url: "/stream_agent" }),
    texts: { title: "Kaval.AI" },
    logo: "roundlogo.png",
    suggestions: ["What is Kaval.AI?"],
    onFeedback: (runId, vote, comment) => saveFeedback(runId, vote, comment),
  });
  widget.on("reply", (reply) => console.log(reply.runId));
</script>
```

## Preview

```bash
python -m http.server        # from the repo root
# open http://localhost:8000/kavalai/widget/preview.html
```

`preview.html` mounts the widget twice — floating with the kaval.ai theme
and a feedback comment box, and inline with a client-style theme and German
`texts` — against a scripted connector, so look and streaming behaviour can
be checked without a backend. `?theme=dark` switches both to a dark token
set, `?open=1` opens the floating one, and an event log shows every `on()`
event and `onFeedback` call.

## What the options cover

- **Connector** — `agentConnector({url, headers, onResponse, storage, …})`
  speaks the agent-server SSE protocol; `headers` may be a function (sync or
  async) resolved per request, and `onResponse(response, connector)` lets a
  host adopt a server-issued conversation id with `setConversationId`. Any
  `async (text, onPartial) => {text, choices, runId}` fits too.
- **Retries** — only a turn the server never started is sent again; once an
  SSE frame has arrived, a failure ends the turn with `texts.interrupted`.
- **Texts** — every visible string, aria labels and error messages
  included, is a `texts` key; `title`, `greeting`, `emptyMessage` and
  `placeholder` remain as top-level shorthands.
- **Disclosure** — an "AI assistant" label in the header, on by default;
  `disclosure: false` removes it.
- **Storage** — the conversation id lives in `"session"` (default),
  `"local"` or `"none"` (memory only).
- **Feedback** — thumbs (and, with `feedbackComment`, a comment box) under
  each answer that has a run id, only when `onFeedback` is given.
- **Events** — `widget.on("open" | "close" | "reply" | "error" | "resize" |
  "feedback", cb)` returns an unsubscribe function.
- **Theming** — every colour, font, size and radius is a `--kcb-*` token.

## Safety properties

- Markdown parses to data and renders through DOM nodes — never
  `innerHTML` — so agent output cannot introduce markup.
- Links render as anchors only for http(s), mailto, site-relative or
  fragment targets; any other scheme stays plain text.
- Errors other than `UserError` reach the visitor as `texts.error`, never
  as internal detail.

## Tests

```bash
node --test kavalai/widget/tests/kaval-chatbot.test.js
```

Covers markdown, SSE parsing and the agent connector, and — over the
minimal DOM in `tests/fake-dom.js` — `mount()`: texts, the disclosure
label, feedback, events, storage and the retry rule. It also checks that
`docs/reference/widget.rst` documents every `texts` key, event, token and
class it names. The look is checked in `preview.html`.
