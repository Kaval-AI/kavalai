/*
Copyright 2026 OÜ KAVAL AI (registry code 17393877)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Kaval.AI production chat widget: a framework-free port of the chatbot on
kaval.ai (an Angular component there), so one implementation can serve the
website, client demos and any page that can load a <script> tag.

  KavalChatbot.mount({
    connector: KavalChatbot.agentConnector({ url: "/api/chat/stream_agent" }),
    mode: "floating",             // or "inline" with target: el-or-selector
    title: "Chatbot",
    theme: { accent: "#acc12f" }, // any --kcb-* token, camelCased
  });

A connector is `async (text, onPartial) => { text, choices }`; the bundled
`agentConnector` speaks the kavalai agent-server SSE protocol, and any other
backend (the webwidget's WebLLM bridge included) fits behind the same shape.

Markdown parses to data and renders through DOM nodes — never innerHTML — so
agent output cannot introduce markup. Replies stream, so every input is
potentially truncated and renders as the block it is becoming.
*/
(function (root, factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory();
  } else {
    root.KavalChatbot = factory();
  }
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  /* ------------------------------------------------------------------ *
   * Markdown (ported from the website's markdown.ts)                    *
   * ------------------------------------------------------------------ */

  const FENCE = /^\s*(?:```|~~~)\s*([\w+#.-]*)\s*$/;
  const FENCE_END = /^\s*(?:```|~~~)\s*$/;
  const HEADING = /^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$/;
  const RULE = /^\s*([-*_])\s*(?:\1\s*){2,}$/;
  const LIST_ITEM = /^\s*(?:([-*+])|(\d{1,9})[.)])\s+(.*)$/;

  const INLINE_CODE = /`([^`\n]+)`/;
  const LINK = /\[([^\]\n]*)\]\(\s*<?([^)\s>]*)>?(?:\s+"[^"\n]*")?\s*\)/;
  /* Bare autolinks. Trailing punctuation is trimmed by `trimUrlTail` so a URL
     at the end of a sentence does not swallow the full stop. */
  const BARE_URL = /(?:https?:\/\/|www\.)[^\s<>[\]{}"'`]+/;
  const STRONG = /\*\*([^\n]+?)\*\*|__([^\n]+?)__/;
  /* Only `*` for italics: `_` would mangle snake_case identifiers, which are
     everywhere in answers about a Python SDK. */
  const EM = /\*([^\s*](?:[^*\n]*?[^\s*])?)\*/;

  /* The chat model sometimes emits its newlines double-escaped, so the decoded
     reply contains a literal backslash-n instead of a line break. Undo that,
     but only when the text has no real newlines at all — otherwise a genuine
     `"\n"` inside a Python snippet would be rewritten. */
  function normalize(raw) {
    const text = raw.replace(/\r\n?/g, "\n");
    if (text.includes("\n") || !text.includes("\\n")) {
      return text;
    }
    return text.replace(/\\n/g, "\n").replace(/\\t/g, "\t");
  }

  /* Links are rendered as anchors, so the scheme is the security boundary:
     anything but http(s), mailto, a site-relative path or a fragment is
     dropped and the link renders as plain text. */
  function safeHref(url) {
    const trimmed = url.trim();
    if (/^(https?:\/\/|mailto:)/i.test(trimmed)) {
      return trimmed;
    }
    if (/^www\./i.test(trimmed)) {
      return "https://" + trimmed;
    }
    if (/^[/#][^/\\]/.test(trimmed) || trimmed === "/") {
      return trimmed;
    }
    return null;
  }

  function trimUrlTail(url) {
    let end = url.length;
    while (end > 0) {
      const char = url[end - 1];
      if (".,;:!?".includes(char)) {
        end--;
        continue;
      }
      /* Keep a closing paren only if the URL opened one (a Wikipedia link). */
      if (char === ")") {
        const slice = url.slice(0, end);
        if (
          (slice.match(/\(/g) || []).length >= (slice.match(/\)/g) || []).length
        ) {
          break;
        }
        end--;
        continue;
      }
      break;
    }
    return url.slice(0, end);
  }

  /* `build` returns the segments the match produces, or null to decline it (an
     unsafe link is left as plain text). It may also shorten `length` when it
     consumes less than the pattern matched, as bare URLs do when a sentence's
     full stop is trimmed off the end. */
  function candidate(pattern, text, build) {
    const match = pattern.exec(text);
    if (!match) {
      return null;
    }
    const built = build(match);
    return built === null
      ? null
      : {
          index: match.index,
          length: built.length !== undefined ? built.length : match[0].length,
          segments: built.segments,
        };
  }

  function parseInline(text, style) {
    style = style || {};
    const out = [];
    let rest = text;

    const push = (segment) => {
      if (segment.text) {
        out.push(segment);
      }
    };

    while (rest) {
      /* Earliest match wins; ties break in listed order, which is why code
         comes first — backticks suppress the markup inside them. */
      const candidates = [
        candidate(INLINE_CODE, rest, (m) => ({
          segments: [Object.assign({ text: m[1], code: true }, style)],
        })),
        candidate(LINK, rest, (m) => {
          const href = safeHref(m[2]);
          return href === null
            ? null
            : { segments: [Object.assign({ text: m[1] || m[2], href: href }, style)] };
        }),
        candidate(BARE_URL, rest, (m) => {
          const url = trimUrlTail(m[0]);
          const href = safeHref(url);
          return href === null
            ? null
            : {
                segments: [Object.assign({ text: url, href: href }, style)],
                length: url.length,
              };
        }),
        candidate(STRONG, rest, (m) => ({
          segments: parseInline(
            m[1] !== undefined ? m[1] : m[2],
            Object.assign({}, style, { strong: true })
          ),
        })),
        candidate(EM, rest, (m) => ({
          segments: parseInline(m[1], Object.assign({}, style, { em: true })),
        })),
      ].filter((found) => found !== null);

      if (candidates.length === 0) {
        break;
      }

      const next = candidates.reduce((best, found) =>
        found.index < best.index ? found : best
      );
      if (next.length <= 0) {
        break; // a pattern that consumes nothing would spin forever
      }
      push(Object.assign({ text: rest.slice(0, next.index) }, style));
      out.push.apply(out, next.segments);
      rest = rest.slice(next.index + next.length);
    }

    push(Object.assign({ text: rest }, style));
    return out;
  }

  function parseMarkdown(raw) {
    const lines = normalize(raw).split("\n");
    const blocks = [];
    let i = 0;

    while (i < lines.length) {
      const line = lines[i];

      if (!line.trim()) {
        i++;
        continue;
      }

      const fence = FENCE.exec(line);
      if (fence) {
        const body = [];
        i++;
        while (i < lines.length && !FENCE_END.test(lines[i])) {
          body.push(lines[i]);
          i++;
        }
        i++; // the closing fence, if the reply has streamed that far
        blocks.push({ kind: "code", language: fence[1], text: body.join("\n") });
        continue;
      }

      const heading = HEADING.exec(line);
      if (heading) {
        blocks.push({
          kind: "heading",
          level: heading[1].length,
          segments: parseInline(heading[2]),
        });
        i++;
        continue;
      }

      if (RULE.test(line)) {
        blocks.push({ kind: "rule" });
        i++;
        continue;
      }

      const item = LIST_ITEM.exec(line);
      if (item) {
        const ordered = item[2] !== undefined;
        const items = [];
        while (i < lines.length) {
          const next = LIST_ITEM.exec(lines[i]);
          if (next && (next[2] !== undefined) === ordered) {
            items.push(next[3]);
            i++;
            continue;
          }
          /* An indented, non-blank line continues the item above (lazy
             continuation); anything else ends the list. */
          if (
            items.length > 0 &&
            /^\s+\S/.test(lines[i]) &&
            !FENCE.test(lines[i])
          ) {
            items[items.length - 1] += "\n" + lines[i].trim();
            i++;
            continue;
          }
          break;
        }
        blocks.push({
          kind: "list",
          ordered: ordered,
          items: items.map((text) => parseInline(text)),
        });
        continue;
      }

      const paragraph = [];
      while (i < lines.length && lines[i].trim()) {
        const current = lines[i];
        if (FENCE.test(current) || HEADING.test(current) || RULE.test(current)) {
          break;
        }
        /* A list may start straight after a paragraph line without a blank
           line between them, which is how the model usually writes them. */
        if (paragraph.length > 0 && LIST_ITEM.test(current)) {
          break;
        }
        paragraph.push(current.trim());
        i++;
      }
      blocks.push({ kind: "paragraph", segments: parseInline(paragraph.join("\n")) });
    }

    return blocks;
  }

  /* ------------------------------------------------------------------ *
   * Agent-server SSE protocol (ported from the website's service)       *
   * ------------------------------------------------------------------ */

  /* An error whose message is safe and useful to show the visitor. Everything
     else surfaces as a generic apology so internal detail never reaches the
     UI. */
  class UserError extends Error {}

  /* The server streams a JSON object, so a partial frame is a truncated JSON
     document. Pull the value of a string key out of it so the reply can be
     rendered as it arrives, tolerating the closing quote being absent. Frames
     for auxiliary streams (thoughts, instructions) do not contain the key at
     all and are skipped by returning null. */
  function extractPartialString(json, key) {
    const marker = '"' + key + '"';
    const keyAt = json.indexOf(marker);
    if (keyAt === -1) {
      return null;
    }
    const openQuote = json.indexOf('"', json.indexOf(":", keyAt + marker.length) + 1);
    if (openQuote === -1) {
      return null;
    }

    let out = "";
    for (let i = openQuote + 1; i < json.length; i++) {
      const char = json[i];
      if (char === "\\") {
        const escaped = json[i + 1];
        if (escaped === undefined) {
          break; // truncated mid-escape; emit what we have
        }
        const simple = {
          n: "\n",
          t: "\t",
          r: "\r",
          b: "\b",
          f: "\f",
          '"': '"',
          "\\": "\\",
          "/": "/",
        };
        if (escaped === "u") {
          const hex = json.slice(i + 2, i + 6);
          if (hex.length < 4) {
            break;
          }
          out += String.fromCharCode(parseInt(hex, 16));
          i += 5;
        } else {
          out += simple[escaped] !== undefined ? simple[escaped] : escaped;
          i += 1;
        }
        continue;
      }
      if (char === '"') {
        return out; // closing quote: the value is complete
      }
      out += char;
    }
    return out;
  }

  /* Split a raw SSE buffer into complete frames, returning any trailing
     partial frame so it can be prepended to the next chunk. Comment frames
     (`: ping`) are dropped. */
  function parseSseFrames(buffer) {
    const blocks = buffer.split("\n\n");
    const rest = blocks.pop() || "";
    const events = [];

    for (const block of blocks) {
      const trimmed = block.trim();
      if (!trimmed || trimmed.startsWith(":")) {
        continue;
      }
      const dataLine = trimmed
        .split("\n")
        .find((line) => line.startsWith("data: "));
      if (!dataLine) {
        continue;
      }
      try {
        events.push(JSON.parse(dataLine.slice("data: ".length)));
      } catch (_error) {
        /* A frame we cannot parse is not worth failing the whole reply over. */
      }
    }

    return { events: events, rest: rest };
  }

  /* Conversation ids survive within the tab where storage exists; where it
     does not (tests, privacy modes that throw on access) an in-memory id
     still keeps one page-load's messages in one conversation. */
  function conversationStore(storageKey) {
    let memory = null;
    return function () {
      try {
        let id = sessionStorage.getItem(storageKey);
        if (!id) {
          id = crypto.randomUUID();
          sessionStorage.setItem(storageKey, id);
        }
        return id;
      } catch (_error) {
        if (!memory) {
          memory = crypto.randomUUID();
        }
        return memory;
      }
    };
  }

  /* A connector for the kavalai agent server (`create_agent_router`): posts
     the message and consumes the SSE response. `EventSource` cannot be used
     here — it cannot issue a POST body.

     Options: `url` (the stream endpoint), `inputKey`/`replyKey`/`choicesKey`
     for workflows whose data types name things differently, `storageKey` for
     the sessionStorage slot of the conversation id, `fetchImpl` for tests.

     The returned object has `send(text, onPartial) -> {text, choices}` and
     `reset()`, which starts a fresh conversation. */
  function agentConnector(options) {
    options = options || {};
    const url = options.url || "/api/chat/stream_agent";
    const inputKey = options.inputKey || "message";
    const replyKey = options.replyKey || "agent_response";
    const choicesKey = options.choicesKey || "choices";
    const storageKey = options.storageKey || "kavalai-chat-conversation";
    const fetchImpl = options.fetchImpl || fetch.bind(globalThis);
    let currentId = conversationStore(storageKey);

    async function send(text, onPartial) {
      const data = {};
      data[inputKey] = text;
      const response = await fetchImpl(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ external_id: currentId(), data: data }),
      });

      if (response.status === 429) {
        throw new UserError("Too many messages. Please wait a moment and try again.");
      }
      if (response.status === 503) {
        throw new UserError(
          "The chat assistant is unavailable right now. Please try again later."
        );
      }
      if (!response.ok || !response.body) {
        throw new Error("Chat request failed with status " + response.status);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let reply = null;

      while (true) {
        const chunk = await reader.read();
        if (chunk.done) {
          break;
        }
        buffer += decoder.decode(chunk.value, { stream: true });

        const parsed = parseSseFrames(buffer);
        buffer = parsed.rest;

        for (const event of parsed.events) {
          switch (event.type) {
            case "restart":
              /* The LLM call was retried; everything streamed so far for this
                 run will be re-sent, so drop it. */
              if (onPartial) onPartial("");
              break;
            case "partial":
            case "complete": {
              const partial = extractPartialString(event.value || "", replyKey);
              if (partial !== null && onPartial) {
                onPartial(partial);
              }
              break;
            }
            case "workflow_completed":
              reply = event.output_data || null;
              break;
            case "workflow_failed":
              throw new Error(event.value || "The chat agent failed to respond.");
          }
        }
      }

      if (!reply) {
        throw new Error("The chat agent did not return a reply.");
      }
      return { text: reply[replyKey], choices: reply[choicesKey] || [] };
    }

    function reset() {
      try {
        sessionStorage.removeItem(storageKey);
      } catch (_error) {
        /* memory-backed ids rotate below */
      }
      currentId = conversationStore(storageKey);
    }

    return { send: send, reset: reset };
  }

  /* ------------------------------------------------------------------ *
   * Widget                                                              *
   * ------------------------------------------------------------------ */

  const MIN_WIDTH = 320;
  const MIN_HEIGHT = 360;
  const VIEWPORT_MARGIN = 48;
  /* Vertical space below the window that it cannot use: the bottom offset of
     the widget, the gap and the float bubble stacked beneath it. */
  const RESERVED_HEIGHT = 24 + 12 + 56;
  /* Maximizing is primarily vertical, but a full-height sliver is a poor
     place to read code in, so a narrow window widens too. */
  const MAXIMIZED_MIN_WIDTH = 520;
  /* Slightly longer than the 0.25s close CSS animation */
  const CLOSE_ANIMATION_MS = 300;
  /* A failed request is retried before the visitor is told anything went
     wrong: a dropped stream or an instance recycling under the agent endpoint
     is transient, and the visitor cannot do anything useful with either.
     Delays double so a restarting instance gets time to come back. */
  const MAX_RETRIES = 3;
  const RETRY_BASE_DELAY_MS = 500;

  const ICONS = {
    close:
      "M6 6 L18 18 M18 6 L6 18",
    expand:
      "M9 4H4v5 M15 4h5v5 M9 20H4v-5 M15 20h5v-5",
    compress:
      "M4 9h5V4 M20 9h-5V4 M4 15h5v5 M20 15h-5v5",
    chat:
      "M4 5h16v11H8l-4 4z",
  };

  function svgIcon(name) {
    const ns = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(ns, "svg");
    svg.setAttribute("viewBox", "0 0 24 24");
    svg.setAttribute("aria-hidden", "true");
    svg.classList.add("kcb-icon");
    const path = document.createElementNS(ns, "path");
    path.setAttribute("d", ICONS[name]);
    path.setAttribute("fill", "none");
    path.setAttribute("stroke", "currentColor");
    path.setAttribute("stroke-width", "2");
    path.setAttribute("stroke-linecap", "round");
    path.setAttribute("stroke-linejoin", "round");
    svg.appendChild(path);
    return svg;
  }

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) {
      node.className = className;
    }
    if (text !== undefined) {
      node.textContent = text;
    }
    return node;
  }

  /* Render inline segments into `parent`. Everything is `textContent`, so
     agent text can never become markup. */
  function renderSegments(parent, segments) {
    for (const segment of segments) {
      let node;
      if (segment.href) {
        node = el("a", "kcb-md-link", segment.text);
        node.href = segment.href;
        node.target = "_blank";
        node.rel = "noopener noreferrer";
      } else if (segment.code) {
        node = el("code", "kcb-inline-code", segment.text);
      } else {
        node = el("span", null, segment.text);
      }
      if (segment.strong) {
        node.classList.add("kcb-md-strong");
      }
      if (segment.em) {
        node.classList.add("kcb-md-em");
      }
      parent.appendChild(node);
    }
  }

  function renderMarkdown(container, text) {
    container.replaceChildren();
    for (const block of parseMarkdown(text)) {
      switch (block.kind) {
        case "code": {
          const pre = el("pre", "kcb-code");
          pre.appendChild(el("code", null, block.text));
          container.appendChild(pre);
          break;
        }
        case "heading": {
          const heading = el("p", "kcb-md-heading");
          heading.dataset.level = String(block.level);
          renderSegments(heading, block.segments);
          container.appendChild(heading);
          break;
        }
        case "list": {
          const list = el(block.ordered ? "ol" : "ul", "kcb-md-list");
          for (const item of block.items) {
            const li = el("li");
            renderSegments(li, item);
            list.appendChild(li);
          }
          container.appendChild(list);
          break;
        }
        case "rule":
          container.appendChild(el("hr", "kcb-md-rule"));
          break;
        default: {
          const paragraph = el("p", "kcb-md-paragraph");
          renderSegments(paragraph, block.segments);
          container.appendChild(paragraph);
          break;
        }
      }
    }
  }

  function applyTheme(rootEl, theme) {
    for (const key of Object.keys(theme || {})) {
      const token = "--kcb-" + key.replace(/[A-Z]/g, (c) => "-" + c.toLowerCase());
      rootEl.style.setProperty(token, String(theme[key]));
    }
  }

  /* Mount the widget. Options:
       connector  (required) `{send}` or a bare send function
       mode       "floating" (default) or "inline"
       target     element or selector; body for floating, required for inline
       title, logo, greeting, emptyMessage, placeholder, suggestions
       theme      {accent, surface, radius, ...} → --kcb-* custom properties
     Returns {root, open, close, toggle, send, reset, destroy, setStatus,
     setTheme}. */
  function mount(options) {
    options = options || {};
    const connector =
      typeof options.connector === "function"
        ? { send: options.connector }
        : options.connector;
    if (!connector || typeof connector.send !== "function") {
      throw new Error("KavalChatbot.mount needs a connector with a send function");
    }
    const mode = options.mode || "floating";
    const doc = document;
    let target =
      typeof options.target === "string"
        ? doc.querySelector(options.target)
        : options.target;
    if (!target) {
      if (mode === "inline") {
        throw new Error("inline mode needs a target element");
      }
      target = doc.body;
    }
    const texts = {
      title: options.title !== undefined ? options.title : "Chatbot",
      greeting:
        options.greeting !== undefined ? options.greeting : "Hello! Click here for help.",
      empty:
        options.emptyMessage !== undefined
          ? options.emptyMessage
          : "Hi! Ask us anything.",
      placeholder:
        options.placeholder !== undefined ? options.placeholder : "Type a message...",
    };

    const state = {
      isOpen: mode === "inline",
      isSending: false,
      isMaximized: false,
      width: 340,
      height: 480,
      restore: { width: 340, height: 480 },
      suggestions: options.suggestions || [],
      destroyed: false,
    };

    /* --- static skeleton ------------------------------------------- */

    const rootEl = el("div", "kcb-chatbot kcb-" + mode);
    applyTheme(rootEl, options.theme);

    const windowEl = el("div", "kcb-window");
    windowEl.hidden = mode === "floating";

    for (const edge of ["left", "top", "corner"]) {
      const handle = el("div", "kcb-resize kcb-resize-" + edge);
      handle.addEventListener("pointerdown", (event) => startResize(event, edge));
      windowEl.appendChild(handle);
    }

    const header = el("div", "kcb-header");
    const titleEl = el("span", "kcb-title");
    if (options.logo) {
      const logo = el("img", "kcb-title-logo");
      logo.src = options.logo;
      logo.alt = "";
      titleEl.appendChild(logo);
    }
    const titleText = el("div", "kcb-title-text");
    titleText.appendChild(el("span", "kcb-title-label", texts.title));
    const statusEl = el("span", "kcb-status");
    statusEl.hidden = true;
    titleText.appendChild(statusEl);
    titleEl.appendChild(titleText);
    const actions = el("span", "kcb-header-actions");
    const maximizeBtn = el("button", "kcb-icon-btn kcb-maximize");
    maximizeBtn.type = "button";
    maximizeBtn.setAttribute("aria-label", "Maximize chat");
    maximizeBtn.appendChild(svgIcon("expand"));
    maximizeBtn.addEventListener("click", toggleMaximize);
    const headerCloseBtn = el("button", "kcb-bubble kcb-bubble-inline");
    headerCloseBtn.type = "button";
    headerCloseBtn.setAttribute("aria-label", "Close chat");
    headerCloseBtn.appendChild(svgIcon("close"));
    headerCloseBtn.addEventListener("click", toggle);
    actions.appendChild(maximizeBtn);
    if (mode === "floating") {
      actions.appendChild(headerCloseBtn);
    }
    header.appendChild(titleEl);
    header.appendChild(actions);
    windowEl.appendChild(header);

    const historyEl = el("div", "kcb-history");
    const emptyEl = el("p", "kcb-empty", texts.empty);
    historyEl.appendChild(emptyEl);
    const typingEl = el("div", "kcb-message kcb-typing");
    typingEl.setAttribute("role", "status");
    typingEl.setAttribute("aria-label", "Assistant is typing");
    for (let i = 0; i < 3; i++) {
      typingEl.appendChild(el("span", "kcb-dot"));
    }
    typingEl.hidden = true;
    historyEl.appendChild(typingEl);
    windowEl.appendChild(historyEl);

    const choicesEl = el("div", "kcb-choices");
    windowEl.appendChild(choicesEl);

    const form = el("form", "kcb-input-row");
    const input = el("input");
    input.type = "text";
    input.name = "draft";
    input.placeholder = texts.placeholder;
    input.autocomplete = "off";
    const sendBtn = el("button", "kcb-send", "Send");
    sendBtn.type = "submit";
    sendBtn.disabled = true;
    form.appendChild(input);
    form.appendChild(sendBtn);
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      send(input.value);
    });
    input.addEventListener("input", refreshControls);
    windowEl.appendChild(form);

    rootEl.appendChild(windowEl);

    let greetingEl = null;
    let bubbleEl = null;
    if (mode === "floating") {
      if (texts.greeting) {
        greetingEl = el("button", "kcb-greeting", texts.greeting);
        greetingEl.type = "button";
        greetingEl.addEventListener("click", toggle);
        rootEl.appendChild(greetingEl);
      }
      bubbleEl = el("button", "kcb-bubble kcb-bubble-float");
      bubbleEl.type = "button";
      bubbleEl.setAttribute("aria-label", "Open chat");
      if (options.logo) {
        const logo = el("img", "kcb-bubble-logo");
        logo.src = options.logo;
        logo.alt = "";
        bubbleEl.appendChild(logo);
      } else {
        const glyph = el("span", "kcb-bubble-glyph");
        glyph.appendChild(svgIcon("chat"));
        bubbleEl.appendChild(glyph);
      }
      const closeIcon = el("span", "kcb-close-icon");
      closeIcon.appendChild(svgIcon("close"));
      bubbleEl.appendChild(closeIcon);
      bubbleEl.addEventListener("click", toggle);
      rootEl.appendChild(bubbleEl);
    }

    target.appendChild(rootEl);
    renderChoices(state.suggestions);
    applySize();
    window.addEventListener("resize", onViewportResize);

    /* --- behaviour -------------------------------------------------- */

    function maxWidth() {
      return Math.max(window.innerWidth - VIEWPORT_MARGIN, MIN_WIDTH);
    }

    /* Pages with a fixed header set `--kcb-top-offset` so a maximized window
       stops under it instead of covering the navigation. */
    function maxHeight() {
      const declared = getComputedStyle(rootEl).getPropertyValue("--kcb-top-offset");
      const topOffset = parseFloat(declared) || 0;
      return Math.max(
        window.innerHeight - RESERVED_HEIGHT - topOffset,
        MIN_HEIGHT
      );
    }

    function applySize() {
      if (mode !== "floating") {
        return;
      }
      windowEl.style.width = state.width + "px";
      windowEl.style.height = state.height + "px";
    }

    function fitToViewport() {
      state.width = Math.min(state.width, maxWidth());
      state.height = Math.min(state.height, maxHeight());
      applySize();
    }

    function applyMaximizedSize() {
      state.width = Math.min(Math.max(state.width, MAXIMIZED_MIN_WIDTH), maxWidth());
      state.height = maxHeight();
      applySize();
    }

    function toggleMaximize() {
      if (state.isMaximized) {
        state.isMaximized = false;
        state.width = Math.min(state.restore.width, maxWidth());
        state.height = Math.min(state.restore.height, maxHeight());
        applySize();
      } else {
        state.restore = { width: state.width, height: state.height };
        state.isMaximized = true;
        applyMaximizedSize();
      }
      rootEl.classList.toggle("kcb-maximized", state.isMaximized);
      maximizeBtn.setAttribute(
        "aria-label",
        state.isMaximized ? "Restore chat size" : "Maximize chat"
      );
      maximizeBtn.setAttribute("aria-pressed", String(state.isMaximized));
      maximizeBtn.replaceChildren(svgIcon(state.isMaximized ? "compress" : "expand"));
    }

    /* A maximized window follows the viewport; a manually sized one is only
       shrunk when it no longer fits. */
    function onViewportResize() {
      if (!state.isOpen) {
        return;
      }
      if (state.isMaximized) {
        applyMaximizedSize();
      } else {
        fitToViewport();
      }
    }

    /* Desktop only: the window is anchored bottom-right, so dragging the
       left/top edges (or the corner) grows it leftwards/upwards. */
    function startResize(event, edge) {
      event.preventDefault();
      state.isMaximized = false;
      rootEl.classList.remove("kcb-maximized");
      const handle = event.target;
      const startX = event.clientX;
      const startY = event.clientY;
      const startWidth = state.width;
      const startHeight = state.height;

      try {
        handle.setPointerCapture(event.pointerId);
      } catch (_error) {
        /* synthetic events in tests have no active pointer */
      }

      const onMove = (e) => {
        if (edge !== "top") {
          state.width = Math.min(
            Math.max(startWidth + (startX - e.clientX), MIN_WIDTH),
            maxWidth()
          );
        }
        if (edge !== "left") {
          state.height = Math.min(
            Math.max(startHeight + (startY - e.clientY), MIN_HEIGHT),
            maxHeight()
          );
        }
        applySize();
      };

      const onUp = (e) => {
        try {
          handle.releasePointerCapture(e.pointerId);
        } catch (_error) {
          /* see above */
        }
        handle.removeEventListener("pointermove", onMove);
        handle.removeEventListener("pointerup", onUp);
        handle.removeEventListener("pointercancel", onUp);
      };

      handle.addEventListener("pointermove", onMove);
      handle.addEventListener("pointerup", onUp);
      handle.addEventListener("pointercancel", onUp);
    }

    function open() {
      if (state.isOpen || mode === "inline") {
        state.isOpen = true;
        return;
      }
      state.isOpen = true;
      if (greetingEl) {
        /* The invitation has done its job as soon as the chat is opened. */
        greetingEl.remove();
        greetingEl = null;
      }
      fitToViewport();
      windowEl.hidden = false;
      windowEl.classList.remove("kcb-window-close");
      windowEl.classList.add("kcb-window-opening");
      rootEl.classList.add("kcb-open");
      if (bubbleEl) {
        bubbleEl.setAttribute("aria-label", "Close chat");
      }
      input.focus();
    }

    function close() {
      if (!state.isOpen || mode === "inline") {
        return;
      }
      state.isOpen = false;
      rootEl.classList.remove("kcb-open");
      windowEl.classList.remove("kcb-window-opening");
      windowEl.classList.add("kcb-window-close");
      if (bubbleEl) {
        bubbleEl.setAttribute("aria-label", "Open chat");
      }
      setTimeout(() => {
        if (!state.isOpen) {
          windowEl.hidden = true;
        }
      }, CLOSE_ANIMATION_MS);
    }

    function toggle() {
      if (state.isOpen) {
        close();
      } else {
        open();
      }
    }

    function refreshControls() {
      sendBtn.disabled = state.isSending || !input.value.trim();
      for (const chip of choicesEl.children) {
        chip.disabled = state.isSending;
      }
    }

    function scrollHistory() {
      historyEl.scrollTop = historyEl.scrollHeight;
    }

    function renderChoices(choices) {
      choicesEl.replaceChildren();
      for (const choice of choices) {
        const chip = el("button", "kcb-choice", choice);
        chip.type = "button";
        chip.addEventListener("click", () => {
          if (!state.isSending) {
            send(choice);
          }
        });
        choicesEl.appendChild(chip);
      }
    }

    function addUserMessage(text) {
      const bubble = el("div", "kcb-message kcb-from-user", text);
      historyEl.insertBefore(bubble, typingEl);
    }

    function addAgentMessage() {
      const bubble = el("div", "kcb-message");
      /* An agent entry starts empty and fills as the reply streams in; the
         typing indicator stands in for it until then, so an empty bubble is
         never shown. */
      bubble.hidden = true;
      historyEl.insertBefore(bubble, typingEl);
      return bubble;
    }

    function setAgentText(bubble, text) {
      bubble.hidden = !text;
      typingEl.hidden = !state.isSending || Boolean(text);
      renderMarkdown(bubble, text);
      scrollHistory();
    }

    function delay(ms) {
      return new Promise((resolve) => setTimeout(resolve, ms));
    }

    /* Sends the message, retrying a failed attempt with exponential backoff.
       A UserError is terminal: rate limiting and "unavailable" are deliberate
       answers from the backend, and retrying would only walk further into the
       per-IP limiter. The typing dots stay up throughout, so a retried
       request looks to the visitor like a slow one. */
    async function request(text, bubble) {
      for (let attempt = 0; ; attempt++) {
        try {
          return await connector.send(text, (partial) => setAgentText(bubble, partial));
        } catch (error) {
          if (error instanceof UserError || attempt >= MAX_RETRIES) {
            throw error;
          }
          /* Whatever streamed before the failure belongs to a reply that will
             never finish. Dropping it brings the dots back and keeps the next
             attempt's partials from being read as a continuation. */
          setAgentText(bubble, "");
          await delay(RETRY_BASE_DELAY_MS * Math.pow(2, attempt));
        }
      }
    }

    async function send(text) {
      text = (text || "").trim();
      if (!text || state.isSending || state.destroyed) {
        return;
      }
      input.value = "";
      emptyEl.hidden = true;
      addUserMessage(text);
      renderChoices([]);
      state.isSending = true;
      typingEl.hidden = false;
      refreshControls();
      scrollHistory();

      const bubble = addAgentMessage();
      try {
        const reply = await request(text, bubble);
        setAgentText(bubble, reply.text);
        renderChoices(reply.choices || []);
      } catch (error) {
        setAgentText(
          bubble,
          error instanceof UserError
            ? error.message
            : "Something went wrong. Please try again."
        );
      } finally {
        state.isSending = false;
        typingEl.hidden = true;
        refreshControls();
        scrollHistory();
      }
    }

    function reset() {
      for (const node of Array.from(historyEl.children)) {
        if (node !== emptyEl && node !== typingEl) {
          node.remove();
        }
      }
      emptyEl.hidden = false;
      renderChoices(state.suggestions);
      if (typeof connector.reset === "function") {
        connector.reset();
      }
    }

    function destroy() {
      state.destroyed = true;
      window.removeEventListener("resize", onViewportResize);
      rootEl.remove();
    }

    /* A short line under the title — model loading progress, "Ready",
       a fallback notice; empty text hides it. */
    function setStatus(text) {
      statusEl.textContent = text || "";
      statusEl.hidden = !text;
    }

    /* Replace the theme wholesale: tokens the new theme does not name fall
       back to the stylesheet defaults instead of lingering from the old one. */
    function setTheme(theme) {
      for (const name of Array.from(rootEl.style)) {
        if (name.startsWith("--kcb-")) {
          rootEl.style.removeProperty(name);
        }
      }
      applyTheme(rootEl, theme);
    }

    return {
      root: rootEl,
      open: open,
      close: close,
      toggle: toggle,
      send: send,
      reset: reset,
      destroy: destroy,
      setStatus: setStatus,
      setTheme: setTheme,
    };
  }

  return {
    mount: mount,
    agentConnector: agentConnector,
    UserError: UserError,
    parseMarkdown: parseMarkdown,
    parseInline: parseInline,
    safeHref: safeHref,
    extractPartialString: extractPartialString,
    parseSseFrames: parseSseFrames,
  };
});
