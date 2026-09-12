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

Run with: node --test kavalai/widget/tests/kaval-chatbot.test.js
Covers markdown parsing, SSE frame handling, the agent-server connector and,
over the fake DOM beside this file, mount(): texts, the disclosure label,
feedback, events and the retry rule. The widget's look is checked in
preview.html.
*/

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const K = require("../kaval-chatbot.js");
const { installDom, isShown } = require("./fake-dom.js");

const DOCS = path.resolve(__dirname, "../../../docs/reference/widget.rst");
const CSS = path.resolve(__dirname, "../kaval-chatbot.css");
const JS = path.resolve(__dirname, "../kaval-chatbot.js");

test("markdown: headings, lists, code and rules parse to blocks", () => {
  const blocks = K.parseMarkdown(
    "# Title\n\nSome *text*.\n\n- one\n- two\n\n1. first\n2. second\n\n---\n\n```py\nprint(1)\n```"
  );
  assert.deepEqual(
    blocks.map((b) => b.kind),
    ["heading", "paragraph", "list", "list", "rule", "code"]
  );
  assert.equal(blocks[0].level, 1);
  assert.equal(blocks[2].ordered, false);
  assert.equal(blocks[3].ordered, true);
  assert.equal(blocks[5].language, "py");
  assert.equal(blocks[5].text, "print(1)");
});

test("markdown: an unclosed fence renders as the code block it is becoming", () => {
  const blocks = K.parseMarkdown("```python\nx = 1\ny = 2");
  assert.deepEqual(blocks, [{ kind: "code", language: "python", text: "x = 1\ny = 2" }]);
});

test("markdown: backticks suppress the markup inside them", () => {
  const [block] = K.parseMarkdown("use `**not bold**` here");
  const code = block.segments.find((s) => s.code);
  assert.equal(code.text, "**not bold**");
  assert.ok(!code.strong);
});

test("markdown: unsafe link schemes render as plain text", () => {
  const [block] = K.parseMarkdown("[click](javascript:alert(1))");
  assert.ok(block.segments.every((s) => !s.href));
});

test("markdown: bare urls keep their meaning, not the sentence's full stop", () => {
  const [block] = K.parseMarkdown("see https://kaval.ai/docs.");
  const link = block.segments.find((s) => s.href);
  assert.equal(link.href, "https://kaval.ai/docs");
  assert.equal(block.segments[block.segments.length - 1].text, ".");
});

test("markdown: bold and italic nest and snake_case survives", () => {
  const [block] = K.parseMarkdown("**bold *both* bold** and snake_case_name");
  assert.deepEqual(block.segments[0], { text: "bold ", strong: true });
  assert.deepEqual(block.segments[1], { text: "both", strong: true, em: true });
  assert.deepEqual(block.segments[2], { text: " bold", strong: true });
  assert.equal(block.segments[3].text, " and snake_case_name");
  assert.ok(!block.segments[3].em);
});

test("markdown: double-escaped newlines are undone only without real ones", () => {
  // "a\n\nb" with literal backslashes becomes a real blank line: two blocks.
  assert.equal(K.parseMarkdown("a\\n\\nb").length, 2);
  // With a real newline present, a literal "\n" is left alone (code snippets).
  const [block] = K.parseMarkdown('code "\\n" stays\nsecond line');
  assert.ok(block.segments[0].text.includes("\\n"));
});

test("safeHref: the scheme is the security boundary", () => {
  assert.equal(K.safeHref("https://kaval.ai"), "https://kaval.ai");
  assert.equal(K.safeHref("www.kaval.ai"), "https://www.kaval.ai");
  assert.equal(K.safeHref("/docs"), "/docs");
  assert.equal(K.safeHref("#anchor"), "#anchor");
  assert.equal(K.safeHref("javascript:alert(1)"), null);
  assert.equal(K.safeHref("data:text/html,x"), null);
});

test("extractPartialString reads a truncated JSON value", () => {
  const key = "agent_response";
  assert.equal(K.extractPartialString('{"agent_response": "Hello"}', key), "Hello");
  assert.equal(K.extractPartialString('{"agent_response": "Hel', key), "Hel");
  assert.equal(K.extractPartialString('{"agent_response": "a\\nb\\u0041', key), "a\nbA");
  assert.equal(K.extractPartialString('{"thoughts": "hmm"}', key), null);
  assert.equal(K.extractPartialString('{"agent_response":', key), null);
});

test("parseSseFrames splits frames and keeps the trailing partial", () => {
  const { events, rest } = K.parseSseFrames(
    ': ping\n\ndata: {"type":"partial","value":"x"}\n\nnot json\n\ndata: {"type":"comp'
  );
  assert.equal(events.length, 1);
  assert.equal(events[0].type, "partial");
  assert.equal(rest, 'data: {"type":"comp');
});

const encoder = new TextEncoder();

function frame(event) {
  return encoder.encode("data: " + JSON.stringify(event) + "\n\n");
}

function sseResponse(frames, status = 200, headers = {}) {
  const stream = new ReadableStream({
    start(controller) {
      for (const event of frames) {
        controller.enqueue(frame(event));
      }
      controller.close();
    },
  });
  return { status, ok: status < 400, body: stream, headers: new Headers(headers) };
}

/* Delivers `frames` in one chunk, then fails the connection on the next
   read — a stream dropped after the server has started the run. An empty
   `frames` fails the very first read. */
function droppedResponse(frames) {
  let delivered = frames.length === 0;
  const stream = new ReadableStream({
    pull(controller) {
      if (!delivered) {
        delivered = true;
        for (const event of frames) controller.enqueue(frame(event));
        return;
      }
      controller.error(new TypeError("connection reset"));
    },
  });
  return { status: 200, ok: true, body: stream, headers: new Headers() };
}

const COMPLETED = {
  type: "workflow_completed",
  output_data: { agent_response: "Hello", choices: [] },
};

test("agentConnector streams partials and resolves with the reply and run id", async () => {
  let requestBody = null;
  const connector = K.agentConnector({
    url: "/chat",
    fetchImpl: async (url, init) => {
      requestBody = JSON.parse(init.body);
      return sseResponse([
        { type: "workflow_started", run_id: "run-1", session_id: "s-1" },
        { type: "partial", name: "reply", value: '{"agent_response": "Hel' },
        { type: "partial", name: "reply", value: '{"agent_response": "Hello"' },
        {
          type: "workflow_completed",
          name: "wf",
          output_data: { agent_response: "Hello", choices: ["Docs"] },
        },
      ]);
    },
  });

  const partials = [];
  const reply = await connector.send("hi", (text) => partials.push(text));
  assert.deepEqual(partials, ["Hel", "Hello"]);
  assert.deepEqual(reply, { text: "Hello", choices: ["Docs"], runId: "run-1" });
  assert.equal(requestBody.data.message, "hi");
  assert.ok(requestBody.external_id);

  /* The conversation id is stable across messages and rotates on reset. */
  const firstId = requestBody.external_id;
  await connector.send("again", () => {});
  assert.equal(requestBody.external_id, firstId);
  connector.reset();
  await connector.send("fresh", () => {});
  assert.notEqual(requestBody.external_id, firstId);
});

test("agentConnector: a restart event drops the streamed fragment", async () => {
  const connector = K.agentConnector({
    fetchImpl: async () =>
      sseResponse([
        { type: "partial", value: '{"agent_response": "fragment' },
        { type: "restart", name: "reply" },
        { type: "partial", value: '{"agent_response": "clean"' },
        { type: "workflow_completed", output_data: { agent_response: "clean" } },
      ]),
  });
  const partials = [];
  const reply = await connector.send("hi", (text) => partials.push(text));
  assert.deepEqual(partials, ["fragment", "", "clean"]);
  assert.deepEqual(reply, { text: "clean", choices: [], runId: null });
});

test("agentConnector: deliberate backend answers become user-facing errors", async () => {
  const limited = K.agentConnector({ fetchImpl: async () => ({ status: 429 }) });
  await assert.rejects(limited.send("hi"), (e) => e instanceof K.UserError && e.code === "rateLimited");

  const down = K.agentConnector({ fetchImpl: async () => ({ status: 503 }) });
  await assert.rejects(down.send("hi"), (e) => e instanceof K.UserError && e.code === "unavailable");

  const broken = K.agentConnector({ fetchImpl: async () => ({ status: 500, ok: false }) });
  await assert.rejects(broken.send("hi"), /status 500/);
  await assert.rejects(broken.send("hi"), (e) => !(e instanceof K.UserError) && K.isRetryable(e));

  const refused = K.agentConnector({ fetchImpl: async () => ({ status: 401, ok: false }) });
  await assert.rejects(refused.send("hi"), (e) => e.status === 401 && !K.isRetryable(e));
});

test("agentConnector: a failed workflow rejects with its message and is final", async () => {
  const connector = K.agentConnector({
    fetchImpl: async () =>
      sseResponse([{ type: "workflow_failed", value: "model exploded" }]),
  });
  await assert.rejects(connector.send("hi"), /model exploded/);
  await assert.rejects(connector.send("hi"), (e) => !K.isRetryable(e) && e.code === "failed");

  const silent = K.agentConnector({ fetchImpl: async () => sseResponse([]) });
  await assert.rejects(silent.send("hi"), (e) => /did not return a reply/.test(e.message) && K.isRetryable(e));
});

test("agentConnector: a stream dropped after the first frame is never retryable", async () => {
  const started = K.agentConnector({
    fetchImpl: async () => droppedResponse([{ type: "workflow_started", run_id: "r" }]),
  });
  await assert.rejects(started.send("hi"), (e) => e instanceof K.UserError && e.code === "interrupted");

  const ended = K.agentConnector({
    fetchImpl: async () => sseResponse([{ type: "workflow_started", run_id: "r" }]),
  });
  await assert.rejects(ended.send("hi"), (e) => e.code === "interrupted" && !K.isRetryable(e));

  const unstarted = K.agentConnector({ fetchImpl: async () => droppedResponse([]) });
  await assert.rejects(unstarted.send("hi"), (e) => /connection reset/.test(e.message) && K.isRetryable(e));
});

test("agentConnector: custom input and reply keys", async () => {
  let requestBody = null;
  const connector = K.agentConnector({
    inputKey: "user_message",
    replyKey: "answer",
    choicesKey: "options",
    fetchImpl: async (url, init) => {
      requestBody = JSON.parse(init.body);
      return sseResponse([
        { type: "workflow_completed", output_data: { answer: "ok", options: ["a"] } },
      ]);
    },
  });
  const reply = await connector.send("hi");
  assert.equal(requestBody.data.user_message, "hi");
  assert.deepEqual(reply, { text: "ok", choices: ["a"], runId: null });
});

test("agentConnector: headers from an object are sent with the JSON type", async () => {
  let headers = null;
  const connector = K.agentConnector({
    storage: "none",
    headers: { Authorization: "Bearer abc" },
    fetchImpl: async (url, init) => {
      headers = init.headers;
      return sseResponse([COMPLETED]);
    },
  });
  await connector.send("hi");
  assert.deepEqual(headers, {
    "Content-Type": "application/json",
    Authorization: "Bearer abc",
  });
});

test("agentConnector: a header function is awaited afresh for every request", async () => {
  const seen = [];
  let calls = 0;
  const connector = K.agentConnector({
    storage: "none",
    headers: async () => {
      calls += 1;
      return { Authorization: "Bearer token-" + calls };
    },
    fetchImpl: async (url, init) => {
      seen.push(init.headers.Authorization);
      return sseResponse([COMPLETED]);
    },
  });
  await connector.send("one");
  await connector.send("two");
  assert.deepEqual(seen, ["Bearer token-1", "Bearer token-2"]);

  const sync = K.agentConnector({
    storage: "none",
    headers: () => ({ "X-Site": "acme" }),
    fetchImpl: async (url, init) => {
      seen.push(init.headers["X-Site"]);
      return sseResponse([COMPLETED]);
    },
  });
  await sync.send("three");
  assert.equal(seen[2], "acme");
});

test("agentConnector: onResponse sees every response and can adopt a conversation id", async () => {
  const ids = [];
  const statuses = [];
  let status = 200;
  const connector = K.agentConnector({
    storage: "none",
    onResponse: (response, self) => {
      statuses.push(response.status);
      const issued = response.headers && response.headers.get("X-Conversation");
      if (issued) self.setConversationId(issued);
    },
    fetchImpl: async (url, init) => {
      ids.push(JSON.parse(init.body).external_id);
      if (status === 429) return { status: 429, headers: new Headers() };
      return sseResponse([COMPLETED], 200, { "X-Conversation": "server-42" });
    },
  });
  await connector.send("first");
  await connector.send("second");
  assert.notEqual(ids[0], "server-42");
  assert.equal(ids[1], "server-42");
  assert.equal(connector.conversationId(), "server-42");

  status = 429;
  await assert.rejects(connector.send("third"), K.UserError);
  assert.deepEqual(statuses, [200, 200, 429]);

  /* Clearing the id starts a fresh conversation. */
  connector.setConversationId(null);
  assert.notEqual(connector.conversationId(), "server-42");
});

function fakeStorage() {
  const items = new Map();
  return {
    items,
    getItem: (key) => (items.has(key) ? items.get(key) : null),
    setItem: (key, value) => items.set(key, String(value)),
    removeItem: (key) => items.delete(key),
  };
}

function withStorages(session, local, body) {
  const saved = [globalThis.sessionStorage, globalThis.localStorage];
  globalThis.sessionStorage = session;
  globalThis.localStorage = local;
  return Promise.resolve()
    .then(body)
    .finally(() => {
      globalThis.sessionStorage = saved[0];
      globalThis.localStorage = saved[1];
    });
}

test("storage: session and local keep the id in their own area", async () => {
  const session = fakeStorage();
  const local = fakeStorage();
  await withStorages(session, local, () => {
    const tab = K.agentConnector({ storageKey: "k", fetchImpl: async () => null });
    const id = tab.conversationId();
    assert.equal(session.items.get("k"), id);
    assert.equal(local.items.size, 0);
    /* Another connector on the same key continues the conversation. */
    assert.equal(K.agentConnector({ storageKey: "k" }).conversationId(), id);

    const visit = K.agentConnector({ storage: "local", storageKey: "k" });
    const localId = visit.conversationId();
    assert.equal(local.items.get("k"), localId);
    assert.notEqual(localId, id);
    visit.reset();
    assert.equal(local.items.has("k"), false);
  });
});

test("storage: none writes nothing and keeps the id in memory", async () => {
  const session = fakeStorage();
  const local = fakeStorage();
  await withStorages(session, local, () => {
    const connector = K.agentConnector({ storage: "none" });
    const id = connector.conversationId();
    assert.equal(connector.conversationId(), id);
    connector.setConversationId("adopted");
    assert.equal(connector.conversationId(), "adopted");
    connector.reset();
    assert.notEqual(connector.conversationId(), "adopted");
    assert.equal(session.items.size + local.items.size, 0);
  });
});

test("storage: an area that refuses writes falls back to a stable memory id", async () => {
  const refusing = {
    getItem: () => null,
    setItem: () => {
      throw new Error("QuotaExceededError");
    },
    removeItem: () => {
      throw new Error("SecurityError");
    },
  };
  await withStorages(refusing, undefined, () => {
    const connector = K.agentConnector({});
    const id = connector.conversationId();
    assert.equal(connector.conversationId(), id);
    connector.setConversationId("adopted");
    assert.equal(connector.conversationId(), "adopted");
    connector.reset();
    assert.notEqual(connector.conversationId(), "adopted");

    const missing = K.agentConnector({ storage: "local" });
    assert.equal(missing.conversationId(), missing.conversationId());
  });
});

test("storage: an unknown mode is refused", () => {
  assert.throws(() => K.agentConnector({ storage: "cookie" }), /storage must be/);
  assert.throws(() => K.agentConnector({ storage: "toString" }), /storage must be/);
});

test("storage: ids come from getRandomValues outside a secure context", () => {
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, "crypto");
  Object.defineProperty(globalThis, "crypto", {
    configurable: true,
    value: { getRandomValues: (bytes) => bytes.fill(171) },
  });
  try {
    const id = K.agentConnector({ storage: "none" }).conversationId();
    assert.equal(id, "ab".repeat(16));
  } finally {
    Object.defineProperty(globalThis, "crypto", descriptor);
  }
});

test("texts: legacy options map in, texts win, unknown keys are refused", () => {
  const texts = K.resolveTexts({
    title: "Legacy title",
    placeholder: "Legacy placeholder",
    texts: { placeholder: "Kirjuta...", send: "Saada" },
  });
  assert.equal(texts.title, "Legacy title");
  assert.equal(texts.placeholder, "Kirjuta...");
  assert.equal(texts.send, "Saada");
  assert.equal(texts.disclosure, "AI assistant");
  assert.deepEqual(K.resolveTexts(), Object.assign({}, K.DEFAULT_TEXTS));
  assert.throws(() => K.resolveTexts({ texts: { sned: "x" } }), /Unknown texts key: sned/);
});

test("errorText: codes select texts, other errors stay generic", () => {
  const texts = K.resolveTexts({ texts: { interrupted: "Katkes", error: "Viga" } });
  assert.equal(K.errorText(new K.UserError("x", "interrupted"), texts), "Katkes");
  assert.equal(K.errorText(new K.UserError("Say this verbatim"), texts), "Say this verbatim");
  assert.equal(K.errorText(new K.UserError("x", "title"), texts), "x");
  assert.equal(K.errorText(new Error("internal detail"), texts), "Viga");
});

test("isRetryable: only errors from turns the server never started", () => {
  assert.equal(K.isRetryable(new Error("network")), true);
  assert.equal(K.isRetryable(new K.UserError("limit")), false);
  const started = new Error("failed");
  started.retryable = false;
  assert.equal(K.isRetryable(started), false);
  assert.equal(K.isRetryable(undefined), true);
});

/* --- mount(), over the fake DOM ------------------------------------- */

const dom = installDom();

function mountWith(options) {
  const widget = K.mount(
    Object.assign({ retryDelayMs: 0, mode: "floating" }, options)
  );
  const find = (selector) => widget.root.querySelector(selector);
  const all = (selector) => widget.root.querySelectorAll(selector);
  return { widget, find, all };
}

function scripted(reply) {
  return async (text, onPartial) => {
    onPartial("partial");
    return Object.assign({ text: "Answer to " + text, choices: [] }, reply);
  };
}

test("mount: the disclosure label is on by default, in both modes", () => {
  const floating = mountWith({ connector: scripted() });
  const label = floating.find(".kcb-disclosure");
  assert.ok(label, "floating widget has no disclosure label");
  assert.equal(label.textContent, "AI assistant");
  assert.ok(label.parentNode.matches(".kcb-title-text"));
  floating.widget.destroy();

  const host = dom.document.createElement("div");
  dom.document.body.appendChild(host);
  const inline = mountWith({
    connector: scripted(),
    mode: "inline",
    target: host,
    texts: { disclosure: "Tehisintellekt" },
  });
  assert.equal(inline.find(".kcb-disclosure").textContent, "Tehisintellekt");
  inline.widget.destroy();
});

test("mount: disclosure false hides the label", () => {
  const { widget, find } = mountWith({ connector: scripted(), disclosure: false });
  assert.equal(find(".kcb-disclosure"), null);
  widget.destroy();
});

test("mount: texts replace every visible string and aria label", async () => {
  const texts = {
    title: "Tugi",
    emptyMessage: "Küsi julgelt.",
    placeholder: "Kirjuta...",
    inputLabel: "Sõnum",
    send: "Saada",
    openChat: "Ava vestlus",
    closeChat: "Sulge vestlus",
    maximize: "Suurenda",
    restore: "Taasta",
    typing: "Assistent kirjutab",
    greeting: "Tere!",
  };
  const { widget, find } = mountWith({ connector: scripted(), texts });
  assert.equal(find(".kcb-title-label").textContent, "Tugi");
  assert.equal(find(".kcb-empty").textContent, "Küsi julgelt.");
  assert.equal(find(".kcb-greeting").textContent, "Tere!");
  assert.equal(find(".kcb-send").textContent, "Saada");
  const input = find(".kcb-input-row").children[0];
  assert.equal(input.placeholder, "Kirjuta...");
  assert.equal(input.getAttribute("aria-label"), "Sõnum");
  assert.equal(find(".kcb-typing").getAttribute("aria-label"), "Assistent kirjutab");
  assert.equal(find(".kcb-bubble-float").getAttribute("aria-label"), "Ava vestlus");
  assert.equal(find(".kcb-bubble-inline").getAttribute("aria-label"), "Sulge vestlus");

  widget.open();
  assert.equal(find(".kcb-bubble-float").getAttribute("aria-label"), "Sulge vestlus");
  const maximize = find(".kcb-maximize");
  assert.equal(maximize.getAttribute("aria-label"), "Suurenda");
  maximize.click();
  assert.equal(maximize.getAttribute("aria-label"), "Taasta");
  widget.destroy();
});

test("mount: texts replace the error messages a turn can end with", async () => {
  const failing = (error) => async () => {
    throw error;
  };
  const texts = {
    error: "Midagi läks valesti.",
    rateLimited: "Liiga palju sõnumeid.",
    interrupted: "Vastus katkes.",
  };
  const cases = [
    [new K.UserError("Too many", "rateLimited"), "Liiga palju sõnumeid."],
    [new K.UserError("cut", "interrupted"), "Vastus katkes."],
    [Object.assign(new Error("secret stack"), { retryable: false }), "Midagi läks valesti."],
  ];
  for (const [error, shown] of cases) {
    const { widget, find } = mountWith({ connector: failing(error), texts });
    await widget.send("hi");
    const bubble = find(".kcb-error");
    assert.ok(bubble.matches(".kcb-message.kcb-from-agent"));
    assert.equal(bubble.textContent, shown);
    widget.destroy();
  }
});

test("mount: feedback controls render only when onFeedback is given", async () => {
  const without = mountWith({ connector: scripted({ runId: "run-1" }) });
  await without.widget.send("hi");
  assert.equal(without.find(".kcb-feedback"), null);
  without.widget.destroy();

  const votes = [];
  const noRun = mountWith({
    connector: scripted({ runId: null }),
    onFeedback: (...args) => votes.push(args),
  });
  await noRun.widget.send("hi");
  assert.equal(noRun.find(".kcb-feedback"), null, "no run id, nothing to rate");
  noRun.widget.destroy();

  assert.throws(() => mountWith({ connector: scripted(), onFeedback: "yes" }), /onFeedback/);
});

test("mount: a vote reaches onFeedback with the run id and marks the thumb", async () => {
  const votes = [];
  const events = [];
  const { widget, find } = mountWith({
    connector: scripted({ runId: "run-7" }),
    onFeedback: (...args) => votes.push(args),
    texts: { feedbackUp: "Hea vastus", feedbackDown: "Halb vastus" },
  });
  widget.on("feedback", (payload) => events.push(payload));
  await widget.send("hi");

  const row = find(".kcb-feedback");
  assert.equal(row.getAttribute("role"), "group");
  const up = find(".kcb-feedback-up");
  const down = find(".kcb-feedback-down");
  assert.equal(up.getAttribute("aria-label"), "Hea vastus");
  assert.equal(down.getAttribute("aria-label"), "Halb vastus");

  up.click();
  down.click();
  assert.deepEqual(votes, [
    ["run-7", "up", null],
    ["run-7", "down", null],
  ]);
  assert.equal(up.getAttribute("aria-pressed"), "false");
  assert.equal(down.getAttribute("aria-pressed"), "true");
  assert.deepEqual(events[1], { runId: "run-7", vote: "down", comment: null });
  assert.equal(find(".kcb-feedback-comment"), null, "no comment box unless asked");

  /* A rejected or throwing callback is reported, not thrown into the UI. */
  const errors = [];
  const original = console.error;
  console.error = (error) => errors.push(error);
  try {
    const rejecting = mountWith({
      connector: scripted({ runId: "r" }),
      onFeedback: () => Promise.reject(new Error("store down")),
    });
    await rejecting.widget.send("hi");
    rejecting.find(".kcb-feedback-up").click();
    await new Promise((resolve) => setImmediate(resolve));
    const throwing = mountWith({
      connector: scripted({ runId: "r" }),
      onFeedback: () => {
        throw new Error("sync failure");
      },
    });
    await throwing.widget.send("hi");
    throwing.find(".kcb-feedback-down").click();
    rejecting.widget.destroy();
    throwing.widget.destroy();
  } finally {
    console.error = original;
  }
  assert.deepEqual(errors.map((e) => e.message), ["store down", "sync failure"]);
  widget.destroy();
});

test("mount: feedbackComment asks for a comment after the first vote", async () => {
  const votes = [];
  const { widget, find, all } = mountWith({
    connector: scripted({ runId: "run-9" }),
    onFeedback: (...args) => votes.push(args),
    feedbackComment: true,
    texts: { feedbackThanks: "Aitäh!" },
  });
  await widget.send("hi");
  find(".kcb-feedback-down").click();
  const form = find(".kcb-feedback-comment");
  assert.ok(form, "the comment box follows the vote");
  find(".kcb-feedback-up").click();
  assert.equal(all(".kcb-feedback-comment").length, 1, "only one box per answer");

  const field = form.children[0];
  field.value = "   ";
  form.dispatch("submit");
  assert.equal(votes.length, 2, "an empty comment is not delivered");

  field.value = "Too long";
  form.dispatch("submit");
  assert.deepEqual(votes[2], ["run-9", "up", "Too long"]);
  assert.equal(find(".kcb-feedback-thanks").textContent, "Aitäh!");
  widget.destroy();
});

test("mount: events fire and an unsubscribed listener stops hearing them", async () => {
  const heard = [];
  const { widget } = mountWith({ connector: scripted({ runId: "run-3" }) });
  const offOpen = widget.on("open", () => heard.push("open"));
  widget.on("close", () => heard.push("close"));
  widget.on("reply", (reply) => heard.push(reply));
  widget.on("error", (payload) => heard.push(payload));

  widget.open();
  widget.open();
  widget.close();
  offOpen();
  widget.open();
  assert.deepEqual(heard, ["open", "close"]);

  await widget.send("hi");
  assert.deepEqual(heard[2], {
    message: "hi",
    text: "Answer to hi",
    choices: [],
    runId: "run-3",
  });

  assert.throws(() => widget.on("opened", () => {}), /Unknown widget event: opened/);
  assert.throws(() => widget.on("open", "nope"), /must be a function/);
  widget.destroy();
});

test("mount: a throwing listener is reported and the turn still completes", async () => {
  const errors = [];
  const original = console.error;
  console.error = (error) => errors.push(error);
  try {
    const { widget, find } = mountWith({ connector: scripted() });
    widget.on("reply", () => {
      throw new Error("host bug");
    });
    await widget.send("hi");
    assert.equal(find(".kcb-from-agent").textContent, "Answer to hi");
    assert.equal(find(".kcb-error"), null);
    widget.destroy();
  } finally {
    console.error = original;
  }
  assert.equal(errors[0].message, "host bug");
});

test("mount: resize reports the window size on maximise, restore and viewport change", () => {
  const sizes = [];
  const { widget, find } = mountWith({ connector: scripted() });
  widget.on("resize", (size) => sizes.push(size));
  widget.open();
  find(".kcb-maximize").click();
  dom.window.innerHeight = 700;
  dom.window.dispatch("resize");
  find(".kcb-maximize").click();
  dom.window.dispatch("resize");
  dom.window.innerHeight = 800;

  assert.deepEqual(sizes, [
    { width: 520, height: 708, maximized: true },
    { width: 520, height: 608, maximized: true },
    { width: 340, height: 480, maximized: false },
  ]);

  /* Dragging the left edge after maximising restores the button state. */
  find(".kcb-maximize").click();
  const edge = find(".kcb-resize-left");
  edge.dispatch("pointerdown", { clientX: 900, clientY: 300 });
  edge.dispatch("pointermove", { clientX: 800, clientY: 300 });
  edge.dispatch("pointerup", {});
  assert.equal(find(".kcb-maximize").getAttribute("aria-pressed"), "false");
  assert.deepEqual(sizes[sizes.length - 1], { width: 620, height: 708, maximized: false });
  widget.destroy();
});

test("mount: a turn the server has started is never sent again", async () => {
  let calls = 0;
  const heard = [];
  const connector = K.agentConnector({
    storage: "none",
    fetchImpl: async () => {
      calls += 1;
      return droppedResponse([
        { type: "workflow_started", run_id: "r" },
        { type: "partial", value: '{"agent_response": "Half an ans' },
      ]);
    },
  });
  const { widget, find } = mountWith({
    connector,
    texts: { interrupted: "Vastus katkes." },
  });
  widget.on("error", (payload) => heard.push(payload));
  await widget.send("hi");
  assert.equal(calls, 1);
  assert.equal(find(".kcb-error").textContent, "Vastus katkes.");
  assert.equal(heard[0].message, "hi");
  assert.equal(heard[0].text, "Vastus katkes.");
  assert.equal(heard[0].error.code, "interrupted");
  widget.destroy();

  let failedCalls = 0;
  const failed = mountWith({
    connector: K.agentConnector({
      storage: "none",
      fetchImpl: async () => {
        failedCalls += 1;
        return sseResponse([
          { type: "workflow_started", run_id: "r" },
          { type: "workflow_failed", value: "Traceback: secret" },
        ]);
      },
    }),
  });
  await failed.widget.send("hi");
  assert.equal(failedCalls, 1);
  assert.equal(failed.find(".kcb-error").textContent, K.DEFAULT_TEXTS.error);
  failed.widget.destroy();
});

test("mount: a turn that never started is retried, up to maxRetries", async () => {
  let calls = 0;
  const flaky = mountWith({
    connector: K.agentConnector({
      storage: "none",
      fetchImpl: async () => {
        calls += 1;
        if (calls === 1) throw new TypeError("fetch failed");
        if (calls === 2) return droppedResponse([]);
        return sseResponse([
          { type: "workflow_started", run_id: "run-5" },
          { type: "workflow_completed", output_data: { agent_response: "Third time" } },
        ]);
      },
    }),
  });
  await flaky.widget.send("hi");
  assert.equal(calls, 3);
  assert.equal(flaky.find(".kcb-from-agent").textContent, "Third time");
  flaky.widget.destroy();

  let attempts = 0;
  const down = mountWith({
    maxRetries: 1,
    connector: async () => {
      attempts += 1;
      throw new Error("still down");
    },
  });
  await down.widget.send("hi");
  assert.equal(attempts, 2);
  assert.equal(down.find(".kcb-error").textContent, K.DEFAULT_TEXTS.error);
  down.widget.destroy();
});

test("mount: reset clears answers and their feedback rows", async () => {
  const { widget, find, all } = mountWith({
    connector: scripted({ runId: "r" }),
    onFeedback: () => {},
  });
  await widget.send("hi");
  assert.equal(all(".kcb-feedback").length, 1);
  widget.reset();
  assert.equal(all(".kcb-message.kcb-from-agent").length, 0);
  assert.equal(all(".kcb-feedback").length, 0);
  assert.equal(find(".kcb-empty").hidden, false);
  assert.equal(isShown(find(".kcb-empty")), false, "the floating window is still closed");
  widget.open();
  assert.ok(isShown(find(".kcb-empty")));
  widget.destroy();
});

/* --- the reference page stays true to the code ----------------------- */

const docs = fs.readFileSync(DOCS, "utf8");
const flatDocs = docs.replace(/\s+/g, " ");

test("docs: every texts key is documented with its default", () => {
  for (const [key, value] of Object.entries(K.DEFAULT_TEXTS)) {
    assert.ok(flatDocs.includes("``" + key + "``"), "undocumented texts key " + key);
    assert.ok(flatDocs.includes(value), "default of " + key + " not in the docs");
  }
});

test("docs: every event is documented", () => {
  for (const name of K.EVENTS) {
    assert.ok(docs.includes('``"' + name + '"``'), "undocumented event " + name);
  }
});

test("docs: every token the stylesheet declares is documented", () => {
  const css = fs.readFileSync(CSS, "utf8");
  const block = css.slice(css.indexOf(".kcb-chatbot {"), css.indexOf("}", css.indexOf(".kcb-chatbot {")));
  const tokens = block.match(/--kcb-[a-z-]+(?=:)/g);
  assert.ok(tokens.length > 10);
  for (const token of tokens) {
    assert.ok(docs.includes("``" + token + "``"), "undocumented token " + token);
  }
});

test("docs: every class the page names exists in the widget", () => {
  const source = fs.readFileSync(JS, "utf8") + fs.readFileSync(CSS, "utf8");
  const named = new Set(docs.match(/kcb-[a-z-]+/g).filter((name) => !name.endsWith("-")));
  for (const name of named) {
    if (name.startsWith("kcb-") && !docs.includes("--" + name)) {
      assert.ok(source.includes(name), "the docs name a class the widget lacks: " + name);
    }
  }
});
