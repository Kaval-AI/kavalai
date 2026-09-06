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

Run with: node --test chatbotwidget/tests/
Covers the DOM-free logic: markdown parsing, SSE frame handling and the
agent-server connector. The widget's DOM behaviour is exercised manually via
preview.html.
*/

const test = require("node:test");
const assert = require("node:assert/strict");
const K = require("../kaval-chatbot.js");

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

function sseResponse(frames, status = 200) {
  const encoder = new TextEncoder();
  const stream = new ReadableStream({
    start(controller) {
      for (const frame of frames) {
        controller.enqueue(encoder.encode("data: " + JSON.stringify(frame) + "\n\n"));
      }
      controller.close();
    },
  });
  return { status, ok: status < 400, body: stream };
}

test("agentConnector streams partials and resolves with the reply", async () => {
  let requestBody = null;
  const connector = K.agentConnector({
    url: "/chat",
    fetchImpl: async (url, init) => {
      requestBody = JSON.parse(init.body);
      return sseResponse([
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
  assert.deepEqual(reply, { text: "Hello", choices: ["Docs"] });
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
  assert.deepEqual(reply, { text: "clean", choices: [] });
});

test("agentConnector: deliberate backend answers become user-facing errors", async () => {
  const limited = K.agentConnector({ fetchImpl: async () => ({ status: 429 }) });
  await assert.rejects(limited.send("hi"), K.UserError);

  const down = K.agentConnector({ fetchImpl: async () => ({ status: 503 }) });
  await assert.rejects(down.send("hi"), K.UserError);

  const broken = K.agentConnector({ fetchImpl: async () => ({ status: 500, ok: false }) });
  await assert.rejects(broken.send("hi"), /status 500/);
  await assert.rejects(broken.send("hi"), (e) => !(e instanceof K.UserError));
});

test("agentConnector: a failed workflow rejects with its message", async () => {
  const connector = K.agentConnector({
    fetchImpl: async () =>
      sseResponse([{ type: "workflow_failed", value: "model exploded" }]),
  });
  await assert.rejects(connector.send("hi"), /model exploded/);

  const silent = K.agentConnector({ fetchImpl: async () => sseResponse([]) });
  await assert.rejects(silent.send("hi"), /did not return a reply/);
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
  assert.deepEqual(reply, { text: "ok", choices: ["a"] });
});
