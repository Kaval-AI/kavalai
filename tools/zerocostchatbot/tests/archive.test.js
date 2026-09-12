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

Run with: node --test tools/zerocostchatbot/tests/archive.test.js
*/

const test = require("node:test");
const assert = require("node:assert/strict");
const A = require("../archive.js");

const PAGES = [
  ["https://docs.kaval.ai/", "Home", 200, "http", "2026-09-06T10:00:00"],
  ["https://docs.kaval.ai/guide", "Guide", 200, "browser", "2026-09-06T10:01:00"],
  ["https://docs.kaval.ai/api/", "API", 200, "http", "2026-09-06T10:02:00"],
];
const HTML = {
  "https://docs.kaval.ai/": "<html><head></head><body>home</body></html>",
  "https://docs.kaval.ai/guide": "<p>guide</p>",
  "https://docs.kaval.ai/api/": "<p>api</p>",
};

/* A stand-in for a sql.js Database: enough SQL to serve ArchiveIndex. */
function fakeDb() {
  return {
    exec(sql, params) {
      if (sql.includes("SELECT html")) {
        const html = HTML[params[0]];
        return html === undefined ? [] : [{ columns: ["html"], values: [[html]] }];
      }
      if (sql.includes("SELECT screenshot")) {
        return params[0] === "https://docs.kaval.ai/"
          ? [{ columns: ["screenshot"], values: [[new Uint8Array([1, 2])]] }]
          : [{ columns: ["screenshot"], values: [[null]] }];
      }
      return [
        {
          columns: ["url", "title", "status_code", "fetch_mode", "last_crawled_at"],
          values: PAGES,
        },
      ];
    },
  };
}

test("normalizeUrl resolves, drops fragments and rejects non-http", () => {
  assert.equal(A.normalizeUrl("../x#top", "https://Docs.Kaval.ai/a/b/"), "https://docs.kaval.ai/a/x");
  assert.equal(A.normalizeUrl("mailto:hi@kaval.ai"), null);
  assert.equal(A.normalizeUrl("not a url"), null);
});

test("candidateUrls covers slash and scheme variants but not query strings", () => {
  assert.deepEqual(A.candidateUrls("https://a.io/docs"), [
    "https://a.io/docs",
    "https://a.io/docs/",
    "http://a.io/docs",
    "http://a.io/docs/",
  ]);
  assert.deepEqual(A.candidateUrls("https://a.io/"), ["https://a.io/", "http://a.io/"]);
  assert.deepEqual(A.candidateUrls("https://a.io/s?q=1"), ["https://a.io/s?q=1", "http://a.io/s?q=1"]);
});

test("ArchiveIndex lists fetched pages and finds link targets", () => {
  const index = new A.ArchiveIndex(fakeDb());
  assert.deepEqual(index.pages.map((p) => p.title), ["Home", "Guide", "API"]);
  assert.equal(index.find("/guide#section", "https://docs.kaval.ai/").url, "https://docs.kaval.ai/guide");
  // Trailing-slash and scheme differences are forgiven.
  assert.equal(index.find("http://docs.kaval.ai/api").url, "https://docs.kaval.ai/api/");
  assert.equal(index.find("https://docs.kaval.ai/missing"), null);
  assert.equal(index.find("javascript:void(0)"), null);
  assert.equal(index.html("https://docs.kaval.ai/guide"), "<p>guide</p>");
  assert.equal(index.html("https://docs.kaval.ai/nope"), null);
  assert.deepEqual(Array.from(index.screenshot("https://docs.kaval.ai/")), [1, 2]);
  assert.equal(index.screenshot("https://docs.kaval.ai/guide"), null);
});

test("prepareArchivedHtml strips scripts, anchors the base and injects the interceptor", () => {
  const html =
    '<html><head><base href="https://evil/"><script src="x.js"></script></head>' +
    '<body><a href="/guide">g</a><script>alert(1)</script></body></html>';
  const out = A.prepareArchivedHtml(html, 'https://docs.kaval.ai/"q', ["https://docs.kaval.ai/guide"]);
  assert.ok(!out.includes("alert(1)") && !out.includes("x.js"));
  assert.ok(!out.includes('href="https://evil/"'));
  // The page URL is quoted safely and sits right after <head>.
  assert.ok(out.startsWith('<html><head><base href="https://docs.kaval.ai/&quot;q">'));
  assert.ok(out.includes("kaval-archive-navigate"));
  assert.ok(out.includes('new Set(["https://docs.kaval.ai/guide"])'));
  // Only one script survives: the interceptor itself.
  assert.equal((out.match(/<script/g) || []).length, 1);
});

test("prepareArchivedHtml can keep site scripts and handles headless fragments", () => {
  const kept = A.prepareArchivedHtml("<head></head><script>run()</script>", "https://a.io/", [], {
    runScripts: true,
  });
  assert.ok(kept.includes("run()"));
  const fragment = A.prepareArchivedHtml("<p>x</p>", "https://a.io/", []);
  assert.ok(fragment.startsWith('<base href="https://a.io/">'));
  assert.ok(fragment.endsWith("<p>x</p>"));
});

/* ---- RAG index ------------------------------------------------------- */

function f32(values) {
  return new Uint8Array(new Float32Array(values).buffer);
}

const CHUNKS = [
  ["Acme › Pricing\n\nAnvils cost ten dollars each, shipping included.", f32([1, 0, 0]), { url: "https://acme.com/pricing", title: "Pricing", heading: "Pricing" }],
  ["Acme › About\n\nAcme makes fine anvils for discerning coyotes.", f32([0, 1, 0]), { url: "https://acme.com/", title: "Acme", heading: "About" }],
  ["Acme › Contact\n\nWrite to us about anvils and delivery times.", f32([0.7, 0.7, 0]), { url: "https://acme.com/contact", title: "Contact", heading: "" }],
  ["no vector row", null, null],
];

function fakeRagDb(collections) {
  return {
    exec(sql) {
      if (sql.includes("FROM rag_collections")) {
        return collections.length
          ? [{ columns: ["name", "table_name", "model", "embedding_size"], values: collections }]
          : [];
      }
      assert.ok(sql.includes("FROM rag_c_acme"), "reads the table the registry names");
      return [
        {
          columns: ["content", "embedding", "metadata"],
          values: CHUNKS.map(([text, vector, meta]) => [text, vector, meta && JSON.stringify(meta)]),
        },
      ];
    },
  };
}

const ONE = [["acme.com", "rag_c_acme_1234", "fastembed/snowflake/snowflake-arctic-embed-s", 3]];

test("decodeF32 and cosine", () => {
  assert.deepEqual(Array.from(A.decodeF32(f32([1.5, -2]))), [1.5, -2]);
  // An unaligned view still decodes: sql.js may hand back offsets inside a page.
  const padded = new Uint8Array(1 + 8);
  padded.set(f32([3, 4]), 1);
  assert.deepEqual(Array.from(A.decodeF32(padded.subarray(1))), [3, 4]);
  assert.equal(A.cosine([1, 0], [0, 1]), 0);
  assert.equal(A.cosine([0, 0], [1, 1]), 0);
  assert.ok(Math.abs(A.cosine([1, 1], [2, 2]) - 1) < 1e-9);
});

test("queryText adds the arctic prefix only for arctic models", () => {
  assert.ok(A.queryText("fastembed/snowflake/snowflake-arctic-embed-s", "hi").startsWith("Represent this sentence"));
  assert.equal(A.queryText("fastembed/BAAI/bge-small-en-v1.5", "hi"), "hi");
});

test("RagIndex loads the single collection and ranks by cosine", () => {
  const rag = new A.RagIndex(fakeRagDb(ONE));
  assert.equal(rag.name, "acme.com");
  assert.equal(rag.embeddingSize, 3);
  assert.equal(rag.chunks.length, 4);
  assert.deepEqual(rag.chunks[3].metadata, {});
  assert.equal(rag.chunks[3].vector, null);

  const hits = rag.topK([1, 0.1, 0], 2);
  assert.deepEqual(hits.map((h) => h.chunk.metadata.url), ["https://acme.com/pricing", "https://acme.com/contact"]);
  assert.ok(hits[0].score > hits[1].score);
});

test("RagIndex lexical ranking finds passages by their words", () => {
  const rag = new A.RagIndex(fakeRagDb(ONE));
  const hits = rag.lexicalTopK("how much do anvils cost?", 2);
  assert.equal(hits[0].chunk.metadata.url, "https://acme.com/pricing");
  assert.deepEqual(rag.lexicalTopK("zzzz", 3), []);
});

test("RagIndex refuses ambiguity and missing collections", () => {
  const two = ONE.concat([["other", "rag_c_acme_9", "m", 3]]);
  assert.throws(() => new A.RagIndex(fakeRagDb(two)), /Pick one of the collections: acme.com, other/);
  assert.equal(new A.RagIndex(fakeRagDb(two), "acme.com").name, "acme.com");
  assert.throws(() => new A.RagIndex(fakeRagDb(two), "nope"), /Pick one/);
  assert.throws(() => new A.RagIndex(fakeRagDb([])), /no collections/);
});

test("buildMessages grounds the model in the passages and keeps history", () => {
  const rag = new A.RagIndex(fakeRagDb(ONE));
  const hits = rag.topK([1, 0, 0], 1);
  const history = [{ role: "user", content: "earlier" }, { role: "assistant", content: "reply" }];
  const messages = A.buildMessages("How much?", hits, history, "acme.com");
  assert.equal(messages.length, 4);
  assert.equal(messages[0].role, "system");
  assert.ok(messages[0].content.includes("acme.com"));
  assert.ok(messages[0].content.includes("[1] Pricing › Pricing\nAcme › Pricing"));
  assert.deepEqual(messages.slice(1, 3), history);
  assert.deepEqual(messages[3], { role: "user", content: "How much?" });
});

test("sourcesMarkdown and passagesMarkdown link each page once", () => {
  const rag = new A.RagIndex(fakeRagDb(ONE));
  const hits = rag.topK([0.7, 0.7, 0], 3);
  const sources = A.sourcesMarkdown(hits);
  assert.ok(sources.startsWith("\n\n**Sources**\n- ["));
  assert.equal((sources.match(/acme\.com/g) || []).length, 3);
  assert.equal(A.sourcesMarkdown([]), "");

  const passages = A.passagesMarkdown(hits.slice(0, 1));
  assert.ok(passages.includes("**Contact** — Write to us about anvils"));
  assert.ok(passages.includes("[Read more](https://acme.com/contact)"));
  assert.ok(A.passagesMarkdown([]).startsWith("I could not find"));
});
