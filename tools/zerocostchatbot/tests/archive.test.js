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
