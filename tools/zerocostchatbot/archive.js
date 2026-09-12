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

The DOM-free half of archive.html: URL matching against the archived
pages, and preparing a stored page for display in an iframe. archive.html
supplies the sql.js database and the UI; tests run this file under node.
*/
(function (root, factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory();
  } else {
    root.KavalArchive = factory();
  }
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  /* An absolute URL without its fragment, or null for anything that is not
     an http(s) location — the same shape scrape.py stores. */
  function normalizeUrl(href, base) {
    let url;
    try {
      url = base ? new URL(href, base) : new URL(href);
    } catch (_error) {
      return null;
    }
    if (url.protocol !== "http:" && url.protocol !== "https:") {
      return null;
    }
    url.hash = "";
    url.hostname = url.hostname.toLowerCase();
    return url.href;
  }

  /* The forms a stored URL may have been recorded in: the crawler keeps
     what the site linked, so `/docs` and `/docs/` (and the other scheme)
     are all worth a look before declaring a page missing. */
  function candidateUrls(url) {
    const out = [url];
    const parsed = new URL(url);
    if (!parsed.search) {
      if (parsed.pathname.endsWith("/") && parsed.pathname !== "/") {
        out.push(url.replace(/\/$/, ""));
      } else if (!parsed.pathname.endsWith("/")) {
        out.push(url + "/");
      }
    }
    for (const variant of out.slice()) {
      out.push(
        variant.startsWith("https:")
          ? "http:" + variant.slice(6)
          : "https:" + variant.slice(5)
      );
    }
    return out;
  }

  /* The archived pages of one database. `db` is an opened sql.js Database
     (anything with `exec(sql, params)` returning [{columns, values}]). */
  class ArchiveIndex {
    constructor(db) {
      this.db = db;
      this.pages = rows(
        db.exec(
          "SELECT url, title, status_code, fetch_mode, last_crawled_at" +
            " FROM pages WHERE html IS NOT NULL ORDER BY rowid"
        )
      );
      this.byUrl = new Map(this.pages.map((page) => [page.url, page]));
    }

    /* The archived page a link points at, or null. */
    find(href, base) {
      const url = normalizeUrl(href, base);
      if (!url) {
        return null;
      }
      for (const candidate of candidateUrls(url)) {
        const page = this.byUrl.get(candidate);
        if (page) {
          return page;
        }
      }
      return null;
    }

    /* The stored HTML of one page, read on demand — a crawl's HTML is the
       bulk of the file and only one page is shown at a time. */
    html(url) {
      const found = rows(
        this.db.exec("SELECT html FROM pages WHERE url = ?", [url])
      );
      return found.length ? found[0].html : null;
    }

    /* The start page's PNG capture as bytes, or null. */
    screenshot(url) {
      const found = rows(
        this.db.exec("SELECT screenshot FROM pages WHERE url = ?", [url])
      );
      return found.length && found[0].screenshot ? found[0].screenshot : null;
    }
  }

  function rows(result) {
    if (!result || !result.length) {
      return [];
    }
    const { columns, values } = result[0];
    return values.map((value) =>
      Object.fromEntries(columns.map((column, i) => [column, value[i]]))
    );
  }

  const SCRIPT_TAGS = /<script\b[^>]*>[\s\S]*?<\/script\s*>/gi;
  const BASE_TAGS = /<base\b[^>]*>/gi;

  /* Runs inside the iframe: every link click is reported to the viewer
     instead of navigating, and links are marked by whether the viewer holds
     their target, so a reader can see where the archive ends. */
  const INTERCEPTOR = [
    "<script>(function () {",
    "  var archived = new Set(__ARCHIVED__);",
    "  function key(href) {",
    "    try { var u = new URL(href, document.baseURI); u.hash = ''; return u.href; }",
    "    catch (e) { return null; }",
    "  }",
    "  function known(href) {",
    "    var k = key(href); if (!k) return false;",
    "    return archived.has(k) || archived.has(k.replace(/\\/$/, '')) || archived.has(k + '/');",
    "  }",
    "  document.addEventListener('click', function (event) {",
    "    var a = event.target && event.target.closest ? event.target.closest('a[href]') : null;",
    "    if (!a) return;",
    "    event.preventDefault();",
    "    parent.postMessage({ type: 'kaval-archive-navigate', href: a.href, archived: known(a.href) }, '*');",
    "  }, true);",
    "  document.addEventListener('DOMContentLoaded', function () {",
    "    document.querySelectorAll('a[href]').forEach(function (a) {",
    "      a.setAttribute('data-archived', known(a.href) ? 'yes' : 'no');",
    "    });",
    "  });",
    "})();</script>",
    "<style>a[data-archived=\"no\"] { text-decoration-style: dotted !important; opacity: 0.6; }</style>",
  ].join("\n");

  /* The stored HTML made displayable in an iframe: site scripts removed
     unless asked for (a snapshot should not phone home or navigate on its
     own), a `<base>` so relative styles and images resolve against the
     live site, and the interceptor above. */
  function prepareArchivedHtml(html, pageUrl, archivedUrls, options) {
    options = options || {};
    let out = html.replace(BASE_TAGS, "");
    if (!options.runScripts) {
      out = out.replace(SCRIPT_TAGS, "");
    }
    const injected =
      '<base href="' + pageUrl.replace(/"/g, "&quot;") + '">' +
      INTERCEPTOR.replace("__ARCHIVED__", JSON.stringify(archivedUrls));
    const at = out.search(/<head\b[^>]*>/i);
    if (at !== -1) {
      const end = out.indexOf(">", at) + 1;
      return out.slice(0, end) + injected + out.slice(end);
    }
    return injected + out;
  }

  /* ------------------------------------------------------------------ *
   * RAG index (the .rag.db beside the pages database)                   *
   * ------------------------------------------------------------------ */

  /* Embeddings are stored as little-endian FLOAT32 blobs (SqliteRagService's
     `vector_as_f32`); sql.js hands them back as Uint8Array views that may
     not be 4-byte aligned, hence the copy. */
  function decodeF32(blob) {
    const bytes = blob instanceof Uint8Array ? blob : new Uint8Array(blob);
    const copy = bytes.slice();
    return new Float32Array(copy.buffer, 0, Math.floor(copy.byteLength / 4));
  }

  function cosine(a, b) {
    let dot = 0;
    let na = 0;
    let nb = 0;
    const n = Math.min(a.length, b.length);
    for (let i = 0; i < n; i++) {
      dot += a[i] * b[i];
      na += a[i] * a[i];
      nb += b[i] * b[i];
    }
    return na && nb ? dot / Math.sqrt(na * nb) : 0;
  }

  function tokenize(text) {
    return (text || "").toLowerCase().match(/[a-z0-9]{2,}/g) || [];
  }

  /* Snowflake's arctic-embed models are trained with a query-side prefix;
     documents were embedded bare, queries should not be. */
  const ARCTIC_QUERY_PREFIX =
    "Represent this sentence for searching relevant passages: ";

  function queryText(model, text) {
    return /arctic/i.test(model || "") ? ARCTIC_QUERY_PREFIX + text : text;
  }

  /* One collection of a RAG index, loaded in full — a site's chunks and
     384-float vectors are a few megabytes, and a linear cosine scan over
     them is instant next to a model call. */
  class RagIndex {
    constructor(db, collectionName) {
      const registry = rows(
        db.exec("SELECT name, table_name, model, embedding_size FROM rag_collections")
      );
      const names = registry.map((r) => r.name);
      let info = null;
      if (collectionName) {
        info = registry.find((r) => r.name === collectionName) || null;
      } else if (registry.length === 1) {
        info = registry[0];
      }
      if (!info) {
        throw new Error(
          registry.length
            ? "Pick one of the collections: " + names.join(", ")
            : "The RAG index holds no collections"
        );
      }
      this.name = info.name;
      this.model = info.model;
      this.embeddingSize = info.embedding_size;
      this.chunks = rows(
        db.exec("SELECT content, embedding, metadata FROM " + info.table_name)
      ).map((row) => ({
        text: row.content || "",
        vector: row.embedding ? decodeF32(row.embedding) : null,
        metadata: parseJson(row.metadata),
      }));

      this._docs = this.chunks.map((chunk) => tokenize(chunk.text));
      this._df = {};
      for (const terms of this._docs) {
        for (const term of new Set(terms)) {
          this._df[term] = (this._df[term] || 0) + 1;
        }
      }
    }

    /* The k chunks closest to a query vector, best first. */
    topK(queryVector, k) {
      return this.chunks
        .filter((chunk) => chunk.vector)
        .map((chunk) => ({ chunk: chunk, score: cosine(queryVector, chunk.vector) }))
        .sort((a, b) => b.score - a.score)
        .slice(0, k);
    }

    /* A BM25-flavoured lexical ranking, for browsers without WebGPU (no
       query embedding possible) and as the fallback when a model fails. */
    lexicalTopK(text, k) {
      const queryTerms = tokenize(text);
      const total = this.chunks.length;
      const hits = [];
      this.chunks.forEach((chunk, i) => {
        const terms = this._docs[i];
        const counts = {};
        for (const term of terms) {
          counts[term] = (counts[term] || 0) + 1;
        }
        let score = 0;
        for (const term of queryTerms) {
          const tf = counts[term] || 0;
          if (!tf) continue;
          const idf = Math.log(1 + total / (this._df[term] || 1));
          score += (idf * tf) / (tf + 1.2 * (0.25 + (0.75 * (terms.length || 1)) / 120));
        }
        if (score > 0) hits.push({ chunk: chunk, score: score });
      });
      return hits.sort((a, b) => b.score - a.score).slice(0, k);
    }
  }

  function parseJson(text) {
    if (!text) return {};
    try {
      return JSON.parse(text) || {};
    } catch (_error) {
      return {};
    }
  }

  /* The chat request for a grounded answer: the retrieved passages as the
     model's only source, the recent turns for follow-up questions. */
  function buildMessages(question, hits, history, siteName) {
    const passages = hits
      .map((hit, i) => {
        const m = hit.chunk.metadata;
        const label = [m.title, m.heading].filter(Boolean).join(" › ") || m.url || "";
        return "[" + (i + 1) + "] " + label + "\n" + hit.chunk.text;
      })
      .join("\n\n");
    const system =
      "You are the assistant for the website " + (siteName || "") + ". Answer the" +
      " visitor's question using only the passages below. Be concise and" +
      " concrete; if the passages do not contain the answer, say so and" +
      " suggest where on the site to look. Do not invent facts.\n\nPassages:\n\n" +
      passages;
    return [{ role: "system", content: system }]
      .concat(history || [])
      .concat([{ role: "user", content: question }]);
  }

  /* A markdown sources list for the answer, one link per page, in rank order. */
  function sourcesMarkdown(hits) {
    const seen = new Set();
    const lines = [];
    for (const hit of hits) {
      const m = hit.chunk.metadata;
      if (!m.url || seen.has(m.url)) continue;
      seen.add(m.url);
      lines.push("- [" + (m.title || m.url) + "](" + m.url + ")");
    }
    return lines.length ? "\n\n**Sources**\n" + lines.join("\n") : "";
  }

  /* Passages quoted directly, for the lexical fallback: no model, still
     the site's own words with links. */
  function passagesMarkdown(hits) {
    if (!hits.length) {
      return "I could not find anything about that on this site.";
    }
    const lines = ["Here is what this site says about that:", ""];
    for (const hit of hits) {
      const m = hit.chunk.metadata;
      const body = hit.chunk.text.split("\n\n").slice(1).join(" ").replace(/\s+/g, " ").trim() || hit.chunk.text;
      const snippet = body.length > 320 ? body.slice(0, 320) + "…" : body;
      lines.push("**" + (m.heading || m.title || m.url || "") + "** — " + snippet);
      if (m.url) lines.push("[Read more](" + m.url + ")");
      lines.push("");
    }
    return lines.join("\n").trim();
  }

  return {
    normalizeUrl: normalizeUrl,
    candidateUrls: candidateUrls,
    ArchiveIndex: ArchiveIndex,
    prepareArchivedHtml: prepareArchivedHtml,
    RagIndex: RagIndex,
    decodeF32: decodeF32,
    cosine: cosine,
    queryText: queryText,
    buildMessages: buildMessages,
    sourcesMarkdown: sourcesMarkdown,
    passagesMarkdown: passagesMarkdown,
  };
});
