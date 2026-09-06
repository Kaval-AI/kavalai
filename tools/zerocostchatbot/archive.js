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

  return {
    normalizeUrl: normalizeUrl,
    candidateUrls: candidateUrls,
    ArchiveIndex: ArchiveIndex,
    prepareArchivedHtml: prepareArchivedHtml,
  };
});
