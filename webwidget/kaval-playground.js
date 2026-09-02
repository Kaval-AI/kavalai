/*
 * Kaval.AI playground widget.
 *
 * A single, dependency-free script that runs Python (and Kaval.AI) entirely in
 * the browser via Pyodide, with a WebLLM bridge so "browser/..." models work
 * with no API key, no server and no CORS. It powers three surfaces from one
 * codebase:
 *
 *   - the docs: a "Run in browser" button on code blocks marked
 *     `run-in-browser` opens a shared drawer (KavalPlayground.attachButtons);
 *   - any website: an inline, embeddable widget (KavalPlayground.mount); and
 *   - the standalone webwidget/chat-playground.html, paired with the chat
 *     widget (kaval-chat.js) via KavalPlayground.workflowBridge().
 *
 * Public API (window.KavalPlayground):
 *   configure(opts)        merge config: { pyodideUrl, wheelUrl, models,
 *                          embedModel, webllmUrl, cmBase }
 *   mount(el, opts)        render an inline widget into `el` and return its
 *                          instance ({ setCode, run, ... }); opts:
 *                          { code, examples, showModel, showPackages, title }
 *   attachButtons(opts)    add Run buttons to `.run-in-browser` code blocks
 *   open(code)             load `code` into the shared drawer and open it
 *   bridge()               the WebLLM bridge (window.kavalBrowserLLM)
 *   workflowBridge(opts)   chat bridge to a `workflow` defined in the
 *                          playground -> { send, reset, sessionId } (drives
 *                          the separate chat widget in kaval-chat.js)
 *
 * NOTE: the page must be served over http(s); opening built HTML via a file://
 * path stops the browser from fetching the kavalai wheel.
 */
(function () {
  "use strict";

  // -- Configuration --------------------------------------------------------
  var CONFIG = {
    pyodideUrl: "https://cdn.jsdelivr.net/pyodide/v314.0.0/full/pyodide.js",
    cmBase: "https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.16/",
    webllmUrl: "https://esm.run/@mlc-ai/web-llm",
    wheelUrl: null,
    // In-browser chat models offered in the picker, all q4f32 builds (they run
    // on GPUs without FP16 shaders, unlike q4f16). Curated to models that hold
    // a multi-turn conversation — 1B-class models parrot their context instead
    // of answering, so the smallest offered chat default is Qwen2.5-1.5B. The
    // chosen id is exposed to Python as the KAVAL_BROWSER_MODEL global.
    models: [
      ["Qwen2.5-1.5B-Instruct-q4f32_1-MLC", "Qwen2.5-1.5B · ~1.9 GB VRAM"],
      ["Qwen2.5-3B-Instruct-q4f32_1-MLC", "Qwen2.5-3B · ~2.9 GB VRAM"],
      ["Llama-3.2-3B-Instruct-q4f32_1-MLC", "Llama-3.2-3B · ~3.0 GB VRAM"],
      ["Qwen2.5-0.5B-Instruct-q4f32_1-MLC", "Qwen2.5-0.5B · ~1.1 GB VRAM"],
    ],
    // A small, full-precision (q0f32) embedding model — no FP16 GPU required.
    embedModel: "snowflake-arctic-embed-s-q0f32-MLC-b4",
  };

  // Resolve where this script lives so a relative wheel name (from the docs'
  // generated config) works regardless of the current page's depth.
  var SELF_SRC = (document.currentScript && document.currentScript.src) || "";
  var STATIC_BASE = SELF_SRC.replace(/kaval-playground\.js(\?.*)?$/, "");

  // window.KAVAL_PLAYGROUND_CONFIG is emitted by the Sphinx extension and
  // loaded before this script; merge it in (wheelName is relative to us).
  (function applyHostConfig() {
    var g = window.KAVAL_PLAYGROUND_CONFIG;
    if (!g) return;
    if (g.pyodideUrl) CONFIG.pyodideUrl = g.pyodideUrl;
    if (g.wheelUrl) CONFIG.wheelUrl = g.wheelUrl;
    if (g.wheelName) CONFIG.wheelUrl = STATIC_BASE + g.wheelName;
    if (g.models) CONFIG.models = g.models;
    if (g.embedModel) CONFIG.embedModel = g.embedModel;
  })();

  var LS = {
    model: "kaval-pg-browser-model",
  };

  // Kaval.AI icon mark (matches docs/_static/favicon.svg), inlined.
  var LOGO_SVG =
    '<svg class="kaval-pg-logo" viewBox="0 0 79.375 79.375" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">' +
    '<g transform="translate(-27.011657,-24.494702)">' +
    '<g transform="matrix(0.26458333,0,0,0.26458333,-506.65292,-7.2552973)">' +
    '<path d="m 2317,120 h -300 v 300 h 300 z" fill="#002626"/>' +
    '<path d="m 2145.42,235.72 c -1.16,-1.164 -1.82,-2.752 -1.82,-4.424 V 120 H 2017 v 1.842 L 2138.82,233.9 c 1.29,1.185 2.01,2.857 2.01,4.593 V 420 h 176.15 v -12.679 z" fill="#acc12f"/>' +
    "</g></g></svg>";

  // -- Tiny loaders ---------------------------------------------------------
  function loadScript(src) {
    return new Promise(function (resolve, reject) {
      var s = document.createElement("script");
      s.src = src;
      s.onload = resolve;
      s.onerror = function () {
        reject(new Error("Failed to load script: " + src));
      };
      document.head.appendChild(s);
    });
  }

  function loadCss(href) {
    var link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = href;
    document.head.appendChild(link);
  }

  // -- Shared output sink ---------------------------------------------------
  // One Pyodide interpreter is shared by every widget on the page, but stdout
  // is global, so route it to whichever instance is currently running.
  var activeSink = null;
  function sinkAppend(text, kind) {
    if (activeSink) activeSink.append(text, kind);
  }
  function sinkStatus(text) {
    if (activeSink) activeSink.status(text);
  }

  // -- Pyodide bootstrap (lazy, shared) -------------------------------------
  var pyodidePromise = null;
  function ensurePyodide() {
    if (!pyodidePromise) pyodidePromise = bootPyodide();
    return pyodidePromise;
  }

  async function bootPyodide() {
    sinkStatus("Loading Pyodide…");
    await loadScript(CONFIG.pyodideUrl);
    var indexURL = CONFIG.pyodideUrl.replace(/pyodide\.js(\?.*)?$/, "");
    var pyodide = await loadPyodide({ indexURL: indexURL });

    pyodide.setStdout({ batched: function (s) { sinkAppend(s + "\n"); } });
    pyodide.setStderr({
      batched: function (s) { sinkAppend(s + "\n", "stderr"); },
    });

    sinkStatus("Loading micropip…");
    await pyodide.loadPackage("micropip");
    var micropip = pyodide.pyimport("micropip");

    // Route httpx/requests/urllib through the browser's fetch so Kaval.AI's
    // HTTP tools (web search, RSS, REST) work from Python (subject to CORS).
    // LLM inference itself stays fully in-browser via the WebLLM bridge.
    sinkStatus("Installing pyodide-http…");
    await micropip.install("pyodide-http");
    await pyodide.runPythonAsync("import pyodide_http; pyodide_http.patch_all()");

    if (CONFIG.wheelUrl) {
      sinkStatus("Installing kavalai (downloading dependencies)…");
      await micropip.install(CONFIG.wheelUrl);
    } else {
      sinkAppend(
        "Note: no kavalai wheel is configured, so only plain Python (Pyodide " +
          "stdlib) is available. Build it with `uv build --wheel` and point the " +
          "widget at it.\n",
        "meta"
      );
    }

    pyodide._kavalMicropip = micropip;
    sinkStatus("Ready");
    return pyodide;
  }

  // -- In-browser LLM/embeddings engine (WebLLM) ----------------------------
  // Exposes window.kavalBrowserLLM.{chat,embed} so kavalai's "browser/..."
  // clients run fully client-side over WebGPU. WebLLM and the model are
  // downloaded lazily on first use and cached by the browser.
  var webllmModulePromise = null;
  var enginePromise = null;
  var loadedModel = null;

  function loadWebLLM() {
    if (!webllmModulePromise) webllmModulePromise = import(CONFIG.webllmUrl);
    return webllmModulePromise;
  }

  async function getWebLLMEngine(modelId) {
    if (!navigator.gpu) {
      throw new Error(
        "WebGPU is not available in this browser, so 'browser/...' models " +
          "cannot run. Use a recent Chrome/Edge (or Firefox with " +
          "dom.webgpu.enabled). Verify at chrome://gpu."
      );
    }
    // Reuse the engine while the model is unchanged; rebuild on a switch
    // (e.g. moving from a chat model to an embedding model).
    if (enginePromise && loadedModel === modelId) return enginePromise;
    var webllm = await loadWebLLM();
    var previous = enginePromise;
    loadedModel = modelId;
    var loading = (async function () {
      if (previous) {
        // Two models rarely fit in VRAM together, so release the previous
        // engine's GPU buffers before loading the next one.
        try {
          await (await previous).unload();
        } catch (e) {
          /* the previous engine never loaded or is already unusable */
        }
      }
      return webllm.CreateMLCEngine(modelId, {
        initProgressCallback: function (report) {
          sinkStatus(report && report.text ? report.text : "Loading model…");
        },
      });
    })();
    enginePromise = loading;
    // A failed load (download error, GPU out of memory, FP16 shaders missing)
    // must not be cached, or every retry would replay the stale rejection —
    // recovering used to require a page reload.
    loading.catch(function () {
      if (enginePromise === loading) {
        enginePromise = null;
        loadedModel = null;
      }
    });
    return loading;
  }

  function installBrowserLLMBridge() {
    if (window.kavalBrowserLLM) return;
    window.kavalBrowserLLM = {
      // {model, messages, temperature?, top_p?, response_format?} -> {content, usage} | {error}
      chat: async function (requestJson) {
        try {
          var req =
            typeof requestJson === "string" ? JSON.parse(requestJson) : requestJson;
          var engine = await getWebLLMEngine(req.model);
          var opts = { messages: req.messages, stream: false };
          if (req.temperature != null) opts.temperature = req.temperature;
          if (req.top_p != null) opts.top_p = req.top_p;
          if (req.max_tokens != null) opts.max_tokens = req.max_tokens;
          if (req.response_format) {
            var rf = { type: req.response_format.type || "json_object" };
            if (req.response_format.schema != null) {
              // WebLLM expects the JSON schema as a string.
              rf.schema =
                typeof req.response_format.schema === "string"
                  ? req.response_format.schema
                  : JSON.stringify(req.response_format.schema);
            }
            opts.response_format = rf;
          }
          var reply = await engine.chat.completions.create(opts);
          var choice = reply.choices && reply.choices[0];
          return JSON.stringify({
            content: (choice && choice.message && choice.message.content) || "",
            usage: reply.usage || {},
          });
        } catch (err) {
          return JSON.stringify({ error: String((err && err.message) || err) });
        }
      },
      // {model, input: [texts]} -> {embeddings, usage} | {error}
      embed: async function (requestJson) {
        try {
          var req =
            typeof requestJson === "string" ? JSON.parse(requestJson) : requestJson;
          var engine = await getWebLLMEngine(req.model);
          var reply = await engine.embeddings.create({
            input: req.input,
            model: req.model,
          });
          return JSON.stringify({
            embeddings: (reply.data || []).map(function (d) {
              return d.embedding;
            }),
            usage: reply.usage || {},
          });
        } catch (err) {
          return JSON.stringify({ error: String((err && err.message) || err) });
        }
      },
    };
  }

  // -- CodeMirror bootstrap (lazy, shared) ----------------------------------
  var cmPromise = null;
  function ensureCodeMirror() {
    if (!cmPromise) {
      loadCss(CONFIG.cmBase + "codemirror.min.css");
      loadCss(CONFIG.cmBase + "theme/dracula.min.css");
      cmPromise = loadScript(CONFIG.cmBase + "codemirror.min.js").then(function () {
        return loadScript(CONFIG.cmBase + "mode/python/python.min.js");
      });
    }
    return cmPromise;
  }

  // -- Run core (shared by every instance) ----------------------------------
  var installedPackages = new Set();

  // Expose the chosen in-browser model ids to the running Python. Shared by a
  // normal Run and by the workflow chat bridge so both pick up whatever the
  // toolbar's model picker selected. Kaval.AI runs fully client-side here via
  // the WebLLM bridge, so there are no provider API keys to inject.
  function setModelGlobals(pyodide, ctx) {
    pyodide.globals.set("KAVAL_BROWSER_MODEL", ctx.model || CONFIG.models[0][0]);
    pyodide.globals.set("KAVAL_BROWSER_EMBED_MODEL", ctx.embedModel || CONFIG.embedModel);
  }

  async function runPython(code, ctx) {
    activeSink = ctx.sink;

    // Browsers block fetch() of file:// URLs, so the wheel (and Pyodide
    // packages) can't be loaded from disk. Point the user at a web server.
    if (window.location.protocol === "file:") {
      ctx.sink.append(
        "This page was opened via a file:// path, so the browser won't let the " +
          "playground download the kavalai wheel.\n\nServe the files over HTTP " +
          "instead — e.g. run `python -m http.server` from this folder and open " +
          "the printed http://localhost:… URL.\n",
        "stderr"
      );
      ctx.sink.status("Open over http:// to run");
      return;
    }

    var pyodide = await ensurePyodide();

    // Expose the chosen in-browser model ids to the running code.
    setModelGlobals(pyodide, ctx);

    // Install any extra PyPI packages requested by the widget.
    if (ctx.packages && ctx.packages.length) {
      var micropip = pyodide._kavalMicropip;
      for (var i = 0; i < ctx.packages.length; i++) {
        var name = ctx.packages[i];
        if (installedPackages.has(name)) continue;
        ctx.sink.status("Installing " + name + "…");
        await micropip.install(name);
        installedPackages.add(name);
      }
    }

    // A snippet using the picked chat model would otherwise trigger the model
    // download inside the engine's reply timeout window — load it up front,
    // like the chat bridge does. The embedding model is small enough not to
    // need this.
    if (/KAVAL_BROWSER_MODEL/.test(code)) {
      await getWebLLMEngine(ctx.model || CONFIG.models[0][0]);
    }

    ctx.sink.status("Running…");
    // Auto-load bundled Pyodide packages referenced by imports (numpy, …).
    await pyodide.loadPackagesFromImports(code);
    var result = await pyodide.runPythonAsync(code);
    if (result !== undefined) ctx.sink.append("=> " + result + "\n", "meta");
    ctx.sink.status("Done");
  }

  // -- Workflow chat bridge --------------------------------------------------
  // Lets a separate chat UI (kaval-chat.js) talk to a `workflow` the user has
  // defined and run in the playground. Each turn calls
  // `workflow.run({user_message: ...}, external_id=...)` on the shared Pyodide,
  // reusing one chat id so the engine's history-aware nodes (use_history,
  // on by default) see the whole conversation. The bridge knows nothing about
  // the chat UI — it just exposes a send(message) callback the UI can drive.
  //
  // opts.ensureWorkflow: optional async fn run once if no `workflow` exists yet
  // (e.g. () => playground.run()), so an embedded demo can answer the first
  // message without the user clicking Run ▶ first.

  // Runs one chat turn. Reads the `workflow` global, gives it an AgentService
  // over in-browser SQLite if the author wired none (so history is remembered
  // — the compat sessionmaker uses SQLAlchemy's sync engine, since neither
  // greenlet nor aiosqlite runs under Pyodide), runs it for the message and
  // returns a JSON string of {reply} or {error,detail}.
  var CHAT_TURN_PY =
    "import json as _kaval_json\n" +
    "async def _kaval_chat_turn():\n" +
    "    _wf = globals().get('workflow')\n" +
    "    if _wf is None:\n" +
    "        return _kaval_json.dumps({'error': 'no_workflow'})\n" +
    "    if not hasattr(_wf, 'run'):\n" +
    "        return _kaval_json.dumps({'error': 'not_a_workflow'})\n" +
    "    if getattr(_wf, 'agent_service', None) is None:\n" +
    "        try:\n" +
    "            from kavalai.agent_service import AgentService as _KavalAS\n" +
    "            from kavalai.db import db_manager as _kaval_dbm\n" +
    "            _wf.agent_service = _KavalAS(\n" +
    "                _kaval_dbm.get_sqlite_compat_sessionmaker()\n" +
    "            )\n" +
    "        except Exception:\n" +
    "            pass\n" +
    "    try:\n" +
    "        _state = await _wf.run(\n" +
    "            {_kaval_chat_input_key: _kaval_chat_msg},\n" +
    "            external_id=_kaval_chat_session,\n" +
    "        )\n" +
    "    except Exception as _exc:\n" +
    "        return _kaval_json.dumps({'error': 'run_failed', 'detail': str(_exc)})\n" +
    "    _out = _state.output_data or {}\n" +
    "    _data = _out if isinstance(_out, dict) else None\n" +
    "    if isinstance(_out, dict):\n" +
    "        _reply = _out.get(_kaval_chat_reply_key)\n" +
    "        if _reply is None and _out:\n" +
    "            _reply = _kaval_json.dumps(_out)\n" +
    "    else:\n" +
    "        _reply = str(_out)\n" +
    "    return _kaval_json.dumps(\n" +
    "        {'reply': '' if _reply is None else _reply, 'data': _data}\n" +
    "    )\n" +
    "await _kaval_chat_turn()\n";

  function lsGet(key) {
    try {
      return window.localStorage.getItem(key) || "";
    } catch (e) {
      return "";
    }
  }

  function newSessionId() {
    return (
      "kaval-chat-" +
      Math.random().toString(36).slice(2) +
      Date.now().toString(36)
    );
  }

  function createWorkflowBridge(opts) {
    opts = opts || {};
    var inputKey = opts.inputKey || "user_message";
    var replyKey = opts.replyKey || "agent_response";
    // One id for the life of the conversation; reused across turns so the
    // engine accumulates chat history under it. reset() starts a fresh chat.
    var sessionId = newSessionId();

    // The same in-browser model the playground's picker selected.
    function pickedModel() {
      return lsGet(LS.model) || CONFIG.models[0][0];
    }

    // Run a single turn: refresh the globals the snippet reads, then execute it.
    async function runTurn(pyodide, message) {
      setModelGlobals(pyodide, {
        model: pickedModel(),
        embedModel: CONFIG.embedModel,
      });
      pyodide.globals.set("_kaval_chat_msg", String(message));
      pyodide.globals.set("_kaval_chat_session", sessionId);
      pyodide.globals.set("_kaval_chat_input_key", inputKey);
      pyodide.globals.set("_kaval_chat_reply_key", replyKey);
      return JSON.parse(await pyodide.runPythonAsync(CHAT_TURN_PY));
    }

    async function send(message) {
      if (window.location.protocol === "file:") {
        return {
          error:
            "This page was opened from disk. Serve it over HTTP (e.g. " +
            "`python -m http.server`) so the kavalai wheel can load.",
        };
      }
      var pyodide = await ensurePyodide();

      // Load the model before entering Python: the first use of a model
      // downloads it (minutes on a slow connection), and the engine's response
      // streamer would count that against its inactivity timeout and fail the
      // turn. Loading here keeps the download outside the timed reply.
      try {
        await getWebLLMEngine(pickedModel());
      } catch (err) {
        return { error: String((err && err.message) || err) };
      }

      var data = await runTurn(pyodide, message);
      // If nothing has defined `workflow` yet and the host gave us a way to
      // bootstrap it (e.g. run the editor's code), do that once and retry — so
      // an embedded demo answers the first message without a manual Run ▶.
      if (data.error === "no_workflow" && typeof opts.ensureWorkflow === "function") {
        try {
          await opts.ensureWorkflow();
        } catch (e) {
          /* fall through to the friendly no_workflow message below */
        }
        data = await runTurn(pyodide, message);
      }
      if (data.error === "no_workflow") {
        return {
          error:
            "No `workflow` is defined yet. Assign a workflow (e.g. " +
            "`workflow = WorkflowEngine.from_yaml(...)`) in the playground and " +
            "click Run ▶ first.",
        };
      }
      if (data.error === "not_a_workflow") {
        return {
          error:
            "`workflow` is defined but isn't runnable — it needs an async " +
            "`run(input, external_id=...)` method (a WorkflowEngine).",
        };
      }
      if (data.error) {
        return { error: data.detail || "The workflow run failed." };
      }
      // `data.data` is the workflow's full structured output (e.g. a Reply with
      // `choices`), so a UI can render more than the plain reply text.
      return { reply: data.reply, data: data.data || null };
    }

    return {
      // send(message) -> Promise<{reply} | {error}>
      send: send,
      // Start a new conversation (drops the remembered history).
      reset: function () {
        sessionId = newSessionId();
      },
      sessionId: function () {
        return sessionId;
      },
    };
  }

  // -- A playground instance (one editor + output, drawer or inline) --------
  function modelOptionsHtml() {
    return CONFIG.models
      .map(function (m) {
        return '<option value="' + m[0] + '">' + m[1] + "</option>";
      })
      .join("");
  }

  function createInstance(root, opts) {
    opts = opts || {};
    var variant = opts.variant === "drawer" ? "drawer" : "embed";
    var showModel = opts.showModel !== false;
    var showPackages = !!opts.showPackages;
    var examples = opts.examples || null; // { label: code, ... }
    var title = opts.title || "Run in browser";

    root.className =
      (variant === "drawer" ? "kaval-pg-panel" : "kaval-pg-embed") +
      (root.className ? " " + root.className : "");
    root.setAttribute("role", variant === "drawer" ? "dialog" : "group");
    root.setAttribute("aria-label", "Kaval.AI playground");

    var exampleKeys = examples ? Object.keys(examples) : [];
    var html =
      '<div class="kaval-pg-header">' +
      '<span class="kaval-pg-title">' + LOGO_SVG + "<span>" + esc(title) + "</span></span>" +
      '<span class="kaval-pg-status" data-pg="status">Idle</span>' +
      (variant === "drawer"
        ? '<button class="kaval-pg-icon" data-pg="close" aria-label="Close playground" title="Close (Esc)">✕</button>'
        : "") +
      "</div>";

    html += '<div class="kaval-pg-editor" data-pg="editor"></div>';

    html += '<div class="kaval-pg-toolbar">';
    if (exampleKeys.length) {
      html +=
        '<select class="kaval-pg-select" data-pg="example" title="Load a ready-made example">' +
        exampleKeys
          .map(function (k) {
            return '<option value="' + esc(k) + '">' + esc(k) + "</option>";
          })
          .join("") +
        "</select>";
    }
    if (showModel) {
      html +=
        '<select class="kaval-pg-model" data-pg="model" title="In-browser model used by browser/... examples (exposed to Python as KAVAL_BROWSER_MODEL)">' +
        modelOptionsHtml() +
        "</select>";
    }
    if (showPackages) {
      html +=
        '<input class="kaval-pg-input" data-pg="packages" type="text" placeholder="extra pip packages, comma-separated" title="Extra PyPI packages to install before running">';
    }
    html +=
      '<button class="kaval-pg-run" data-pg="run">Run ▶</button>' +
      '<button class="kaval-pg-ghost" data-pg="reset" title="Restore the original snippet">Reset</button>' +
      '<button class="kaval-pg-ghost" data-pg="clear" title="Clear the output">Clear output</button>' +
      "</div>";

    html +=
      '<div class="kaval-pg-output-label">Output</div>' +
      '<pre class="kaval-pg-output" data-pg="output"></pre>';

    root.innerHTML = html;

    var q = function (sel) { return root.querySelector(sel); };
    var statusEl = q('[data-pg="status"]');
    var outputEl = q('[data-pg="output"]');
    var editorHost = q('[data-pg="editor"]');
    var runBtn = q('[data-pg="run"]');
    var modelSel = q('[data-pg="model"]');
    var packagesInput = q('[data-pg="packages"]');
    var exampleSel = q('[data-pg="example"]');

    var editor = null;
    var originalCode =
      opts.code != null
        ? opts.code
        : examples && exampleKeys.length
        ? examples[exampleKeys[0]]
        : "";

    var sink = {
      append: function (text, kind) {
        var span = document.createElement("span");
        if (kind) span.className = "kaval-pg-" + kind;
        span.textContent = text;
        outputEl.appendChild(span);
        outputEl.scrollTop = outputEl.scrollHeight;
      },
      status: function (text) {
        statusEl.textContent = text;
      },
    };

    function clearOutput() {
      outputEl.textContent = "";
    }

    async function ensureEditor() {
      if (editor) return editor;
      await ensureCodeMirror();
      editor = window.CodeMirror(editorHost, {
        mode: "python",
        theme: "dracula",
        lineNumbers: true,
        indentUnit: 4,
        viewportMargin: Infinity,
        extraKeys: { "Ctrl-Enter": run, "Cmd-Enter": run },
      });
      editor.setValue(originalCode);
      return editor;
    }

    async function run() {
      runBtn.disabled = true;
      clearOutput();
      try {
        await ensureEditor();
        await runPython(editor.getValue(), {
          sink: sink,
          model: modelSel ? modelSel.value : null,
          embedModel: CONFIG.embedModel,
          packages: packagesInput
            ? packagesInput.value.split(",").map(trim).filter(Boolean)
            : [],
        });
      } catch (err) {
        sink.append(String((err && err.message) || err) + "\n", "stderr");
        sink.status("Error");
      } finally {
        runBtn.disabled = false;
      }
    }

    // Wire controls.
    runBtn.addEventListener("click", run);
    q('[data-pg="clear"]').addEventListener("click", clearOutput);
    q('[data-pg="reset"]').addEventListener("click", function () {
      if (editor) editor.setValue(originalCode);
    });
    if (exampleSel) {
      exampleSel.addEventListener("change", function () {
        var code = examples[exampleSel.value] || "";
        originalCode = code;
        if (editor) editor.setValue(code);
      });
    }
    if (modelSel) wirePersist(modelSel, LS.model);

    function refresh() {
      if (editor) {
        editor.refresh();
      }
    }

    var instance = {
      root: root,
      setCode: function (code) {
        originalCode = code;
        if (editor) editor.setValue(code);
        else opts.code = code;
      },
      getCode: function () {
        return editor ? editor.getValue() : originalCode;
      },
      run: run,
      clear: clearOutput,
      refresh: refresh,
      setStatus: sink.status,
    };

    // Drawer-specific open/close behaviour.
    if (variant === "drawer") {
      var closeDrawer = function () {
        root.classList.remove("open");
        document.body.classList.remove("kaval-pg-open");
      };
      instance.close = closeDrawer;
      q('[data-pg="close"]').addEventListener("click", closeDrawer);
      instance.open = async function (code) {
        await ensureEditor();
        if (code != null) instance.setCode(code);
        root.classList.add("open");
        document.body.classList.add("kaval-pg-open");
        sink.status("Idle");
        // CodeMirror must re-measure once it is visible.
        setTimeout(function () {
          editor.refresh();
          editor.focus();
        }, 0);
      };
    } else {
      // Inline widgets show their code immediately.
      ensureEditor().then(refresh);
    }

    return instance;
  }

  function trim(s) {
    return s.trim();
  }

  function esc(s) {
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  function wirePersist(el, storageKey) {
    try {
      // Only restore when a value was saved, so a <select> keeps its default
      // first option instead of blanking out.
      var stored = window.localStorage.getItem(storageKey);
      if (stored) el.value = stored;
    } catch (e) {
      /* localStorage may be unavailable (private mode) — ignore. */
    }
    el.addEventListener("change", function () {
      try {
        window.localStorage.setItem(storageKey, el.value);
      } catch (e) {
        /* ignore */
      }
    });
  }

  // -- The shared docs drawer ------------------------------------------------
  var drawer = null;
  function getDrawer() {
    if (drawer) return drawer;
    var root = document.createElement("div");
    document.body.appendChild(root);
    drawer = createInstance(root, { variant: "drawer", showModel: true });
    return drawer;
  }

  // -- Button injection (docs) ----------------------------------------------
  function attachButtons() {
    // Skip auto-generated source listings (viewcode) — whole modules, not
    // runnable examples.
    if (/\/_modules\//.test(window.location.pathname)) return;

    // Opt-in: only blocks the author marked as browser-runnable get a button.
    // In reStructuredText add `:class: run-in-browser` to the code-block (the
    // class lands on the highlight div); a wrapping `.. container:: run-in-browser`
    // works too. In notebooks, a `run-in-browser` cell tag becomes the
    // `tag_run-in-browser` class on the cell. Examples that need a provider key
    // (openai/gemini) or anything not Pyodide-compatible are left unmarked.
    var blocks = document.querySelectorAll(
      'div[class*="highlight-python"], div[class*="highlight-ipython"]'
    );
    Array.prototype.forEach.call(blocks, function (block) {
      var marked =
        block.classList.contains("run-in-browser") ||
        block.closest(".run-in-browser, .tag_run-in-browser");
      if (!marked) return;
      // A nested `no-run` (class or cell tag) still wins as a hard opt-out.
      if (block.closest(".no-run, .tag_no-run")) return;
      if (block.querySelector(".kaval-pg-run-btn")) return;

      var pre = block.querySelector("pre");
      var code = (pre || block).textContent.replace(/\n+$/, "");
      if (!code.trim()) return;

      var btn = document.createElement("button");
      btn.type = "button";
      btn.className = "kaval-pg-run-btn";
      btn.textContent = "Run in browser ▶";
      btn.addEventListener("click", function () {
        getDrawer().open(code);
      });
      block.appendChild(btn);
      block.classList.add("kaval-pg-has-btn");
    });
  }

  // -- Public API -----------------------------------------------------------
  var KavalPlayground = {
    configure: function (opts) {
      if (opts) Object.assign(CONFIG, opts);
      return KavalPlayground;
    },
    mount: function (el, opts) {
      var root = typeof el === "string" ? document.querySelector(el) : el;
      if (!root) throw new Error("KavalPlayground.mount: container not found");
      return createInstance(root, Object.assign({ variant: "embed" }, opts || {}));
    },
    attachButtons: attachButtons,
    open: function (code) {
      return getDrawer().open(code);
    },
    bridge: function () {
      installBrowserLLMBridge();
      return window.kavalBrowserLLM;
    },
    // Build a chat bridge to a `workflow` defined in the playground; returns
    // { send(message), reset(), sessionId() }. Hand `send` to KavalChat.mount.
    workflowBridge: function (opts) {
      return createWorkflowBridge(opts);
    },
    config: CONFIG,
  };
  window.KavalPlayground = KavalPlayground;

  // -- Auto-init ------------------------------------------------------------
  function init() {
    // Make the WebLLM bridge available before any snippet runs.
    installBrowserLLMBridge();
    attachButtons();
    // Declarative embeds: <div data-kaval-playground>…python…</div>.
    Array.prototype.forEach.call(
      document.querySelectorAll("[data-kaval-playground]"),
      function (el) {
        var code = (el.textContent || "").trim();
        el.textContent = "";
        KavalPlayground.mount(el, { code: code });
      }
    );
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape" && drawer && drawer.root.classList.contains("open")) {
        drawer.close();
      }
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
