Chat widget
===========

The chat widget is the page-side half of a deployed agent: a floating
launcher or an inline panel that sends each message to the agent server's
streaming endpoint and renders the reply as it arrives. It consists of two
static files — ``kaval-chatbot.js`` and ``kaval-chatbot.css`` — with no build
step and no dependencies, and it ships inside the ``kavalai`` wheel as the
package ``kavalai.widget``. The in-browser playground of
:doc:`../tutorials/run_in_browser`, which runs a whole workflow inside the
page, is a separate component.

Replies are parsed as Markdown into data and rendered through DOM nodes,
never through ``innerHTML``, so agent output cannot introduce markup. Links
become anchors only for ``http(s)``, ``mailto``, site-relative and fragment
targets; any other scheme stays plain text.

Loading the files
-----------------

From a Python application
^^^^^^^^^^^^^^^^^^^^^^^^^

The files are package data, so a Python host serves the version whose
connector speaks the protocol of the agent server it runs.
``kavalai.widget.widget_dir()`` returns their directory, which FastAPI's
``StaticFiles`` serves as it is:

.. code-block:: python

   from fastapi import FastAPI
   from fastapi.staticfiles import StaticFiles
   from fastapi.testclient import TestClient

   from kavalai.widget import widget_dir

   app = FastAPI()
   app.mount("/widget", StaticFiles(directory=widget_dir()), name="widget")

   client = TestClient(app)
   for name in ("kaval-chatbot.js", "kaval-chatbot.css"):
       response = client.get(f"/widget/{name}")
       print(response.status_code, response.headers["content-type"], name)

.. code-block:: text

   200 text/javascript; charset=utf-8 kaval-chatbot.js
   200 text/css; charset=utf-8 kaval-chatbot.css

The directory also holds the package's ``__init__.py``. A host that serves
exactly the two files uses ``asset_path(name)`` with a ``FileResponse``
route instead: it accepts only the names in ``WIDGET_FILES`` and raises
``ValueError`` for any other. ``widget_dir()`` raises ``RuntimeError`` when
``kavalai`` is imported from a zip archive, where there is no directory to
serve.

From a CDN
^^^^^^^^^^

A page without a Python backend loads the files from jsDelivr, which serves
any tagged revision of the GitHub repository. Releases are tagged
``vX.Y.Z``; pin the tag of the ``kavalai`` release the agent server runs, so
that the connector and the server speak the same protocol:

.. code-block:: text

   https://cdn.jsdelivr.net/gh/Kaval-AI/kaval.ai@v1.1.0/kavalai/widget/kaval-chatbot.js
   https://cdn.jsdelivr.net/gh/Kaval-AI/kaval.ai@v1.1.0/kavalai/widget/kaval-chatbot.css

A branch name in place of the tag would change the files under the page with
every commit. The widget is not published to npm: a second release process
for two files would add a way for the widget and the server to drift apart
and nothing else.

Mounting
--------

``KavalChatbot.mount(options)`` builds the widget and returns an object for
controlling it. Floating mode, the default, places a launcher in the
bottom-right corner that opens a resizable window; below 768 pixels of
viewport width the open window covers the screen.

.. code-block:: html

   <link rel="stylesheet" href="/widget/kaval-chatbot.css">
   <script src="/widget/kaval-chatbot.js"></script>
   <script>
     const widget = KavalChatbot.mount({
       connector: KavalChatbot.agentConnector({ url: "/stream_agent" }),
       texts: { title: "Acme support" },
       suggestions: ["Opening hours?"],
     });
   </script>

Inline mode fills a host element, which must have a height of its own. It has
no launcher, greeting or maximise button:

.. code-block:: javascript

   KavalChatbot.mount({
     connector: KavalChatbot.agentConnector({ url: "/stream_agent" }),
     mode: "inline",
     target: "#support-chat",
   });

.. list-table:: Options of ``mount``
   :header-rows: 1
   :widths: 24 18 58

   * - Option
     - Default
     - Meaning
   * - ``connector``
     - required
     - An ``agentConnector``, or any object with ``send`` (see
       `The connector`_); a bare ``send`` function is accepted too.
   * - ``mode``
     - ``"floating"``
     - ``"floating"`` or ``"inline"``.
   * - ``target``
     - ``document.body``
     - Element or selector the widget is appended to; required inline.
   * - ``texts``
     - ``{}``
     - Replacement strings; see `Texts`_. An unknown key raises.
   * - ``title``, ``greeting``, ``emptyMessage``, ``placeholder``
     - —
     - Shorthands for the ``texts`` keys of the same name, kept from
       earlier releases. An entry in ``texts`` wins over its shorthand.
   * - ``disclosure``
     - ``true``
     - ``false`` removes the disclosure label; see `Disclosure`_.
   * - ``logo``
     - none
     - Image URL shown in the header and on the launcher.
   * - ``suggestions``
     - ``[]``
     - Choice chips offered before the first message.
   * - ``theme``
     - ``{}``
     - ``--kcb-*`` tokens with camelCased keys; see `Theming`_.
   * - ``onFeedback``
     - none
     - ``(runId, vote, comment)``; its presence enables the thumbs. See
       `Feedback`_.
   * - ``feedbackComment``
     - ``false``
     - Offers a comment box after the first vote on an answer.
   * - ``maxRetries``
     - ``3``
     - Retries of a turn the server never started; see `Retries`_.
   * - ``retryDelayMs``
     - ``500``
     - Delay before the first retry; it doubles with each further one.

``mount`` returns ``{root, open, close, toggle, send, reset, destroy,
setStatus, setTheme, on}``. ``send(text)`` sends a message as if typed,
``reset()`` clears the history and starts a new conversation,
``setStatus(text)`` shows a short line under the title (an empty string
hides it), ``setTheme(theme)`` replaces the theme wholesale, and
``on(name, listener)`` subscribes to `Events`_.

The connector
-------------

``KavalChatbot.agentConnector(options)`` speaks the SSE protocol of the agent
server (:doc:`../api/server`): it posts ``{"external_id", "data"}`` to the
stream endpoint and reads the reply out of the ``partial`` frames as they
arrive. ``python -m kavalai.server`` serves that endpoint at
``/stream_agent``, and an application that mounts ``create_agent_router``
under a prefix serves it at ``<prefix>/stream_agent``.

.. list-table:: Options of ``agentConnector``
   :header-rows: 1
   :widths: 20 30 50

   * - Option
     - Default
     - Meaning
   * - ``url``
     - ``"/api/chat/stream_agent"``
     - The stream endpoint. The default is the path of the kaval.ai
       website and is kept for it; other deployments set ``url``.
   * - ``inputKey``
     - ``"message"``
     - Field of the workflow input that receives the message.
   * - ``replyKey``
     - ``"agent_response"``
     - Field of the workflow output that holds the reply.
   * - ``choicesKey``
     - ``"choices"``
     - Field of the workflow output that holds the follow-up choices.
   * - ``storage``
     - ``"session"``
     - Where the conversation id lives; see `Storage`_.
   * - ``storageKey``
     - ``"kavalai-chat-conversation"``
     - The storage key of the conversation id.
   * - ``headers``
     - none
     - An object of request headers, or a function returning one (or a
       promise of one), called before every request.
   * - ``onResponse``
     - none
     - ``(response, connector)``, awaited for every response before its
       status is examined.
   * - ``fetchImpl``
     - ``fetch``
     - The ``fetch`` implementation, for tests.

The returned connector has ``send(text, onPartial)``, ``reset()``, which
starts a new conversation, ``conversationId()`` and
``setConversationId(id)``. A header function is the place for a token that
expires, because it runs again before each message; ``onResponse`` is the
place to adopt a conversation id the server issues, because it sees the
response headers before anything else does:

.. code-block:: javascript

   const connector = KavalChatbot.agentConnector({
     url: "/agents/support/stream_agent",
     headers: async () => ({ Authorization: "Bearer " + (await getToken()) }),
     onResponse: (response, self) => {
       const issued = response.headers.get("X-Conversation");
       if (issued) self.setConversationId(issued);
     },
   });

``setConversationId(null)`` starts a new conversation, as ``reset()`` does.

Other backends
^^^^^^^^^^^^^^

A connector is any object with ``send(text, onPartial)`` returning a promise
of ``{text, choices, runId}``, and optionally ``reset()``. ``onPartial``
receives the reply so far, and an empty string discards what has streamed.
An error thrown as ``KavalChatbot.UserError`` is shown to the visitor
verbatim and never retried; any other error is shown as the ``error`` text,
so internal detail does not reach the page. An error with
``retryable = false`` is not retried either.

Retries
-------

The widget sends a failed turn again only when the server never started it:
the connection failed, the server answered with a status of 500 or above
(503 excepted), or the stream ended before its first frame. It waits
``retryDelayMs`` before the first retry, doubles the delay each time, and
gives up after ``maxRetries``; the typing indicator stays up meanwhile.

Once a frame has arrived, the run exists on the server and its model calls
have been paid for, so a second request would pay for them again. A stream
that breaks or ends without a reply after that point ends the turn with the
``interrupted`` text, and a ``workflow_failed`` event ends it with the
``error`` text. Statuses below 500 are not retried: the server has answered
about the request itself, and a repeat receives the same answer. 429 and 503
are shown as ``rateLimited`` and ``unavailable``.

Texts
-----

Every string the widget shows or announces to assistive technology is a key
of ``texts``, error messages included, so a page is translated by passing
them:

.. code-block:: javascript

   KavalChatbot.mount({
     connector: connector,
     texts: {
       title: "Klienditugi",
       disclosure: "Tehisintellekti assistent",
       placeholder: "Kirjuta sõnum…",
       send: "Saada",
     },
   });

.. list-table:: Keys of ``texts``
   :header-rows: 1
   :widths: 22 38 40

   * - Key
     - Default
     - Where it appears
   * - ``title``
     - ``Chatbot``
     - Window header.
   * - ``disclosure``
     - ``AI assistant``
     - Label under the title; see `Disclosure`_.
   * - ``greeting``
     - ``Hello! Click here for help.``
     - Invitation beside the launcher; ``null`` or ``""`` removes it.
   * - ``emptyMessage``
     - ``Hi! Ask us anything.``
     - History area before the first message.
   * - ``placeholder``
     - ``Type a message...``
     - Placeholder of the message input.
   * - ``inputLabel``
     - ``Message``
     - Accessible name of the message input.
   * - ``send``
     - ``Send``
     - Send button.
   * - ``openChat``
     - ``Open chat``
     - Accessible name of the closed launcher.
   * - ``closeChat``
     - ``Close chat``
     - Accessible name of the open launcher and of the header close
       button on narrow screens.
   * - ``maximize``
     - ``Maximize chat``
     - Accessible name of the maximise button.
   * - ``restore``
     - ``Restore chat size``
     - Accessible name of that button while maximised.
   * - ``typing``
     - ``Assistant is typing``
     - Accessible name of the typing indicator.
   * - ``feedbackLabel``
     - ``Rate this answer``
     - Accessible name of the thumbs group.
   * - ``feedbackUp``
     - ``Good answer``
     - Accessible name and tooltip of the thumbs-up button.
   * - ``feedbackDown``
     - ``Bad answer``
     - Accessible name and tooltip of the thumbs-down button.
   * - ``feedbackPlaceholder``
     - ``What could be better? (optional)``
     - Placeholder and accessible name of the comment box.
   * - ``feedbackSubmit``
     - ``Send``
     - Button of the comment box.
   * - ``feedbackThanks``
     - ``Thank you for the feedback.``
     - Replaces the comment box once a comment is sent.
   * - ``error``
     - ``Something went wrong. Please try again.``
     - Any failure that is not a ``UserError``.
   * - ``rateLimited``
     - ``Too many messages. Please wait a moment and try again.``
     - The server answered 429.
   * - ``unavailable``
     - ``The chat assistant is unavailable right now. Please try again
       later.``
     - The server answered 503.
   * - ``interrupted``
     - ``The answer was interrupted. Please try again.``
     - The stream broke after the run had started.

A ``UserError`` whose ``code`` is ``rateLimited``, ``unavailable`` or
``interrupted`` is shown with the corresponding text rather than its own
message, which is how the connector's errors follow the page's language.

Disclosure
----------

The header carries a label, ``AI assistant`` by default, that tells the
visitor the other party is an AI system. It takes the header's colours, so
it remains legible under any theme. Article 50(1) of the EU AI Act
(Regulation (EU) 2024/1689) requires a system intended to interact with
people to be designed so that they are informed of this, unless it is
obvious from the context. The label is therefore on by default, and a
deployment meets that requirement without configuration.

``texts.disclosure`` changes the wording and ``disclosure: false`` removes
the label. Whether a site's context makes the disclosure unnecessary is a
judgement for the site, which knows its visitors, rather than for the widget.

Storage
-------

The conversation id is the ``external_id`` of every request, so the messages
of one conversation form one session in the agent database. ``storage``
chooses where it is kept:

.. list-table::
   :header-rows: 1
   :widths: 16 26 58

   * - ``storage``
     - Kept in
     - Effect
   * - ``"session"``
     - ``sessionStorage``
     - One conversation per tab; a new tab starts a new one.
   * - ``"local"``
     - ``localStorage``
     - The conversation continues across tabs and later visits.
   * - ``"none"``
     - memory
     - Nothing is written to the device; a reload starts a new
       conversation.

The widget writes nothing else to the visitor's device. Whether a site may
store the id, and for how long, is a consent question that the site answers
when it chooses the mode. Where the chosen area is missing or refuses writes,
as in some private browsing modes, the id is kept in memory for the page's
lifetime.

Feedback
--------

With ``onFeedback`` given, a thumbs-up and a thumbs-down button appear under
every answer that carries a run id, which ``agentConnector`` takes from the
``workflow_started`` event. Without ``onFeedback`` nothing is rendered. The
agent server has no feedback endpoint: feedback is stored by the host,
wherever it keeps its own records, under the run id that identifies the run
in the agent database.

.. code-block:: javascript

   KavalChatbot.mount({
     connector: connector,
     feedbackComment: true,
     onFeedback: (runId, vote, comment) =>
       fetch("/feedback", {
         method: "POST",
         headers: { "Content-Type": "application/json" },
         body: JSON.stringify({ run_id: runId, vote, comment }),
       }),
   });

``vote`` is ``"up"`` or ``"down"`` and ``comment`` is ``null`` for a vote.
Each click calls ``onFeedback`` at once, and the chosen button carries
``aria-pressed="true"``. With ``feedbackComment: true`` a comment box follows
the first vote on an answer; a submitted comment arrives in a second call
with the same run id and the current vote, so a host that keeps one record
per run keeps the latest call. A callback that throws or rejects is logged
to the console and not retried.

Events
------

``widget.on(name, listener)`` subscribes to an event and returns a function
that unsubscribes. An unknown name raises, and a listener that throws is
logged and skipped, so host code cannot leave the widget half-updated.

.. list-table::
   :header-rows: 1
   :widths: 16 36 48

   * - Event
     - Payload
     - Fired when
   * - ``"open"``
     - none
     - The floating window opens.
   * - ``"close"``
     - none
     - The floating window closes.
   * - ``"reply"``
     - ``{message, text, choices, runId}``
     - An answer is complete; ``message`` is what the visitor sent.
   * - ``"error"``
     - ``{error, message, text}``
     - A turn fails; ``text`` is what the visitor was shown.
   * - ``"resize"``
     - ``{width, height, maximized}``
     - The floating window changes size: maximised, restored, dragged, or
       shrunk to fit a smaller viewport.
   * - ``"feedback"``
     - ``{runId, vote, comment}``
     - The visitor votes or sends a comment.

A host that places the widget inside an iframe sizes the frame from
``open``, ``close`` and ``resize``; the message protocol between the frame
and its page belongs to the host:

.. code-block:: javascript

   widget.on("resize", ({ width, height }) =>
     parent.postMessage({ type: "chat-size", width, height }, pageOrigin)
   );

Theming
-------

Every colour, font, size and radius is a custom property on
``.kcb-chatbot``. The ``theme`` option and ``setTheme`` take them with
camelCased keys (``accentContent`` sets ``--kcb-accent-content``); a
stylesheet sets them on ``.kcb-chatbot`` directly. The defaults are the
kaval.ai palette.

.. list-table::
   :header-rows: 1
   :widths: 30 28 42

   * - Token
     - Default
     - Used for
   * - ``--kcb-accent``
     - ``#acc12f``
     - Header, launcher, user messages, buttons.
   * - ``--kcb-accent-content``
     - ``#273e47``
     - Text and icons on the accent colour.
   * - ``--kcb-surface``
     - ``#f4f4f6``
     - Window and input background.
   * - ``--kcb-surface-alt``
     - ``#e5e5e8``
     - Agent messages.
   * - ``--kcb-border``
     - ``#d6d6da``
     - Borders and rules.
   * - ``--kcb-text``
     - ``#273e47``
     - Body text; also the background of code blocks.
   * - ``--kcb-link-hover``
     - ``#809537``
     - Links under the pointer.
   * - ``--kcb-radius``
     - ``0.5rem``
     - Window and message corners.
   * - ``--kcb-radius-small``
     - ``0.25rem``
     - Buttons, input and code corners.
   * - ``--kcb-font-family``
     - ``var(--kcb-font)``
     - Body text.
   * - ``--kcb-font``
     - ``system-ui, sans-serif``
     - The earlier name of the body font, still honoured.
   * - ``--kcb-font-size``
     - ``16px``
     - Base size; every text size in the widget is a multiple of it.
   * - ``--kcb-font-header``
     - ``var(--kcb-font-family)``
     - The title and headings inside answers.
   * - ``--kcb-font-mono``
     - ``ui-monospace, monospace``
     - Code.
   * - ``--kcb-shadow``
     - ``rgba(39, 62, 71, 0.35)``
     - Shadows of the window and launcher.
   * - ``--kcb-z``
     - ``1100``
     - Stacking order of the floating widget.
   * - ``--kcb-top-offset``
     - ``0px``
     - Height of a fixed page header, under which a maximised window
       stops.

Classes
^^^^^^^

The following classes are public: they keep their names and meaning across
minor releases, so a host stylesheet may target them. Every other ``kcb-``
class — resize handles, icons, animation states — is internal.

* Root: ``kcb-chatbot``, with ``kcb-floating`` or ``kcb-inline``;
  ``kcb-open`` while the floating window is open and ``kcb-maximized`` while
  it is maximised.
* Frame: ``kcb-window``, ``kcb-header``, ``kcb-title``,
  ``kcb-title-label``, ``kcb-disclosure``, ``kcb-status``,
  ``kcb-greeting`` and the launcher ``kcb-bubble-float``.
* History: ``kcb-history``, ``kcb-empty``, ``kcb-message`` with
  ``kcb-from-user`` or ``kcb-from-agent``, ``kcb-error`` on an agent message
  that reports a failure, and ``kcb-typing``.
* Feedback: ``kcb-feedback``, ``kcb-feedback-btn`` with ``kcb-feedback-up``
  or ``kcb-feedback-down``, ``kcb-feedback-comment``,
  ``kcb-feedback-submit`` and ``kcb-feedback-thanks``.
* Input: ``kcb-choices``, ``kcb-choice``, ``kcb-input-row``, ``kcb-send``.
* Answers: ``kcb-md-paragraph``, ``kcb-md-heading`` (with ``data-level``),
  ``kcb-md-list``, ``kcb-md-rule``, ``kcb-md-link``, ``kcb-md-strong``,
  ``kcb-md-em``, ``kcb-inline-code`` and ``kcb-code``.

The stylesheet resets the page's button styles inside the widget with
``.kcb-chatbot button`` and styles its own buttons with two classes, as in
``.kcb-chatbot .kcb-send``, so an override of a button needs at least that
specificity.

Testing
-------

The widget's tests run under Node without a browser:

.. code-block:: console

   $ node --test kavalai/widget/tests/kaval-chatbot.test.js

They cover the Markdown parser, the SSE handling and the connector, and
``mount`` over a minimal DOM. They also check that this page documents every
``texts`` key with its default, every event, every token and every class it
names. ``kavalai/widget/preview.html`` shows both modes against a scripted
connector, with ``?theme=dark`` for the dark variant.
