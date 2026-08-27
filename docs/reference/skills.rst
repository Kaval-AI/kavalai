Agent skills
============

Kaval.AI ships a set of **skills** for coding agents — short, task-shaped
briefings that teach an agent working in *your* repository how this framework
actually behaves. They matter because an agent's priors come from other
frameworks: without them it writes Jinja2 into a node prompt, inverts a
similarity score, or treats a streamed failure as an HTTP error.

The skills are installed with the package, so there is nothing to download.

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Skill
     - Load it when you are…
   * - ``kavalai``
     - starting out — install, extras, calling a model, structured output, and
       which of the others to read next
   * - ``kavalai-workflows``
     - writing or debugging a workflow graph, in YAML or with
       :class:`~kavalai.WorkflowBuilder`
   * - ``kavalai-tools``
     - giving an agent tools: ``@pythontool`` functions, REST servers, MCP
       servers, and ``allowed_tools``
   * - ``kavalai-serving``
     - serving, deploying, persisting or monitoring a workflow
   * - ``kavalai-rag``
     - indexing documents and retrieving them from a graph
   * - ``kavalai-eval``
     - writing evaluation cases, or testing a workflow without a model

``kavalai-workflows`` and ``kavalai-serving`` carry reference files the agent
reads only once it needs them — the exhaustive node tables, the load-time error
messages, and every environment variable.

Installing them into a project
------------------------------

.. code-block:: console

   $ kavalai-skills install
   installed  .claude/skills/kavalai
   installed  .claude/skills/kavalai-workflows
   ...

``--target`` chooses another directory, and a skill that is already there is
kept rather than overwritten — so your own edits survive an upgrade — unless
you pass ``--force``. Name one or more skills to install just those:

.. code-block:: console

   $ kavalai-skills install kavalai-workflows kavalai-tools --target .agents/skills

``kavalai-skills list`` prints what is bundled, with each skill's description.

The files also sit inside the installed package, at
``kavalai/.agents/skills/<name>/SKILL.md``, which is where an agent that reads
an installed distribution's skills will find them without any install step.

Keeping them honest
-------------------

A skill is documentation an agent reads *instead of* these pages, and it cannot
tell when it has gone stale. ``tests/test_skills.py`` pins the parts a rename
would silently invalidate — every node type, every node input type, every eval
matcher, every key of :class:`~kavalai.eval.EvalCase` and
:class:`~kavalai.eval.EvalSuite`, and every environment variable a skill names
— against the code itself. Renaming any of them fails the suite until the skill
is updated too.
