"""The import smoke test for ``kavalai[runtime]``, run inside a fresh venv.

``tests/test_packaging.py::test_the_runtime_extra_serves_a_workflow`` installs
the package with only that extra into an empty environment and runs this file
there, from a directory outside the repository, so the installed wheel is what
gets imported. The packages only the other extras bring — crawl4ai, fastembed,
scikit-learn, authlib — must be absent, and so must psycopg2, which no extra
brings. The script prints ``ok`` when every check has passed.
"""

import importlib.util
import json
import sqlite3
import sys
import tempfile
from pathlib import Path

ABSENT = ("crawl4ai", "fastembed", "sklearn", "authlib", "psycopg2")

WORKFLOW = """
name: smoke
description: one model call
llm_model: openai/stub
data_types:
  input:
    type: object
    properties:
      user_message: {type: string}
  output:
    type: object
    properties:
      agent_response: {type: string}
nodes:
  - {name: start, type: start, next: reply}
  - name: reply
    type: llm
    prompt: hi
    inputs: {input: {type: context, value: input}}
    output: output
    next: end
  - {name: end, type: end, output: output}
"""


def check_other_extras_are_absent():
    present = [name for name in ABSENT if importlib.util.find_spec(name)]
    if present:
        sys.exit(f"kavalai[runtime] pulled in {present}")


def check_the_wheel_is_imported():
    import kavalai

    package = Path(kavalai.__file__).parent
    if "site-packages" not in package.parts:
        sys.exit(f"kavalai was imported from {package}, not the installed wheel")
    if not (package / "default_prompt_template.j2").is_file():
        sys.exit("the wheel carries no default_prompt_template.j2")


def check_a_workflow_is_served():
    from fastapi.testclient import TestClient

    from kavalai import BaseLlmClient
    from kavalai.server import create_agent_app
    from kavalai.workflow import WorkflowEngine

    class StubClient(BaseLlmClient):
        async def _run_chat_completions(self, chat_history, response_model, streamer):
            value = streamer.get_value_streamer(
                "response", response_model=response_model
            )
            reply = {name: "hello" for name in response_model.model_fields}
            await value.stream_partial(json.dumps(reply))
            await value.stream_complete()

    engine = WorkflowEngine.from_yaml(
        WORKFLOW, client_factory=lambda *args, **kwargs: StubClient()
    )
    app = create_agent_app(engine=engine, auth_dependency=lambda: None)
    with TestClient(app) as client:
        response = client.post("/run_agent", json={"data": {"user_message": "hi"}})
    if response.status_code != 200:
        sys.exit(f"/run_agent answered {response.status_code}: {response.text}")
    if response.json()["data"]["agent_response"] != "hello":
        sys.exit(f"unexpected answer: {response.text}")


def check_a_database_is_migrated(directory: Path):
    from kavalai.migrate_db import migrate

    path = directory / "agents.db"
    migrate("agents", uri=f"sqlite:///{path}")
    with sqlite3.connect(path) as connection:
        rows = connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {name for (name,) in rows}
    if not {"agents", "sessions", "alembic_version"} <= tables:
        sys.exit(f"migration left only {sorted(tables)}")


def main():
    check_other_extras_are_absent()
    check_the_wheel_is_imported()
    check_a_workflow_is_served()
    with tempfile.TemporaryDirectory() as directory:
        check_a_database_is_migrated(Path(directory))
    print("ok")


if __name__ == "__main__":
    main()
