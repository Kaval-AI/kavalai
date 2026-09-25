"""
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
"""

from typing import Optional, Literal

from pydantic import BaseModel, Field

WorkflowStatus = Literal["pending", "running", "completed", "failed"]


class WorkflowState(BaseModel):
    """Serializable runtime state of a single workflow interaction.

    The state is JSON round-trippable (``to_json`` / ``from_json``) and is
    returned by :meth:`WorkflowEngine.run` so a run can be inspected.

    workflow_name: name of the workflow being executed.
    status: lifecycle status of the run.
    current_node: name of the node about to run / last run.
    trace: ordered list of executed node names.
    data: the run context data (``RunContext.data``) passed through ``to_plain``.
    input_data: the original interaction input.
    output_data: the value of the end node's output variable (when finished).
    error: error message when ``status == 'failed'``.
    invocation_id: short id shared by every log line of this run (for scanning).
    token_usage: aggregate model token counts for the run.
    duration_seconds: wall-clock time of the run once it completed or failed,
    as recorded on the run row; ``None`` while it is in progress.
    run_id / session_id / agent_id: persistence identifiers (string UUIDs).
    """

    workflow_name: str
    status: WorkflowStatus = "pending"
    current_node: Optional[str] = None
    trace: list[str] = Field(default_factory=list)
    data: dict = Field(default_factory=dict)
    input_data: dict = Field(default_factory=dict)
    output_data: Optional[dict] = None
    error: Optional[str] = None
    invocation_id: Optional[str] = None
    token_usage: Optional[dict] = None
    duration_seconds: Optional[float] = None
    run_id: Optional[str] = None
    session_id: Optional[str] = None
    agent_id: Optional[str] = None

    def to_json(self) -> str:
        """Serialize the state to a JSON string."""
        return self.model_dump_json()

    @classmethod
    def from_json(cls, data: str) -> "WorkflowState":
        """Deserialize a :class:`WorkflowState` from a JSON string."""
        return cls.model_validate_json(data)
