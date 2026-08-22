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
*/

export interface Task {
  id: string;
  agent_id: string | null;
  session_id: string;
  run_id: string;
  inputs: any | null;
  output: any | null;
  name: string | null;
  node_type?: string | null;
  prompt: string | null;
  errors: string[] | null;
  duration_seconds: number | null;
  /** Position in the run's execution order — this is the executed path. */
  seq?: number | null;
  /** Set on the tool calls an agent node made, naming the node that made them. */
  parent_task_name?: string | null;
  /** The tool that ran: `python://store_order`, `rest://billing.refund`. */
  tool_uri?: string | null;
  created_at: string;
  updated_at: string;
}
