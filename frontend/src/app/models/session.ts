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

import { ChatMessage } from './chat-message';
import { Run } from './run';
import { Task } from './task';

export interface SessionSummary {
  session_id: string;
  agent_id: string;
  agent_name: string;
  /** Caller-supplied key. Evaluation runs use `eval:{suite}:{tag}:{case}:{repeat}`. */
  external_id: string | null;
  runs_count: number;
  tasks_count: number;
  messages_count: number;
  errors_count: number;
  first_message: string | null;
  last_message: string | null;
  created_at: string;
  updated_at: string;
}

export interface SessionDetails {
  session_id: string;
  messages: ChatMessage[];
  runs: Run[];
  tasks: Task[];
}
