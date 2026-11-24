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

export interface LLMCallStat {
  id: string;
  call_type: string;
  model: string;
  agent_id: string | null;
  /** The conversation of the run that made the call; not a foreign key. */
  session_id: string | null;
  /** The run that made the call — a workflow's rag_query embeddings included. */
  run_id: string | null;
  response_code: number | null;
  prompt_tokens: number | null;
  completion_tokens: number | null;
  total_tokens: number | null;
  cached_prompt_tokens: number | null;
  reasoning_tokens: number | null;
  duration_seconds: number | null;
  request_data: any | null;
  response_data: any | null;
  created_at: string;
  updated_at: string;
}

/** Narrows the model call list to one agent, conversation or run. */
export interface LLMCallStatFilters {
  agentId?: string;
  sessionId?: string;
  runId?: string;
  /** Case-insensitive text in the model name, the request or the response. */
  search?: string;
  /** Calls recorded at or after this time, ISO 8601. */
  start?: string;
  /** Calls recorded at or before this time, ISO 8601. */
  end?: string;
}

/** Model calls added up; `average_seconds` is null without calls. */
export interface LLMCallTotals {
  calls: number;
  errors: number;
  prompt_tokens: number;
  completion_tokens: number;
  cached_prompt_tokens: number;
  reasoning_tokens: number;
  total_tokens: number;
  average_seconds: number | null;
}

export interface LLMCallModelTotals extends LLMCallTotals {
  model: string;
  call_type: string;
}

/** One day with calls: LLM input and output tokens, and embedding tokens. */
export interface LLMCallDay {
  date: string;
  calls: number;
  prompt_tokens: number;
  completion_tokens: number;
  embedding_tokens: number;
}

/** `GET /projects/{id}/llm-call-stats/summary`: every matching call, not one page. */
export interface LLMCallSummary {
  totals: LLMCallTotals;
  by_model: LLMCallModelTotals[];
  by_day: LLMCallDay[];
}
