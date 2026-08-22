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

import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { Agent } from '../models/agent';
import { SessionSummary, SessionDetails } from '../models/session';
import { ChatMessage } from '../models/chat-message';
import { LLMCallStat } from '../models/llm-call-stat';

@Injectable({
  providedIn: 'root'
})
export class AgentService {
  constructor(private http: HttpClient) {}

  getAgentsByProject(projectId: string): Observable<Agent[]> {
    return this.http.get<Agent[]>(`/api/agents/all/${projectId}`);
  }

  /** Render a workflow graph to an SVG diagram on the backend. */
  renderWorkflowSvg(workflow: any): Observable<string> {
    return this.http.post('/api/workflows/render-svg', { workflow }, {
      responseType: 'text',
    });
  }

  getAgentStats(projectId: string, agentId?: string): Observable<any> {
    let url = `/api/agents/stats/${projectId}`;
    if (agentId) {
      url += `?agent_id=${agentId}`;
    }
    return this.http.get<any>(url);
  }

  getSummaryStats(projectId: string, agentId?: string): Observable<any> {
    let url = `/api/agents/summary-stats/${projectId}`;
    if (agentId) {
      url += `?agent_id=${agentId}`;
    }
    return this.http.get<any>(url);
  }

  getDailyStats(projectId: string, days: number = 7, agentId?: string): Observable<any> {
    let url = `/api/agents/stats/${projectId}?days=${days}`;
    if (agentId) {
      url += `&agent_id=${agentId}`;
    }
    return this.http.get<any>(url);
  }

  /**
   * List conversations. `search` matches message content; `externalId` matches
   * the session's caller-supplied key as a prefix — paste
   * `eval:my-suite:pr-412:` to see one experiment's runs, or a full id from a
   * failing case to land on the exact conversation that produced it.
   */
  getSessions(
    projectId: string,
    agentId?: string,
    search?: string,
    startDate?: string,
    endDate?: string,
    limit: number = 50,
    offset: number = 0,
    externalId?: string
  ): Observable<any> {
    let url = `/api/agents/sessions/${projectId}?limit=${limit}&offset=${offset}`;
    if (agentId) {
      url += `&agent_id=${agentId}`;
    }
    if (search) {
      url += `&search=${encodeURIComponent(search)}`;
    }
    if (externalId) {
      url += `&external_id=${encodeURIComponent(externalId)}`;
    }
    if (startDate) {
      url += `&start_date=${startDate}`;
    }
    if (endDate) {
      url += `&end_date=${endDate}`;
    }
    return this.http.get<any>(url);
  }

  getSessionDetails(projectId: string, sessionId: string): Observable<SessionDetails> {
    return this.http.get<SessionDetails>(`/api/agents/sessions/${projectId}/${sessionId}/details`);
  }

  getLLMCallStats(projectId: string, callType?: string, limit: number = 50, offset: number = 0): Observable<LLMCallStat[]> {
    let url = `/api/projects/${projectId}/llm-call-stats?limit=${limit}&offset=${offset}`;
    if (callType) {
      url += `&call_type=${callType}`;
    }
    return this.http.get<LLMCallStat[]>(url);
  }
}
