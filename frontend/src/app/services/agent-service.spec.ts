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

import { TestBed } from '@angular/core/testing';
import { HttpClientTestingModule, HttpTestingController } from '@angular/common/http/testing';
import { AgentService } from './agent-service';
import { Agent } from '../models/agent';

describe('AgentService', () => {
  let service: AgentService;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      imports: [HttpClientTestingModule],
      providers: [AgentService]
    });
    service = TestBed.inject(AgentService);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    httpMock.verify();
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });

  it('should fetch agents by project id', () => {
    const mockAgents: Agent[] = [
      { id: '1', name: 'Agent 1' } as Agent,
      { id: '2', name: 'Agent 2' } as Agent
    ];
    const projectId = 'proj123';

    service.getAgentsByProject(projectId).subscribe(agents => {
      expect(agents.length).toBe(2);
      expect(agents).toEqual(mockAgents);
    });

    const req = httpMock.expectOne(`/api/agents/all/${projectId}`);
    expect(req.request.method).toBe('GET');
    req.flush(mockAgents);
  });

  it('should fetch model calls, leaving the unset filters out', () => {
    service.getLLMCallStats('proj123').subscribe((calls) => expect(calls).toEqual([]));
    const req = httpMock.expectOne((r) => r.url === '/api/projects/proj123/llm-call-stats');
    expect(req.request.method).toBe('GET');
    expect(req.request.params.keys().sort()).toEqual(['limit', 'offset']);
    expect(req.request.params.get('limit')).toBe('50');
    req.flush([]);
  });

  it('should fetch model calls of one agent, conversation and run, by text and time', () => {
    service
      .getLLMCallStats('proj123', 'llm', 20, 40, {
        agentId: 'a1',
        sessionId: 's1',
        runId: 'r1',
        search: 'claims',
        start: '2026-11-23T00:00:00.000Z',
        end: '2026-11-24T23:59:00.000Z',
      })
      .subscribe((calls) => expect(calls).toEqual([]));
    const req = httpMock.expectOne((r) => r.url === '/api/projects/proj123/llm-call-stats');
    const params = req.request.params;
    expect(params.get('limit')).toBe('20');
    expect(params.get('offset')).toBe('40');
    expect(params.get('call_type')).toBe('llm');
    expect(params.get('agent_id')).toBe('a1');
    expect(params.get('session_id')).toBe('s1');
    expect(params.get('run_id')).toBe('r1');
    expect(params.get('search')).toBe('claims');
    expect(params.get('start_date')).toBe('2026-11-23T00:00:00.000Z');
    expect(params.get('end_date')).toBe('2026-11-24T23:59:00.000Z');
    req.flush([]);
  });

  it('should fetch the token usage summary with the same filters', () => {
    const summary = { totals: { calls: 1 }, by_model: [], by_day: [] };
    service
      .getLLMCallSummary('proj123', 'embedding', { search: 'x' })
      .subscribe((s) => expect(s as unknown).toEqual(summary));
    const req = httpMock.expectOne((r) => r.url === '/api/projects/proj123/llm-call-stats/summary');
    expect(req.request.params.get('call_type')).toBe('embedding');
    expect(req.request.params.get('search')).toBe('x');
    expect(req.request.params.has('limit')).toBeFalse();
    req.flush(summary);
  });
});
