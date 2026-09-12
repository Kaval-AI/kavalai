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

  it('should fetch model calls without filters', () => {
    service.getLLMCallStats('proj123').subscribe(calls => {
      expect(calls).toEqual([]);
    });

    const req = httpMock.expectOne('/api/projects/proj123/llm-call-stats?limit=50&offset=0');
    expect(req.request.method).toBe('GET');
    req.flush([]);
  });

  it('should fetch model calls of one agent, conversation and run', () => {
    service.getLLMCallStats('proj123', 'llm', 20, 40, { agentId: 'a1', sessionId: 's1', runId: 'r1' })
      .subscribe(calls => {
        expect(calls).toEqual([]);
      });

    const req = httpMock.expectOne(
      '/api/projects/proj123/llm-call-stats?limit=20&offset=40&call_type=llm&agent_id=a1&session_id=s1&run_id=r1'
    );
    expect(req.request.method).toBe('GET');
    req.flush([]);
  });
});
