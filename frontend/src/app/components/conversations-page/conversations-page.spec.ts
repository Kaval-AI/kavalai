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

import { ComponentFixture, TestBed } from '@angular/core/testing';
import { ConversationsPage } from './conversations-page';
import { AgentService } from '../../services/agent-service';
import { UserService } from '../../services/user-service';
import { ActivatedRoute } from '@angular/router';
import { of, throwError, BehaviorSubject } from 'rxjs';
import { Agent } from '../../models/agent';
import { SessionSummary } from '../../models/session';

describe('ConversationsPage', () => {
  let component: ConversationsPage;
  let fixture: ComponentFixture<ConversationsPage>;
  let agentServiceSpy: jasmine.SpyObj<AgentService>;
  let userServiceSpy: jasmine.SpyObj<UserService>;
  let routeMock: any;

  const mockAgents: Agent[] = [
    { id: 'agent1', name: 'Agent 1' } as Agent,
    { id: 'agent2', name: 'Agent 2' } as Agent
  ];

  const mockSessions: SessionSummary[] = [
    {
      session_id: 'sess1',
      agent_id: 'agent1',
      agent_name: 'Agent 1',
      external_id: null,
      runs_count: 1,
      tasks_count: 2,
      messages_count: 3,
      errors_count: 0,
      first_message: 'Hello',
      last_message: 'Bye',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString()
    }
  ];

  beforeEach(async () => {
    agentServiceSpy = jasmine.createSpyObj('AgentService', ['getAgentsByProject', 'getSessions']);
    const userDetailsSubject = new BehaviorSubject<any>({ active_project_id: 'proj1' });
    userServiceSpy = {
      getActiveProjectId: jasmine.createSpy('getActiveProjectId'),
      userDetails: userDetailsSubject.asObservable()
    } as any;
    routeMock = {
      queryParams: of({})
    };

    await TestBed.configureTestingModule({
      imports: [ConversationsPage],
      providers: [
        { provide: AgentService, useValue: agentServiceSpy },
        { provide: UserService, useValue: userServiceSpy },
        { provide: ActivatedRoute, useValue: routeMock }
      ]
    }).compileComponents();

    fixture = TestBed.createComponent(ConversationsPage);
    component = fixture.componentInstance;
  });

  it('should create', () => {
    agentServiceSpy.getAgentsByProject.and.returnValue(of([]));
    agentServiceSpy.getSessions.and.returnValue(of({ sessions: [], total_count: 0 }));
    fixture.detectChanges();
    expect(component).toBeTruthy();
  });

  it('loadSessions', () => {
    agentServiceSpy.getSessions.and.returnValue(of({ sessions: [], total_count: 0 }));
    component.activeProjectId = 'proj1';
    component.loadSessions();
    expect(agentServiceSpy.getSessions).toHaveBeenCalledWith('proj1', undefined, undefined, jasmine.any(String), jasmine.any(String), 20, 0, undefined);
  });

  it('should load agents and sessions on init', () => {
    agentServiceSpy.getAgentsByProject.and.returnValue(of(mockAgents));
    agentServiceSpy.getSessions.and.returnValue(of({ sessions: mockSessions, total_count: 1 }));

    fixture.detectChanges();

    expect(agentServiceSpy.getAgentsByProject).toHaveBeenCalledWith('proj1');
    expect(agentServiceSpy.getSessions).toHaveBeenCalledWith('proj1', undefined, undefined, jasmine.any(String), jasmine.any(String), 20, 0, undefined);
    expect(component.agents).toEqual(mockAgents);
    expect(component.sessions).toEqual(mockSessions);
    expect(component.totalSessions).toBe(1);
  });

  it('should filter by agent when agent selection changes', () => {
    agentServiceSpy.getAgentsByProject.and.returnValue(of(mockAgents));
    agentServiceSpy.getSessions.and.returnValue(of({ sessions: mockSessions, total_count: 1 }));

    fixture.detectChanges();

    component.selectedAgentId = 'agent1';
    component.onFilterChange();

    expect(agentServiceSpy.getSessions).toHaveBeenCalledWith('proj1', 'agent1', undefined, jasmine.any(String), jasmine.any(String), 20, 0, undefined);
  });

  it('should set hasMore to true if exactly limit sessions are returned', () => {
    agentServiceSpy.getAgentsByProject.and.returnValue(of(mockAgents));

    // Create exactly 'limit' (20) sessions
    const twentySessions = Array(20).fill(mockSessions[0]);
    agentServiceSpy.getSessions.and.returnValue(of({ sessions: twentySessions, total_count: 20 }));

    fixture.detectChanges();

    expect(component.hasMore).toBeTrue();
  });

  it('should load more sessions when nextPage is called', () => {
    agentServiceSpy.getAgentsByProject.and.returnValue(of(mockAgents));

    // Initial load returns exactly 'limit' sessions so hasMore remains true
    const twentySessions = Array(20).fill(mockSessions[0]);
    agentServiceSpy.getSessions.and.returnValue(of({ sessions: twentySessions, total_count: 21 }));

    fixture.detectChanges();

    expect(component.hasMore).toBeTrue();

    // Reset spy to track second call
    agentServiceSpy.getSessions.calls.reset();
    agentServiceSpy.getSessions.and.returnValue(of({
      sessions: [{ ...mockSessions[0], session_id: 'sess21' }],
      total_count: 21
    }));

    component.nextPage();

    expect(component.offset).toBe(20);
    expect(agentServiceSpy.getSessions).toHaveBeenCalledWith('proj1', undefined, undefined, jasmine.any(String), jasmine.any(String), 20, 20, undefined);
    expect(component.sessions.length).toBe(21);
    expect(component.totalSessions).toBe(21);
  });

  it('should handle error when loading sessions', () => {
    agentServiceSpy.getAgentsByProject.and.returnValue(of(mockAgents));
    agentServiceSpy.getSessions.and.returnValue(throwError(() => new Error('API Error')));

    spyOn(console, 'error');
    fixture.detectChanges();

    expect(component.error).toBe('Failed to load sessions');
    expect(component.loading).toBeFalse();
  });

  it('should set hasMore to false if fewer than limit sessions are returned', () => {
    agentServiceSpy.getAgentsByProject.and.returnValue(of(mockAgents));
    agentServiceSpy.getSessions.and.returnValue(of({ sessions: mockSessions, total_count: 1 })); // length 1 < limit 20

    fixture.detectChanges();

    expect(component.hasMore).toBeFalse();
  });

  it('should format date to DD-MM-YYYY HH:mm', () => {
    const dateStr = '2023-12-31T23:59:59Z';
    // Note: The result might depend on the timezone of the environment where tests run.
    // However, 2023-12-31T23:59:59Z should always result in 31/12/2023 or 01/01/2024 depending on TZ.
    // Let's use a more stable test that doesn't depend on TZ if possible, or just check the format.
    const result = component.formatDate(dateStr);
    expect(result).toMatch(/^\d{2}-\d{2}-\d{4} \d{2}:\d{2}$/);
  });

  it('should set selectedAgentId from query parameters', () => {
    routeMock.queryParams = of({ agentId: 'agent-789' });
    agentServiceSpy.getAgentsByProject.and.returnValue(of(mockAgents));
    agentServiceSpy.getSessions.and.returnValue(of({ sessions: mockSessions, total_count: 1 }));

    component.ngOnInit();

    expect(component.selectedAgentId).toBe('agent-789');
    expect(agentServiceSpy.getSessions).toHaveBeenCalledWith('proj1', 'agent-789', undefined, jasmine.any(String), jasmine.any(String), 20, 0, undefined);
  });
});
