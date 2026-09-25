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
import { WorkflowsPage } from './workflows-page';
import { AgentService } from '../../services/agent-service';
import { UserService } from '../../services/user-service';
import { NavigationService } from '../../services/navigation-service';
import { of, BehaviorSubject } from 'rxjs';
import { Agent } from '../../models/agent';
import { SessionDetails } from '../../models/session';
import { Run } from '../../models/run';
import { Task } from '../../models/task';

describe('WorkflowsPage', () => {
  let component: WorkflowsPage;
  let fixture: ComponentFixture<WorkflowsPage>;
  let agentServiceSpy: jasmine.SpyObj<AgentService>;
  let userServiceSpy: jasmine.SpyObj<UserService>;
  let navigationServiceSpy: jasmine.SpyObj<NavigationService>;

  const mockAgents: Agent[] = [
    { id: 'agent1', name: 'Agent 1' } as Agent
  ];

  const mockSessions = {
    sessions: [
      { session_id: 'sess1' }
    ],
    total_count: 1
  };

  const mockSessionDetails: SessionDetails = {
    session_id: 'sess1',
    messages: [],
    runs: [
      { id: 'run1', session_id: 'sess1', created_at: new Date().toISOString() } as Run
    ],
    tasks: [
      { id: 'task1', run_id: 'run1', errors: null } as Task
    ]
  };

  beforeEach(async () => {
    agentServiceSpy = jasmine.createSpyObj('AgentService', ['getAgentsByProject', 'getSessions', 'getSessionDetails']);
    const userDetailsSubject = new BehaviorSubject<any>({ active_project_id: 'proj1' });
    userServiceSpy = {
      userDetails: userDetailsSubject.asObservable()
    } as any;
    navigationServiceSpy = jasmine.createSpyObj('NavigationService', ['setBreadcrumbs']);

    await TestBed.configureTestingModule({
      imports: [WorkflowsPage],
      providers: [
        { provide: AgentService, useValue: agentServiceSpy },
        { provide: UserService, useValue: userServiceSpy },
        { provide: NavigationService, useValue: navigationServiceSpy }
      ]
    }).compileComponents();

    fixture = TestBed.createComponent(WorkflowsPage);
    component = fixture.componentInstance;
  });

  it('should create', () => {
    agentServiceSpy.getAgentsByProject.and.returnValue(of([]));
    fixture.detectChanges();
    expect(component).toBeTruthy();
  });

  it('should load workflows on init', () => {
    agentServiceSpy.getAgentsByProject.and.returnValue(of(mockAgents));
    agentServiceSpy.getSessions.and.returnValue(of(mockSessions));
    agentServiceSpy.getSessionDetails.and.returnValue(of(mockSessionDetails));

    fixture.detectChanges();

    expect(agentServiceSpy.getAgentsByProject).toHaveBeenCalledWith('proj1');
    expect(agentServiceSpy.getSessions).toHaveBeenCalledWith('proj1', 'agent1', undefined, undefined, undefined, 10, 0);
    expect(agentServiceSpy.getSessionDetails).toHaveBeenCalledWith('proj1', 'sess1');
    expect(component.workflows.length).toBe(1);
    expect(component.workflows[0].agent.id).toBe('agent1');
    expect(component.workflows[0].lanes.length).toBe(1);
    expect(component.workflows[0].lanes[0].runs[0].status).toBe('success');
    expect(component.workflows[0].lanes[0].runs[0].tasks.length).toBe(1);
  });

  it('should identify error status if task has errors', () => {
    const errorSessionDetails: SessionDetails = {
      ...mockSessionDetails,
      tasks: [{ id: 'task1', run_id: 'run1', errors: ['Some error'] } as Task]
    };
    agentServiceSpy.getAgentsByProject.and.returnValue(of(mockAgents));
    agentServiceSpy.getSessions.and.returnValue(of(mockSessions));
    agentServiceSpy.getSessionDetails.and.returnValue(of(errorSessionDetails));

    fixture.detectChanges();

    expect(component.workflows[0].lanes[0].runs[0].status).toBe('error');
    expect(component.workflows[0].lanes[0].runs[0].tasks[0].status).toBe('error');
  });

  it('should organize overlapping sessions into multiple lanes', () => {
    const now = new Date();
    const overlappingSessionDetails: SessionDetails = {
      session_id: 'sess1',
      messages: [],
      runs: [
        { id: 'run1', session_id: 'sess1', created_at: now.toISOString() } as Run,
        { id: 'run2', session_id: 'sess1', created_at: new Date(now.getTime() + 1000).toISOString() } as Run
      ],
      tasks: [
        { id: 'task1', run_id: 'run1' } as Task,
        { id: 'task2', run_id: 'run2' } as Task
      ]
    };

    agentServiceSpy.getAgentsByProject.and.returnValue(of(mockAgents));
    agentServiceSpy.getSessions.and.returnValue(of(mockSessions));
    agentServiceSpy.getSessionDetails.and.returnValue(of(overlappingSessionDetails));

    fixture.detectChanges();

    expect(component.workflows[0].lanes.length).toBe(2);
    expect(component.workflows[0].lanes[0].runs[0].run.id).toBe('run1');
    expect(component.workflows[0].lanes[1].runs[0].run.id).toBe('run2');
  });

  it('should keep runs in one lane when the recorded duration shows they did not overlap', () => {
    const now = new Date();
    const sessionDetails: SessionDetails = {
      session_id: 'sess1',
      messages: [],
      runs: [
        { id: 'run1', session_id: 'sess1', duration_seconds: 2, created_at: now.toISOString() } as Run,
        { id: 'run2', session_id: 'sess1', duration_seconds: 2, created_at: new Date(now.getTime() + 5000).toISOString() } as Run
      ],
      tasks: [
        { id: 'task1', run_id: 'run1' } as Task,
        { id: 'task2', run_id: 'run2' } as Task
      ]
    };

    agentServiceSpy.getAgentsByProject.and.returnValue(of(mockAgents));
    agentServiceSpy.getSessions.and.returnValue(of(mockSessions));
    agentServiceSpy.getSessionDetails.and.returnValue(of(sessionDetails));

    fixture.detectChanges();

    expect(component.workflows[0].lanes.length).toBe(1);
    expect(component.workflows[0].lanes[0].runs.map(r => r.run.id)).toEqual(['run1', 'run2']);
  });

  it('should assume a minute for a run without a recorded duration', () => {
    const start = new Date('2026-03-26T10:00:00Z');
    const withDuration = { id: 'r', session_id: 's', duration_seconds: 2.5, created_at: start.toISOString() } as Run;
    const withoutDuration = { id: 'r', session_id: 's', duration_seconds: null, created_at: start.toISOString() } as Run;

    expect(component.runEnd(withDuration)).toBe(start.getTime() + 2500);
    expect(component.runEnd(withoutDuration)).toBe(start.getTime() + 60000);
  });

  it('should select task and show overview', () => {
    agentServiceSpy.getAgentsByProject.and.returnValue(of(mockAgents));
    agentServiceSpy.getSessions.and.returnValue(of(mockSessions));
    agentServiceSpy.getSessionDetails.and.returnValue(of(mockSessionDetails));

    fixture.detectChanges();

    const taskWorkflow = component.workflows[0].lanes[0].runs[0].tasks[0];
    component.selectTask(taskWorkflow, 'sess1');

    expect(component.selectedTask).toBe(taskWorkflow);
    expect(component.selectedSessionId).toBe('sess1');

    component.closeTaskOverview();
    expect(component.selectedTask).toBeNull();
  });
});
