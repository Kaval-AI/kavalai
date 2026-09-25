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
import { SessionDetailPage } from './session-detail-page';
import { AgentService } from '../../services/agent-service';
import { UserService } from '../../services/user-service';
import { NavigationService } from '../../services/navigation-service';
import { ActivatedRoute, Router, convertToParamMap, provideRouter } from '@angular/router';
import { of, BehaviorSubject, throwError } from 'rxjs';
import { ChatMessage } from '../../models/chat-message';
import { Run } from '../../models/run';
import { Task } from '../../models/task';
import { JsonTreeComponent } from '../json-tree/json-tree';

describe('SessionDetailPage', () => {
  let component: SessionDetailPage;
  let fixture: ComponentFixture<SessionDetailPage>;
  let agentServiceSpy: jasmine.SpyObj<AgentService>;
  let userServiceSpy: any;
  let navigationServiceSpy: jasmine.SpyObj<NavigationService>;
  let userDetailsSubject: BehaviorSubject<any>;

  const mockSessionDetails = {
    session_id: 'sess1',
    messages: [
      { id: 'm1', role: 'user', content: 'Hello', created_at: '2026-03-26T10:00:00Z', run_id: 'run1' } as ChatMessage,
      { id: 'm2', role: 'assistant', content: 'Hi there', created_at: '2026-03-26T10:00:05Z', run_id: 'run1' } as ChatMessage,
    ],
    runs: [
      { id: 'run1', session_id: 'sess1', tasks_count: 1, duration_seconds: 4.25, created_at: '2026-03-26T10:00:01Z' } as Run,
    ],
    tasks: [
      { id: 't1', run_id: 'run1', session_id: 'sess1', agent_id: 'a1', name: 'Task 1', created_at: '2026-03-26T10:00:02Z', updated_at: '2026-03-26T10:00:02Z', duration_seconds: 1.5, errors: [], inputs: {}, output: {}, prompt: '' } as Task,
    ]
  };

  beforeEach(async () => {
    agentServiceSpy = jasmine.createSpyObj('AgentService', ['getSessionDetails']);
    navigationServiceSpy = jasmine.createSpyObj('NavigationService', ['setBreadcrumbs']);
    userDetailsSubject = new BehaviorSubject({ active_project_id: 'proj1' });
    userServiceSpy = {
      userDetails: userDetailsSubject.asObservable()
    };

    await TestBed.configureTestingModule({
      imports: [SessionDetailPage, JsonTreeComponent],
      providers: [
        provideRouter([]),
        { provide: AgentService, useValue: agentServiceSpy },
        { provide: UserService, useValue: userServiceSpy },
        { provide: NavigationService, useValue: navigationServiceSpy },
        {
          provide: ActivatedRoute,
          useValue: {
            paramMap: of(convertToParamMap({ sessionId: 'sess1' }))
          }
        }
      ]
    }).compileComponents();

    fixture = TestBed.createComponent(SessionDetailPage);
    component = fixture.componentInstance;
  });

  it('should create', () => {
    agentServiceSpy.getSessionDetails.and.returnValue(of(mockSessionDetails));
    fixture.detectChanges();
    expect(component).toBeTruthy();
  });

  it('should load session details and build timeline', () => {
    agentServiceSpy.getSessionDetails.and.returnValue(of(mockSessionDetails));
    fixture.detectChanges();

    expect(agentServiceSpy.getSessionDetails).toHaveBeenCalledWith('proj1', 'sess1');
    expect(component.runBlocks.length).toBe(1);
    const block = component.runBlocks[0];
    expect(block.run.id).toBe('run1');
    expect(block.timeline.length).toBe(3);

    // Check order: message (10:00:00), task (10:00:02), message (10:00:05)
    expect(component.isChatMessage(block.timeline[0])).toBeTrue();
    expect((block.timeline[0] as ChatMessage).id).toBe('m1');

    expect(component.isTask(block.timeline[1])).toBeTrue();
    expect((block.timeline[1] as Task).id).toBe('t1');

    expect(component.isChatMessage(block.timeline[2])).toBeTrue();
    expect((block.timeline[2] as ChatMessage).id).toBe('m2');

    const header = fixture.nativeElement.textContent as string;
    expect(header).toContain('1 tasks executed');
    expect(header).toContain('4.25s');
  });

  it('should show no duration for a run that recorded none', () => {
    const details = {
      ...mockSessionDetails,
      runs: [{ ...mockSessionDetails.runs[0], duration_seconds: null }]
    };
    agentServiceSpy.getSessionDetails.and.returnValue(of(details));
    fixture.detectChanges();

    expect(fixture.nativeElement.textContent).not.toContain('4.25s');
  });

  it('should handle error when loading details', () => {
    agentServiceSpy.getSessionDetails.and.returnValue(throwError(() => new Error('API Error')));
    spyOn(console, 'error');
    fixture.detectChanges();

    expect(component.error).toBe('Failed to load session details');
    expect(component.loading).toBeFalse();
  });

  it('should identify chat messages and tasks correctly', () => {
    const msg = { role: 'user' } as ChatMessage;
    const task = { run_id: 'r1', agent_id: 'a1' } as Task;

    expect(component.isChatMessage(msg)).toBeTrue();
    expect(component.isChatMessage(task as any)).toBeFalse();

    expect(component.isTask(task)).toBeTrue();
    expect(component.isTask(msg as any)).toBeFalse();
  });

  it('should get task name and errors count', () => {
    const taskWithName = { name: 'My Task', errors: ['err1'] } as Task;
    const taskWithoutName = { id: 't1', name: null, errors: null } as Task;

    expect(component.getTaskName(taskWithName)).toBe('My Task');
    expect(component.getTaskName(taskWithoutName)).toBe('Task');

    expect(component.getTaskErrorsCount(taskWithName)).toBe(1);
    expect(component.getTaskErrorsCount(taskWithoutName)).toBe(0);
  });
});
