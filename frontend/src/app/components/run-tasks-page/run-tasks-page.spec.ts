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
import { RunTasksPage } from './run-tasks-page';
import { AgentService } from '../../services/agent-service';
import { UserService } from '../../services/user-service';
import { ActivatedRoute, Router, convertToParamMap, provideRouter } from '@angular/router';
import { of, BehaviorSubject, throwError } from 'rxjs';
import { Task } from '../../models/task';
import { LLMCallStat } from '../../models/llm-call-stat';
import { TasksList } from '../tasks-list/tasks-list';

describe('RunTasksPage', () => {
  let component: RunTasksPage;
  let fixture: ComponentFixture<RunTasksPage>;
  let agentServiceSpy: jasmine.SpyObj<AgentService>;
  let userServiceSpy: jasmine.SpyObj<UserService>;
  let router: Router;
  let userDetailsSubject: BehaviorSubject<any>;

  const mockTasks: Task[] = [
    { id: 'task1', agent_id: 'agent1', session_id: 'sess1', run_id: 'run1', name: 'Task 1', created_at: new Date().toISOString(), updated_at: new Date().toISOString() } as Task,
    { id: 'task2', agent_id: 'agent1', session_id: 'sess1', run_id: 'run1', name: 'Task 2', created_at: new Date().toISOString(), updated_at: new Date().toISOString() } as Task,
    { id: 'task3', agent_id: 'agent1', session_id: 'sess1', run_id: 'run2', name: 'Task 3', created_at: new Date().toISOString(), updated_at: new Date().toISOString() } as Task
  ];

  beforeEach(async () => {
    agentServiceSpy = jasmine.createSpyObj('AgentService', ['getSessionDetails', 'getLLMCallStats']);
    agentServiceSpy.getLLMCallStats.and.returnValue(of([]));
    userDetailsSubject = new BehaviorSubject({ active_project_id: 'proj1' });
    userServiceSpy = {
      userDetails: userDetailsSubject.asObservable()
    } as any;

    await TestBed.configureTestingModule({
      imports: [RunTasksPage, TasksList],
      providers: [
        provideRouter([]),
        { provide: AgentService, useValue: agentServiceSpy },
        { provide: UserService, useValue: userServiceSpy },
        {
          provide: ActivatedRoute,
          useValue: {
            paramMap: of(convertToParamMap({ sessionId: 'sess1', runId: 'run1' }))
          }
        }
      ]
    }).compileComponents();

    fixture = TestBed.createComponent(RunTasksPage);
    component = fixture.componentInstance;
    router = TestBed.inject(Router);
    spyOn(router, 'navigate');
  });

  describe('model calls', () => {
    const call = (id: string, createdAt: string, totalTokens: number | null): LLMCallStat => ({
      id, call_type: 'llm', model: 'openai/gpt-5-mini', agent_id: 'agent1',
      session_id: 'sess1', run_id: 'run1', response_code: 200,
      prompt_tokens: totalTokens, completion_tokens: null, total_tokens: totalTokens,
      cached_prompt_tokens: null, reasoning_tokens: null, duration_seconds: 0.25,
      request_data: null, response_data: null, created_at: createdAt, updated_at: createdAt
    });

    beforeEach(() => {
      agentServiceSpy.getSessionDetails.and.returnValue(of({ session_id: 'sess1', tasks: mockTasks, messages: [], runs: [] }));
    });

    it('loads the calls of this run only, oldest first', () => {
      agentServiceSpy.getLLMCallStats.and.returnValue(of([
        call('newer', '2026-09-01T10:00:02Z', 30),
        call('older', '2026-09-01T10:00:01Z', null)
      ]));
      fixture.detectChanges();

      expect(agentServiceSpy.getLLMCallStats).toHaveBeenCalledWith('proj1', undefined, 100, 0, { runId: 'run1' });
      expect(component.modelCalls.map(c => c.id)).toEqual(['older', 'newer']);
      expect(component.modelCallTokens).toBe(30);

      const section: HTMLElement = fixture.nativeElement.querySelector('.run-model-calls');
      expect(section.textContent).toContain('Model Calls (2)');
      expect(section.textContent).toContain('30 tokens in total');
      expect(section.textContent).toContain('250 ms');
    });

    it('omits the section when the run made no model call', () => {
      fixture.detectChanges();
      expect(fixture.nativeElement.querySelector('.run-model-calls')).toBeNull();
    });

    it('keeps the tasks when the calls cannot be loaded', () => {
      spyOn(console, 'error');
      agentServiceSpy.getLLMCallStats.and.returnValue(throwError(() => new Error('boom')));
      fixture.detectChanges();

      expect(component.modelCalls).toEqual([]);
      expect(component.tasks.length).toBe(2);
      expect(component.error).toBeNull();
    });

    it('formats a missing duration as a dash', () => {
      expect(component.formatDuration(null)).toBe('-');
    });
  });

  it('should create', () => {
    agentServiceSpy.getSessionDetails.and.returnValue(of({ session_id: 'sess1', tasks: [], messages: [], runs: [] }));
    fixture.detectChanges();
    expect(component).toBeTruthy();
  });

  it('should load tasks for the correct run on init', () => {
    agentServiceSpy.getSessionDetails.and.returnValue(of({ session_id: 'sess1', tasks: mockTasks, messages: [], runs: [] }));
    fixture.detectChanges();

    expect(agentServiceSpy.getSessionDetails).toHaveBeenCalledWith('proj1', 'sess1');
    expect(component.tasks.length).toBe(2);
    expect(component.tasks.every(t => t.run_id === 'run1')).toBeTrue();
  });

  it('should handle error when loading tasks', () => {
    agentServiceSpy.getSessionDetails.and.returnValue(throwError(() => new Error('API Error')));
    spyOn(console, 'error');
    fixture.detectChanges();

    expect(component.error).toBe('Failed to load tasks');
    expect(component.loading).toBeFalse();
  });

  it('should navigate back to conversation details when goBack is called', () => {
    agentServiceSpy.getSessionDetails.and.returnValue(of({ session_id: 'sess1', tasks: [], messages: [], runs: [] }));
    fixture.detectChanges();
    component.sessionId = 'sess1';
    component.goBack();
    expect(router.navigate).toHaveBeenCalledWith(['/conversations', 'sess1']);
  });

  it('should reload tasks when project changes', () => {
    agentServiceSpy.getSessionDetails.and.returnValue(of({ session_id: 'sess1', tasks: mockTasks, messages: [], runs: [] }));
    fixture.detectChanges();
    expect(agentServiceSpy.getSessionDetails).toHaveBeenCalledTimes(1);

    agentServiceSpy.getSessionDetails.calls.reset();
    userDetailsSubject.next({ active_project_id: 'proj2' });

    expect(agentServiceSpy.getSessionDetails).toHaveBeenCalledWith('proj2', 'sess1');
  });
});
