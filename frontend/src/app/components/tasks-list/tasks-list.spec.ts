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
import { TasksList } from './tasks-list';
import { Task } from '../../models/task';
import { By } from '@angular/platform-browser';
import { JsonTreeComponent } from '../json-tree/json-tree';

describe('TasksList', () => {
  let component: TasksList;
  let fixture: ComponentFixture<TasksList>;

  const mockTasks: Task[] = [
    {
      id: 'task1',
      agent_id: 'agent1',
      session_id: 'sess1',
      run_id: 'run1',
      name: 'Task 1',
      prompt: 'Test Prompt',
      inputs: { key: 'value' },
      output: { result: 'ok' },
      errors: [],
      duration_seconds: 1.23,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString()
    } as Task,
    {
      id: 'task2',
      agent_id: 'agent1',
      session_id: 'sess1',
      run_id: 'run1',
      name: 'Task 2',
      prompt: 'Error Prompt',
      inputs: { input: 123 },
      output: null,
      errors: ['Error message 1', 'Error message 2'],
      duration_seconds: 0.5,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString()
    } as Task
  ];

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [TasksList, JsonTreeComponent]
    }).compileComponents();

    fixture = TestBed.createComponent(TasksList);
    component = fixture.componentInstance;
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should display empty state when no tasks', () => {
    component.tasks = [];
    fixture.detectChanges();
    const emptyState = fixture.debugElement.query(By.css('.empty-state'));
    expect(emptyState).toBeTruthy();
    expect(emptyState.nativeElement.textContent).toContain('No tasks recorded');
  });

  it('should display tasks when provided', () => {
    component.tasks = mockTasks;
    fixture.detectChanges();
    const taskItems = fixture.debugElement.queryAll(By.css('.task-item'));
    expect(taskItems.length).toBe(2);
    expect(taskItems[0].nativeElement.textContent).toContain('Task 1');
    expect(taskItems[1].nativeElement.textContent).toContain('Task 2');
  });

  it('should display task details correctly', () => {
    component.tasks = [mockTasks[0]];
    fixture.detectChanges();

    expect(fixture.debugElement.query(By.css('.prompt-content')).nativeElement.textContent).toContain('Test Prompt');
    expect(fixture.debugElement.query(By.css('.task-duration')).nativeElement.textContent).toContain('1.23s');
    expect(fixture.debugElement.query(By.css('.no-errors'))).toBeTruthy();
  });

  it('should display errors when present', () => {
    component.tasks = [mockTasks[1]];
    fixture.detectChanges();

    const errorList = fixture.debugElement.query(By.css('.error-list'));
    expect(errorList).toBeTruthy();
    const errorItems = errorList.queryAll(By.css('li'));
    expect(errorItems.length).toBe(2);
    expect(errorItems[0].nativeElement.textContent).toContain('Error message 1');
    expect(errorItems[1].nativeElement.textContent).toContain('Error message 2');

    const taskItem = fixture.debugElement.query(By.css('.task-item'));
    expect(taskItem.nativeElement.classList).toContain('has-errors');
  });

  // --------------------------------------------------------- the trajectory
  const trajectory: Task[] = [
    {
      id: 't0', agent_id: 'a', session_id: 's', run_id: 'r',
      name: 'route', node_type: 'switch', seq: 0,
      inputs: { expr: 'parsed.intent', value: 'refund' },
      output: { taken: 'handle_refund', matched: true },
      prompt: null, errors: null, duration_seconds: 0,
      created_at: new Date().toISOString(), updated_at: new Date().toISOString()
    } as Task,
    {
      id: 't1', agent_id: 'a', session_id: 's', run_id: 'r',
      name: 'research', node_type: 'agent', seq: 1,
      inputs: null, output: null, prompt: null, errors: null,
      duration_seconds: 2, created_at: new Date().toISOString(),
      updated_at: new Date().toISOString()
    } as Task,
    {
      id: 't2', agent_id: 'a', session_id: 's', run_id: 'r',
      name: 'crawl_url', node_type: 'tool_call', seq: 2,
      parent_task_name: 'research', tool_uri: 'python://webtools.crawl_url',
      inputs: { args: { url: 'https://x' }, step: 0 }, output: { result: 'ok' },
      prompt: null, errors: null, duration_seconds: 0.4,
      created_at: new Date().toISOString(), updated_at: new Date().toISOString()
    } as Task
  ];

  it('indents an agent tool call under the node that made it', () => {
    component.tasks = trajectory;
    fixture.detectChanges();

    const items = fixture.debugElement.queryAll(By.css('.task-item'));
    expect(items[1].nativeElement.classList).not.toContain('tool-call');
    expect(items[2].nativeElement.classList).toContain('tool-call');
    expect(component.isToolCall(trajectory[1])).toBeFalse();
    expect(component.isToolCall(trajectory[2])).toBeTrue();
  });

  it('shows the readable half of a tool URI', () => {
    expect(component.toolName(trajectory[2])).toBe('webtools.crawl_url');
    expect(component.toolName({ tool_uri: 'plain' } as Task)).toBe('plain');
    expect(component.toolName({} as Task)).toBe('');

    component.tasks = trajectory;
    fixture.detectChanges();
    expect(fixture.nativeElement.textContent).toContain('webtools.crawl_url');
  });

  it('renders a branch decision with the value it routed on', () => {
    // The value is the diagnostic: a mis-route is usually a classifier
    // emitting "Refund" where the graph expects "refund".
    component.tasks = trajectory;
    fixture.detectChanges();

    const summary = fixture.debugElement.query(By.css('.branch-summary'));
    expect(summary.nativeElement.textContent).toContain('parsed.intent');
    expect(summary.nativeElement.textContent).toContain('"refund"');
    expect(summary.nativeElement.textContent).toContain('handle_refund');
  });

  it('flags a switch that matched no case', () => {
    const unmatched = {
      ...trajectory[0],
      inputs: { expr: 'parsed.intent', value: 'Refund ' },
      output: { taken: 'fallback', matched: false }
    } as Task;
    expect(component.fellThroughToDefault(unmatched)).toBeTrue();
    expect(component.fellThroughToDefault(trajectory[0])).toBeFalse();
    expect(component.fellThroughToDefault(trajectory[1])).toBeFalse();

    component.tasks = [unmatched];
    fixture.detectChanges();
    expect(fixture.nativeElement.textContent).toContain('no case matched');
  });

  it('shows the execution order', () => {
    component.tasks = trajectory;
    fixture.detectChanges();
    const seqs = fixture.debugElement.queryAll(By.css('.task-seq'));
    expect(seqs.map(s => s.nativeElement.textContent.trim())).toEqual(['0', '1', '2']);
  });

});
