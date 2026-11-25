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
import { makeTask, settle } from '../../../testing';

describe('TasksList', () => {
  let fixture: ComponentFixture<TasksList>;
  let el: HTMLElement;

  async function render(tasks: Task[], runId = 'r1'): Promise<void> {
    fixture.componentRef.setInput('tasks', tasks);
    fixture.componentRef.setInput('runId', runId);
    fixture.detectChanges();
    await settle(fixture);
  }

  function all(testId: string): HTMLElement[] {
    return Array.from(el.querySelectorAll<HTMLElement>(`[data-testid="${testId}"]`));
  }

  function query(testId: string): HTMLElement | null {
    return el.querySelector<HTMLElement>(`[data-testid="${testId}"]`);
  }

  const trajectory: Task[] = [
    makeTask({
      id: 't0',
      name: 'route',
      node_type: 'switch',
      seq: 0,
      inputs: { expr: 'parsed.intent', value: 'refund' },
      output: { taken: 'handle_refund', matched: true },
      duration_seconds: 0,
    }),
    makeTask({ id: 't1', name: 'research', node_type: 'agent', seq: 1, prompt: 'Find the company.', duration_seconds: 2 }),
    makeTask({
      id: 't2',
      name: 'crawl_url',
      node_type: 'tool_call',
      seq: 2,
      parent_task_name: 'research',
      tool_uri: 'python://webtools.crawl_url',
      inputs: { args: { url: 'https://x' }, step: 0 },
      output: { result: 'ok' },
      duration_seconds: 0.4,
    }),
  ];

  beforeEach(async () => {
    await TestBed.configureTestingModule({ imports: [TasksList] }).compileComponents();
    fixture = TestBed.createComponent(TasksList);
    el = fixture.nativeElement;
  });

  it('says so when the run recorded no steps', async () => {
    await render([]);
    expect(query('tasks-empty')!.textContent).toContain('No steps recorded');
    expect(query('tasks-table')).toBeNull();
  });

  it('lists the steps in execution order with type, tool, status and duration', async () => {
    await render(trajectory);
    const rows = all('task-row');
    expect(rows.length).toBe(3);
    expect(query('tasks-table')!.getAttribute('data-run')).toBe('r1');
    const cells = rows[2].querySelectorAll('td');
    expect(cells[0].textContent!.trim()).toBe('2');
    expect(cells[1].textContent).toContain('crawl_url');
    expect(cells[2].textContent!.trim()).toBe('tool_call');
    expect(cells[3].textContent!.trim()).toBe('webtools.crawl_url');
    expect(cells[3].querySelector('span')!.title).toBe('python://webtools.crawl_url');
    expect(cells[4].textContent!.trim()).toBe('OK');
    expect(cells[5].textContent!.trim()).toBe('400 ms');
    expect(rows[1].querySelectorAll('td')[5].textContent!.trim()).toBe('2.0 s');
  });

  it('indents an agent tool call under the node that made it', async () => {
    await render(trajectory);
    const rows = all('task-row');
    expect(rows[1].querySelector('.task-nested')).toBeNull();
    expect(rows[2].querySelector('.task-nested')).not.toBeNull();
    expect(rows[2].querySelector('.task-nest-mark')!.getAttribute('title')).toBe('Called by research');
  });

  it('names a step without a name', async () => {
    await render([makeTask({ name: null, seq: null, node_type: null })]);
    const cells = all('task-row')[0].querySelectorAll('td');
    expect(cells[0].textContent!.trim()).toBe('');
    expect(cells[1].textContent).toContain('Unnamed step');
    expect(cells[2].textContent!.trim()).toBe('');
  });

  it('renders a branch decision with the value it routed on', async () => {
    // The value is the diagnostic: a mis-route is usually a classifier
    // emitting "Refund" where the graph expects "refund".
    await render(trajectory);
    const summary = query('branch-summary')!;
    expect(summary.textContent).toContain('parsed.intent = "refund" → handle_refund');
    expect(query('no-case-matched')).toBeNull();
  });

  it('flags a switch that matched no case', async () => {
    const unmatched = makeTask({
      node_type: 'switch',
      inputs: { expr: 'parsed.intent', value: 'Refund ' },
      output: { matched: false },
    });
    await render([unmatched, makeTask({ id: 't9', node_type: 'if', inputs: null, output: null })]);
    const summaries = all('branch-summary');
    expect(summaries[0].textContent).toContain('"Refund "');
    expect(summaries[0].textContent).toContain('nowhere');
    expect(query('no-case-matched')!.textContent).toContain('no case matched');
    expect(summaries[1].textContent).toContain('= undefined → nowhere');
  });

  it('opens a step into its prompt, tool, inputs and output', async () => {
    await render(trajectory);
    expect(query('task-detail')).toBeNull();
    const rows = all('task-row');
    expect(rows[1].getAttribute('aria-expanded')).toBe('false');

    rows[1].click();
    await settle(fixture);
    expect(rows[1].getAttribute('aria-expanded')).toBe('true');
    expect(rows[1].classList).toContain('selected');
    const detail = query('task-detail')!;
    expect(query('task-prompt')!.textContent).toBe('Find the company.');
    expect(detail.querySelectorAll('app-json-tree').length).toBe(2);

    rows[2].dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter' }));
    await settle(fixture);
    expect(all('task-detail').length).toBe(2);
    expect(all('task-detail')[1].textContent).toContain('python://webtools.crawl_url');
    expect(all('task-detail')[1].textContent).toContain('"ok"');

    rows[1].click();
    await settle(fixture);
    expect(all('task-detail').length).toBe(1);
    expect(query('task-prompt')).toBeNull();
  });

  it('opens and closes every step', async () => {
    await render(trajectory);
    query('tasks-open-all')!.click();
    await settle(fixture);
    expect(all('task-detail').length).toBe(3);

    query('tasks-close-all')!.click();
    await settle(fixture);
    expect(all('task-detail').length).toBe(0);
  });

  it('opens a failed step with its errors as alerts', async () => {
    const failed = makeTask({ id: 'tf', name: 'charge', errors: ['Card declined', 'Retry failed'] });
    await render([trajectory[1], failed]);
    const rows = all('task-row');
    expect(rows[1].classList).toContain('task-failed');
    expect(rows[1].querySelector('ui-status-badge')!.textContent!.trim()).toBe('Error');
    expect(all('task-detail').length).toBe(1);
    const errors = all('task-error');
    expect(errors.map((e) => e.textContent!.trim())).toEqual(['Card declined', 'Retry failed']);
    expect(errors[0].getAttribute('role')).toBe('alert');
  });

  it('keeps the steps the reader opened when the run is loaded again', async () => {
    await render(trajectory);
    all('task-row')[1].click();
    await settle(fixture);
    await render(trajectory.map((t) => ({ ...t })));
    expect(all('task-row')[1].getAttribute('aria-expanded')).toBe('true');
  });
});
