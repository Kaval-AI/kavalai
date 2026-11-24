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
import { TaskTimelineChart, taskKind } from './task-timeline-chart';
import { Task } from '../../models/task';
import { makeTask } from '../../../testing';

describe('TaskTimelineChart', () => {
  let fixture: ComponentFixture<TaskTimelineChart>;
  let el: HTMLElement;

  function render(tasks: Task[]): void {
    fixture.componentRef.setInput('tasks', tasks);
    fixture.detectChanges();
  }

  function rows(): HTMLElement[] {
    return Array.from(el.querySelectorAll<HTMLElement>('[data-testid="timeline-row"]'));
  }

  function bar(row: HTMLElement): HTMLElement {
    return row.querySelector<HTMLElement>('.tt-bar')!;
  }

  /** A task row is written when its step ends: `created_at` is the end. */
  const endingAt = (seconds: number, duration: number, overrides: Partial<Task> = {}): Task =>
    makeTask({
      created_at: new Date(Date.UTC(2026, 10, 24, 10, 0, seconds)).toISOString(),
      duration_seconds: duration,
      ...overrides,
    });

  beforeEach(async () => {
    await TestBed.configureTestingModule({ imports: [TaskTimelineChart] }).compileComponents();
    fixture = TestBed.createComponent(TaskTimelineChart);
    el = fixture.nativeElement;
  });

  it('draws nothing without steps', () => {
    render([]);
    expect(el.querySelector('[data-testid="timeline"]')).toBeNull();
    expect(fixture.componentInstance.totalSeconds()).toBe(0);
  });

  it('places each step from its start to its end on the run axis', () => {
    render([
      endingAt(0, 0, { id: 'start', name: 'start', node_type: 'start' }),
      endingAt(5, 5, { id: 'fetch', name: 'fetch', node_type: 'agent' }),
      endingAt(3, 1, { id: 'metric', name: 'metric', node_type: 'tool_call', parent_task_name: 'fetch', tool_uri: 'python://metric' }),
      endingAt(10, 5, { id: 'compose', name: 'compose', node_type: 'llm' }),
    ]);
    const [start, fetch, metric, compose] = rows();
    expect(bar(fetch).style.left).toBe('0%');
    expect(bar(fetch).style.width).toBe('50%');
    expect(bar(metric).style.left).toBe('20%');
    expect(bar(metric).style.width).toBe('10%');
    expect(bar(compose).style.left).toBe('50%');
    expect(bar(start).style.width).toBe('0%');

    expect(metric.querySelector('.tt-label')!.classList).toContain('nested');
    expect(fetch.querySelector('.tt-label')!.textContent!.trim()).toBe('fetch');
    expect(fetch.querySelector('.tt-duration')!.textContent!.trim()).toBe('5.0 s');
    expect(fetch.title).toBe('fetch (agent): 5.0 s');

    const ticks = Array.from(el.querySelectorAll('.tt-ticks span')).map((s) => s.textContent!.trim());
    expect(ticks).toEqual(['0 s', '5.0 s', '10.0 s']);
  });

  it('colours model steps, tools, control flow and failures', () => {
    render([
      endingAt(1, 1, { id: 'a', node_type: 'agent' }),
      endingAt(2, 1, { id: 'b', node_type: 'function' }),
      endingAt(3, 1, { id: 'c', node_type: 'switch' }),
      endingAt(4, 1, { id: 'd', node_type: 'llm', errors: ['Timed out'] }),
    ]);
    expect(rows().map((r) => r.getAttribute('data-kind'))).toEqual(['model', 'tool', 'flow', 'failed']);
    expect(bar(rows()[3]).getAttribute('data-kind')).toBe('failed');
    const legend = Array.from(el.querySelectorAll('[data-testid="timeline-legend"] li')).map((li) =>
      li.textContent!.trim(),
    );
    expect(legend).toEqual(['Model step', 'Tool', 'Control flow', 'Failed']);
  });

  it('lists in the legend only the kinds the run has', () => {
    render([endingAt(1, 1, { node_type: 'rag_query' })]);
    const legend = Array.from(el.querySelectorAll('[data-testid="timeline-legend"] li')).map((li) =>
      li.textContent!.trim(),
    );
    expect(legend).toEqual(['Model step']);
  });

  it('fills the axis when every step took no time', () => {
    render([endingAt(1, 0, { id: 'a', name: null }), endingAt(1, 0, { id: 'b', duration_seconds: null })]);
    expect(rows().map((r) => bar(r).style.width)).toEqual(['100%', '100%']);
    expect(rows()[0].querySelector('.tt-label')!.textContent!.trim()).toBe('a');
    expect(rows()[0].title).toBe('a (agent): 0 ms');
  });

  it('classifies a step by its errors, its tool and its node type', () => {
    expect(taskKind(makeTask({ errors: [], node_type: 'llm' }))).toBe('model');
    expect(taskKind(makeTask({ node_type: null }))).toBe('flow');
    expect(taskKind(makeTask({ node_type: 'end', tool_uri: 'rest://x.y' }))).toBe('tool');
  });

  it('leaves out the node type of a step that has none', () => {
    render([endingAt(1, 1, { name: 'step', node_type: null })]);
    expect(rows()[0].title).toBe('step: 1.0 s');
  });
});
