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

import { Component, computed, input } from '@angular/core';
import { Task } from '../../models/task';
import { formatDuration } from '../../core/format';

/**
 * What a bar stands for, which decides its colour: a step that called a model
 * (lime), a tool (slate), a step of the graph's own control flow (granite),
 * and any step that failed (raspberry), whatever its kind.
 */
export type TaskKind = 'model' | 'tool' | 'flow' | 'failed';

const MODEL_NODES = new Set(['llm', 'agent', 'rag_query']);
const TOOL_NODES = new Set(['function', 'tool_call']);

export function taskKind(task: Task): TaskKind {
  if (task.errors && task.errors.length > 0) return 'failed';
  if (task.tool_uri || TOOL_NODES.has(task.node_type ?? '')) return 'tool';
  if (MODEL_NODES.has(task.node_type ?? '')) return 'model';
  return 'flow';
}

export interface TaskBox {
  id: string;
  name: string;
  kind: TaskKind;
  /** A tool call an agent made, drawn under the step that made it. */
  nested: boolean;
  /** Percentages of the run's span. */
  left: number;
  width: number;
  duration: string;
  tooltip: string;
}

/**
 * The run's steps on one time axis, in execution order. A task row is
 * written when its step finishes, so a bar starts at `created_at` less the
 * step's duration and ends at `created_at`.
 */
@Component({
  selector: 'app-task-timeline-chart',
  standalone: true,
  templateUrl: './task-timeline-chart.html',
  styleUrl: './task-timeline-chart.css',
})
export class TaskTimelineChart {
  readonly tasks = input<Task[]>([]);

  private readonly spans = computed(() =>
    this.tasks().map((task) => {
      const seconds = task.duration_seconds ?? 0;
      const end = new Date(task.created_at).getTime();
      return { task, seconds, start: end - seconds * 1000, end };
    }),
  );

  /** From the first step's start to the last step's end, in seconds. */
  readonly totalSeconds = computed(() => {
    const spans = this.spans();
    if (spans.length === 0) return 0;
    const first = Math.min(...spans.map((s) => s.start));
    const last = Math.max(...spans.map((s) => s.end));
    return (last - first) / 1000;
  });

  readonly boxes = computed<TaskBox[]>(() => {
    const spans = this.spans();
    if (spans.length === 0) return [];
    const first = Math.min(...spans.map((s) => s.start));
    const total = this.totalSeconds() * 1000;
    return spans.map(({ task, seconds, start }) => {
      const name = task.name || task.id;
      const duration = formatDuration(seconds);
      return {
        id: task.id,
        name,
        kind: taskKind(task),
        nested: !!task.parent_task_name,
        left: total === 0 ? 0 : ((start - first) / total) * 100,
        width: total === 0 ? 100 : ((seconds * 1000) / total) * 100,
        duration,
        tooltip: `${name}${task.node_type ? ` (${task.node_type})` : ''}: ${duration}`,
      };
    });
  });

  /** The axis labels: the start, the midpoint and the end of the run. */
  readonly ticks = computed(() => {
    const total = this.totalSeconds();
    return [0, total / 2, total].map((s, i) => (i === 0 ? '0 s' : formatDuration(s)));
  });

  readonly kinds = computed(() => {
    const present = new Set(this.boxes().map((b) => b.kind));
    return LEGEND.filter((entry) => present.has(entry.kind));
  });
}

const LEGEND: { kind: TaskKind; label: string }[] = [
  { kind: 'model', label: 'Model step' },
  { kind: 'tool', label: 'Tool' },
  { kind: 'flow', label: 'Control flow' },
  { kind: 'failed', label: 'Failed' },
];
