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
import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Task } from '../../models/task';
import { JsonTreeComponent } from '../json-tree/json-tree';

@Component({
  selector: 'app-tasks-list',
  standalone: true,
  imports: [CommonModule, JsonTreeComponent],
  templateUrl: './tasks-list.html',
  styleUrl: './tasks-list.css',
})
export class TasksList {
  @Input() tasks: Task[] = [];
  @Input() runId: string = '';

  formatDate(dateStr: string): string {
    const date = new Date(dateStr);
    return date.toLocaleString();
  }

  /**
   * Whether this row is a tool call an agent made, rather than a node of the
   * graph. Those are indented under the node that made them.
   */
  isToolCall(task: Task): boolean {
    return !!task.parent_task_name;
  }

  /** Whether this row records a routing decision rather than work. */
  isBranch(task: Task): boolean {
    return task.node_type === 'if' || task.node_type === 'switch';
  }

  /**
   * A branch decision in one line: `parsed.intent = "refund" -> handle_refund`.
   *
   * The evaluated value is the diagnostic. Nine times in ten a mis-route is
   * not a routing bug, it is the upstream classifier emitting `"Refund"` or
   * `"refund "` — and reading that here turns a two-hour investigation into a
   * glance.
   */
  branchSummary(task: Task): string {
    const expr = task.inputs?.['expr'] ?? '';
    const value = JSON.stringify(task.inputs?.['value']);
    const taken = task.output?.['taken'] ?? 'nowhere';
    return `${expr} = ${value} \u2192 ${taken}`;
  }

  /** A `switch` that matched no case fell through to its default. */
  fellThroughToDefault(task: Task): boolean {
    return this.isBranch(task) && task.output?.['matched'] === false;
  }

  /** The readable half of a tool URI: `python://mail.send` -> `mail.send`. */
  toolName(task: Task): string {
    const uri = task.tool_uri ?? '';
    const index = uri.indexOf('://');
    return index >= 0 ? uri.slice(index + 3) : uri;
  }
}
