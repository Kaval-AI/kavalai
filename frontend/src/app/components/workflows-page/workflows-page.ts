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
import { formatDuration } from '../../core/format';
import { Component, DestroyRef, inject, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { DatePipe } from '@angular/common';
import { RouterLink } from '@angular/router';
import { Observable, Subscription, forkJoin, of } from 'rxjs';
import { catchError, map, switchMap } from 'rxjs/operators';
import { AgentService } from '../../services/agent-service';
import { UserService } from '../../services/user-service';
import { Agent } from '../../models/agent';
import { SessionDetails, SessionSummary } from '../../models/session';
import { Run } from '../../models/run';
import { Task } from '../../models/task';
import { errorMessage } from '../../core/http-error';
import {
  UiAlert,
  UiButton,
  UiCard,
  UiCopyButton,
  UiDialog,
  UiEmptyState,
  UiInfo,
  UiLoading,
  UiPageHeader,
  UiSpinner,
  UiStatusBadge,
} from '../../ui';
import { JsonTreeComponent } from '../json-tree/json-tree';

/** A task square's state: `ok` unless the task recorded an error. */
export type TaskStatus = 'ok' | 'error';

export interface TaskItem {
  task: Task;
  status: TaskStatus;
}

export interface RunItem {
  run: Run;
  sessionId: string;
  tasks: TaskItem[];
  /** `error` when any task failed, `ok` otherwise; `empty` for a run without tasks. */
  status: TaskStatus | 'empty';
}

/** Runs that do not overlap in time, oldest first. */
export interface Lane {
  runs: RunItem[];
}

export interface AgentRuns {
  agent: Agent;
  lanes: Lane[];
}

/** The conversations read per agent. */
const SESSIONS_PER_AGENT = 10;

/** The duration assumed for a run that recorded none, for the overlap check. */
const ASSUMED_RUN_SECONDS = 60;

export function taskStatus(task: Task): TaskStatus {
  return task.errors && task.errors.length > 0 ? 'error' : 'ok';
}

export function runStatus(tasks: TaskItem[]): RunItem['status'] {
  if (tasks.length === 0) return 'empty';
  return tasks.some((t) => t.status === 'error') ? 'error' : 'ok';
}

/**
 * When a run ended, for the lane overlap check. A run that recorded no
 * duration (still in progress, or written before runs carried one) is
 * assumed to have taken a minute.
 */
export function runEnd(run: Run): number {
  const start = new Date(run.created_at).getTime();
  return start + (run.duration_seconds ?? ASSUMED_RUN_SECONDS) * 1000;
}

/** Each run into the first lane whose last run ended before it started. */
export function organizeIntoLanes(runs: RunItem[]): Lane[] {
  const sorted = [...runs].sort(
    (a, b) => new Date(a.run.created_at).getTime() - new Date(b.run.created_at).getTime()
  );
  const lanes: Lane[] = [];
  for (const item of sorted) {
    const start = new Date(item.run.created_at).getTime();
    const lane = lanes.find((l) => start > runEnd(l.runs[l.runs.length - 1].run));
    if (lane) {
      lane.runs.push(item);
    } else {
      lanes.push({ runs: [item] });
    }
  }
  return lanes;
}

function runsOf(details: SessionDetails, sessionId: string): RunItem[] {
  return details.runs.map((run) => {
    const tasks = details.tasks
      .filter((t) => t.run_id === run.id)
      .map((task) => ({ task, status: taskStatus(task) }));
    return { run, sessionId, tasks, status: runStatus(tasks) };
  });
}

/**
 * The recent runs of every agent of the active project, as lanes of task
 * squares: one card per agent, a lane per set of runs that did not overlap,
 * a square per task, lime when it succeeded and raspberry when it failed. A
 * square opens the task in a dialog, with links to its conversation and run.
 */
@Component({
  selector: 'app-workflows-page',
  imports: [
    DatePipe,
    RouterLink,
    UiAlert,
    UiButton,
    UiCard,
    UiCopyButton,
    UiDialog,
    UiEmptyState,
    UiInfo,
    UiLoading,
    UiPageHeader,
    UiSpinner,
    UiStatusBadge,
    JsonTreeComponent,
  ],
  templateUrl: './workflows-page.html',
  styleUrl: './workflows-page.css',
})
export class WorkflowsPage {
  private readonly agentService = inject(AgentService);
  private readonly userService = inject(UserService);

  readonly sessionsPerAgent = SESSIONS_PER_AGENT;
  readonly formatDuration = formatDuration;

  readonly projectId = signal<string | null>(null);
  readonly workflows = signal<AgentRuns[]>([]);
  readonly loading = signal(false);
  readonly loaded = signal(false);
  readonly error = signal<string | null>(null);
  readonly selected = signal<{ item: TaskItem; run: RunItem } | null>(null);

  private subscription: Subscription | null = null;
  private started = false;

  constructor() {
    this.userService.userDetails.pipe(takeUntilDestroyed()).subscribe((user) => {
      if (!user) return;
      const projectId =
        user.active_project_id && user.active_project_id !== 'None' ? user.active_project_id : null;
      if (this.started && projectId === this.projectId()) return;
      this.started = true;
      this.projectId.set(projectId);
      this.workflows.set([]);
      this.loaded.set(false);
      this.load();
    });
    inject(DestroyRef).onDestroy(() => this.subscription?.unsubscribe());
  }

  load(): void {
    const projectId = this.projectId();
    if (!projectId) return;
    this.subscription?.unsubscribe();
    this.loading.set(true);
    this.error.set(null);
    this.subscription = this.agentService
      .getAgentsByProject(projectId)
      .pipe(
        switchMap((agents) =>
          agents.length ? forkJoin(agents.map((agent) => this.runsOfAgent(projectId, agent))) : of([])
        )
      )
      .subscribe({
        next: (workflows) => {
          this.workflows.set(workflows);
          this.loaded.set(true);
          this.loading.set(false);
        },
        error: (err) => {
          this.error.set(errorMessage(err, 'Could not load the agents.'));
          this.loading.set(false);
        },
      });
  }

  /**
   * The runs of an agent's latest conversations. A conversation that fails to
   * load is left out rather than failing the page; so is an agent whose list
   * of conversations fails, which then shows no runs.
   */
  private runsOfAgent(projectId: string, agent: Agent): Observable<AgentRuns> {
    return this.agentService
      .getSessions(projectId, agent.id, undefined, undefined, undefined, SESSIONS_PER_AGENT, 0)
      .pipe(
        switchMap((page: { sessions: SessionSummary[] }) => {
          if (page.sessions.length === 0) return of([] as RunItem[][]);
          return forkJoin(
            page.sessions.map((s) =>
              this.agentService.getSessionDetails(projectId, s.session_id).pipe(
                map((details) => runsOf(details, s.session_id)),
                catchError(() => of([] as RunItem[]))
              )
            )
          );
        }),
        map((runs) => ({ agent, lanes: organizeIntoLanes(runs.flat()) })),
        catchError(() => of({ agent, lanes: [] }))
      );
  }

  select(item: TaskItem, run: RunItem): void {
    this.selected.set({ item, run });
  }

  close(): void {
    this.selected.set(null);
  }

  shortId(id: string): string {
    return id.slice(0, 8);
  }

  runTitle(item: RunItem): string {
    const started = new Date(item.run.created_at).toLocaleString();
    return `Run started ${started}, ${formatDuration(item.run.duration_seconds)}, ${item.tasks.length} tasks`;
  }

  taskLabel(item: TaskItem): string {
    return `${item.task.name || 'Unnamed task'}: ${item.status === 'ok' ? 'succeeded' : 'failed'}`;
  }
}
