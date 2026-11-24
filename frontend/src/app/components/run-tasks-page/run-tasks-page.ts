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
import { Component, DestroyRef, OnInit, computed, inject, signal } from '@angular/core';
import { DatePipe, DecimalPipe } from '@angular/common';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { ActivatedRoute, Router, RouterLink } from '@angular/router';
import { Subscription } from 'rxjs';
import { AgentService } from '../../services/agent-service';
import { UserService } from '../../services/user-service';
import { Task } from '../../models/task';
import { Run } from '../../models/run';
import { LLMCallStat } from '../../models/llm-call-stat';
import { errorMessage } from '../../core/http-error';
import {
  UiAlert,
  UiBadge,
  UiButton,
  UiCard,
  UiCopyButton,
  UiEmptyState,
  UiLoading,
  UiPageHeader,
  UiStat,
  UiStats,
  UiStatusBadge,
  UiTable,
} from '../../ui';
import { TasksList } from '../tasks-list/tasks-list';
import { TaskTimelineChart } from '../task-timeline-chart/task-timeline-chart';
import { formatDuration, shortId } from '../../core/format';

/**
 * One run of a conversation: its figures, its steps on a timeline and as a
 * list, and the model calls it made. The steps come from the conversation's
 * details, narrowed to this run; the calls from the model call list, filtered
 * by the run.
 */
@Component({
  selector: 'app-run-tasks-page',
  standalone: true,
  imports: [
    DatePipe,
    DecimalPipe,
    RouterLink,
    TasksList,
    TaskTimelineChart,
    UiAlert,
    UiBadge,
    UiButton,
    UiCard,
    UiCopyButton,
    UiEmptyState,
    UiLoading,
    UiPageHeader,
    UiStat,
    UiStats,
    UiStatusBadge,
    UiTable,
  ],
  templateUrl: './run-tasks-page.html',
  styleUrl: './run-tasks-page.css',
})
export class RunTasksPage implements OnInit {
  private route = inject(ActivatedRoute);
  private router = inject(Router);
  private agentService = inject(AgentService);
  private userService = inject(UserService);
  private destroyRef = inject(DestroyRef);

  /** The run's model calls are listed up to this many, oldest first. */
  readonly modelCallsLimit = 100;

  readonly sessionId = signal<string | null>(null);
  readonly runId = signal<string | null>(null);
  readonly projectId = signal<string | null>(null);

  readonly run = signal<Run | null>(null);
  readonly tasks = signal<Task[]>([]);
  readonly loading = signal(false);
  readonly error = signal<string | null>(null);

  readonly modelCalls = signal<LLMCallStat[]>([]);
  readonly modelCallsLoading = signal(false);
  readonly modelCallsError = signal<string | null>(null);

  readonly formatDuration = formatDuration;
  readonly shortId = shortId;

  readonly backLink = computed(() => ['/conversations', this.sessionId() ?? '']);
  readonly failedTasks = computed(() => this.tasks().filter((t) => !!t.errors?.length).length);
  readonly promptTokens = computed(() => this.sum((c) => c.prompt_tokens));
  readonly completionTokens = computed(() => this.sum((c) => c.completion_tokens));
  readonly totalTokens = computed(() => this.sum((c) => c.total_tokens));

  private detailsSubscription: Subscription | null = null;
  private callsSubscription: Subscription | null = null;

  ngOnInit(): void {
    this.route.paramMap.pipe(takeUntilDestroyed(this.destroyRef)).subscribe((params) => {
      this.sessionId.set(params.get('sessionId'));
      this.runId.set(params.get('runId'));
      this.load();
    });
    this.userService.userDetails.pipe(takeUntilDestroyed(this.destroyRef)).subscribe((user) => {
      const projectId =
        user?.active_project_id && user.active_project_id !== 'None' ? user.active_project_id : null;
      if (projectId !== this.projectId()) {
        this.projectId.set(projectId);
        this.load();
      }
    });
    this.destroyRef.onDestroy(() => {
      this.detailsSubscription?.unsubscribe();
      this.callsSubscription?.unsubscribe();
    });
  }

  load(): void {
    this.loadTasks();
    this.loadModelCalls();
  }

  loadTasks(): void {
    const projectId = this.projectId();
    const sessionId = this.sessionId();
    const runId = this.runId();
    if (!projectId || !sessionId || !runId) return;
    this.detailsSubscription?.unsubscribe();
    this.loading.set(true);
    this.error.set(null);
    this.detailsSubscription = this.agentService.getSessionDetails(projectId, sessionId).subscribe({
      next: (details) => {
        this.run.set(details.runs.find((r) => r.id === runId) ?? null);
        this.tasks.set(details.tasks.filter((t) => t.run_id === runId));
        this.loading.set(false);
      },
      error: (err) => {
        this.error.set(errorMessage(err, 'Could not load the run.'));
        this.loading.set(false);
      },
    });
  }

  loadModelCalls(): void {
    const projectId = this.projectId();
    const runId = this.runId();
    if (!projectId || !runId) return;
    this.callsSubscription?.unsubscribe();
    this.modelCallsLoading.set(true);
    this.modelCallsError.set(null);
    this.callsSubscription = this.agentService
      .getLLMCallStats(projectId, undefined, this.modelCallsLimit, 0, { runId })
      .subscribe({
        next: (calls) => {
          this.modelCalls.set([...calls].reverse());
          this.modelCallsLoading.set(false);
        },
        error: (err) => {
          this.modelCalls.set([]);
          this.modelCallsError.set(errorMessage(err, 'Could not load the model calls.'));
          this.modelCallsLoading.set(false);
        },
      });
  }

  /** Opens a call on the Model calls page, narrowed to this run. */
  openCall(call: LLMCallStat): void {
    this.router.navigate(['/llm-call-stats'], {
      queryParams: { run_id: this.runId(), call: call.id },
    });
  }

  callStatus(call: LLMCallStat): string {
    if (call.response_code === null || call.response_code === undefined) return 'unknown';
    return call.response_code >= 200 && call.response_code < 300 ? 'ok' : 'error';
  }

  private sum(pick: (call: LLMCallStat) => number | null): number {
    return this.modelCalls().reduce((total, call) => total + (pick(call) ?? 0), 0);
  }
}
