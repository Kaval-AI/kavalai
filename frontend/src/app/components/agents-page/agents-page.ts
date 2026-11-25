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
import { Component, DestroyRef, computed, inject, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { DatePipe, DecimalPipe } from '@angular/common';
import { RouterLink } from '@angular/router';
import { Subscription, forkJoin } from 'rxjs';
import { BaseChartDirective } from 'ng2-charts';
import { Chart, registerables } from 'chart.js';
import { AgentService } from '../../services/agent-service';
import { UserService } from '../../services/user-service';
import { Agent } from '../../models/agent';
import { errorMessage } from '../../core/http-error';
import {
  ACTIVITY_SERIES,
  DailyStats,
  TOKEN_SERIES,
  activityBars,
  callDurationLines,
  durationChartOptions,
  hasActivity,
  runDurationLines,
  tokenBars,
} from '../../core/daily-stats';
import {
  BarChart,
  UiAlert,
  UiButton,
  UiCard,
  UiDialog,
  UiEmptyState,
  UiLoading,
  UiPageHeader,
  UiStat,
  UiStats,
  UiTable,
} from '../../ui';
import { JsonTreeComponent } from '../json-tree/json-tree';
import { WorkflowGraphComponent } from '../workflow-graph/workflow-graph';

Chart.register(...registerables);

/** `GET /agents/summary-stats/{project}`: totals over the last 30 days. */
export interface SummaryStats {
  total_sessions: number;
  total_runs: number;
  total_messages: number;
  total_tasks: number;
  total_prompt_tokens: number;
  total_completion_tokens: number;
  total_embedding_tokens: number;
}

/** The JSON documents an agent carries, each opened in a dialog. */
export type AgentDocument = 'workflow' | 'input' | 'output';

const DOCUMENT_TITLES: Record<AgentDocument, string> = {
  workflow: 'Workflow',
  input: 'Input schema',
  output: 'Output schema',
};

/** The days the charts cover. */
const CHART_DAYS = 7;

/**
 * The agents of the active project, as a list beside the selected agent: its
 * workflow graph, its totals over the last 30 days and its activity, tokens
 * and durations per day over the last week.
 */
@Component({
  selector: 'app-agents-page',
  imports: [
    DatePipe,
    DecimalPipe,
    RouterLink,
    BaseChartDirective,
    BarChart,
    UiAlert,
    UiButton,
    UiCard,
    UiDialog,
    UiEmptyState,
    UiLoading,
    UiPageHeader,
    UiStat,
    UiStats,
    UiTable,
    JsonTreeComponent,
    WorkflowGraphComponent,
  ],
  templateUrl: './agents-page.html',
  styleUrl: './agents-page.css',
})
export class AgentsPage {
  private readonly agentService = inject(AgentService);
  private readonly userService = inject(UserService);
  private readonly destroyRef = inject(DestroyRef);

  readonly activitySeries = ACTIVITY_SERIES;
  readonly tokenSeries = TOKEN_SERIES;
  readonly durationOptions = durationChartOptions();
  readonly chartDays = CHART_DAYS;

  readonly projectId = signal<string | null>(null);
  readonly agents = signal<Agent[]>([]);
  readonly loading = signal(false);
  readonly error = signal<string | null>(null);

  readonly selected = signal<Agent | null>(null);
  readonly summary = signal<SummaryStats | null>(null);
  readonly daily = signal<DailyStats | null>(null);
  readonly statsLoading = signal(false);
  readonly statsError = signal<string | null>(null);

  readonly document = signal<AgentDocument | null>(null);
  readonly documentTitle = computed(() => {
    const document = this.document();
    return document ? DOCUMENT_TITLES[document] : '';
  });
  readonly documentData = computed(() => {
    const agent = this.selected();
    const document = this.document();
    const documents = { workflow: agent?.workflow, input: agent?.input_schema, output: agent?.output_schema };
    return document ? documents[document] : null;
  });

  readonly activity = computed(() => {
    const daily = this.daily();
    return daily ? activityBars(daily) : [];
  });
  readonly tokens = computed(() => {
    const daily = this.daily();
    return daily ? tokenBars(daily) : [];
  });
  readonly callDurations = computed(() => {
    const daily = this.daily();
    return daily ? callDurationLines(daily) : null;
  });
  readonly runDurations = computed(() => {
    const daily = this.daily();
    return daily ? runDurationLines(daily) : null;
  });
  readonly active = computed(() => hasActivity(this.daily()));

  private statsSubscription: Subscription | null = null;
  /** Whether the first user has been seen: later emissions reload only on a project change. */
  private started = false;

  constructor() {
    this.userService.userDetails.pipe(takeUntilDestroyed()).subscribe((user) => {
      if (!user) return;
      const projectId =
        user.active_project_id && user.active_project_id !== 'None' ? user.active_project_id : null;
      if (this.started && projectId === this.projectId()) return;
      this.started = true;
      this.projectId.set(projectId);
      this.load();
    });
    this.destroyRef.onDestroy(() => this.statsSubscription?.unsubscribe());
  }

  load(): void {
    const projectId = this.projectId();
    this.selected.set(null);
    this.agents.set([]);
    if (!projectId) return;
    this.loading.set(true);
    this.error.set(null);
    this.agentService
      .getAgentsByProject(projectId)
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (agents) => {
          this.agents.set(agents);
          this.loading.set(false);
          if (agents.length) this.select(agents[0]);
        },
        error: (err) => {
          this.error.set(errorMessage(err, 'Could not load the agents.'));
          this.loading.set(false);
        },
      });
  }

  select(agent: Agent): void {
    this.selected.set(agent);
    this.loadStats();
  }

  loadStats(): void {
    const projectId = this.projectId();
    const agent = this.selected();
    if (projectId && agent) this.fetchStats(projectId, agent);
  }

  private fetchStats(projectId: string, agent: Agent): void {
    this.statsSubscription?.unsubscribe();
    this.summary.set(null);
    this.daily.set(null);
    this.statsLoading.set(true);
    this.statsError.set(null);
    this.statsSubscription = forkJoin({
      summary: this.agentService.getSummaryStats(projectId, agent.id),
      daily: this.agentService.getDailyStats(projectId, CHART_DAYS, agent.id),
    }).subscribe({
      next: ({ summary, daily }) => {
        this.summary.set(summary);
        this.daily.set(daily);
        this.statsLoading.set(false);
      },
      error: (err) => {
        this.statsError.set(errorMessage(err, `Could not load the figures of ${agent.name}.`));
        this.statsLoading.set(false);
      },
    });
  }

  open(document: AgentDocument): void {
    this.document.set(document);
  }

  closeDocument(): void {
    this.document.set(null);
  }
}
