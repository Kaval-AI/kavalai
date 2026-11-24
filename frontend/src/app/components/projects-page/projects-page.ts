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
import { DecimalPipe } from '@angular/common';
import { Component, DestroyRef, OnInit, computed, inject, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { RouterLink } from '@angular/router';
import { Chart, registerables } from 'chart.js';
import { BaseChartDirective } from 'ng2-charts';
import { forkJoin } from 'rxjs';
import { distinctUntilChanged, map } from 'rxjs/operators';
import {
  ACTIVITY_SERIES,
  DailyStats,
  TOKEN_SERIES,
  activityBars,
  callDurationLines,
  durationChartOptions,
  runDurationLines,
  tokenBars,
} from '../../core/daily-stats';
import { errorMessage } from '../../core/http-error';
import { Project } from '../../models/project';
import { UserDetails } from '../../models/user-details';
import { AgentService } from '../../services/agent-service';
import { ProjectService } from '../../services/project-service';
import { UserService } from '../../services/user-service';
import {
  BarChart,
  UiAlert,
  UiBadge,
  UiButton,
  UiCard,
  UiCopyButton,
  UiEmptyState,
  UiLoading,
  UiPageHeader,
  UiSpinner,
  UiStat,
  UiStats,
} from '../../ui';

Chart.register(...registerables);

/** The days the charts cover, today included. */
const CHART_DAYS = 7;

/** What `GET /agents/summary-stats/{project}` answers. */
export interface SummaryStats {
  total_sessions: number;
  total_messages: number;
  total_tasks: number;
  total_runs?: number | null;
  total_prompt_tokens?: number | null;
  total_completion_tokens?: number | null;
  total_embedding_tokens?: number | null;
}

/** One connection string: what it is for, and its text as shown and copied. */
export interface ConnectionString {
  key: string;
  label: string;
  value: string;
}

const MASK = '••••••••';

/**
 * The connection strings of a project's agent database, for the libraries a
 * reader is likely to connect with. `password` is the text put where the
 * password goes: the mask, or the revealed password.
 */
export function connectionStrings(project: Project, password: string): ConnectionString[] {
  if (project.db_type === 'sqlite') {
    const file = project.db_name || 'agents.db';
    return [
      { key: 'sqlalchemy', label: 'SQLAlchemy (SQLite)', value: `sqlite:///${file}` },
      { key: 'async', label: 'SQLAlchemy async (aiosqlite)', value: `sqlite+aiosqlite:///${file}` },
      { key: 'jdbc', label: 'JDBC (SQLite)', value: `jdbc:sqlite:${file}` },
      { key: 'env', label: 'KAVALAI_DB_URI', value: `KAVALAI_DB_URI=sqlite:///${file}` },
    ];
  }
  const user = encodeURIComponent(project.db_user || 'user');
  const secret = password === MASK ? MASK : encodeURIComponent(password);
  const server = `${project.db_host || 'localhost'}:${project.db_port || 5432}`;
  const database = project.db_name || 'kavalai';
  const uri = `postgresql://${user}:${secret}@${server}/${database}`;
  return [
    { key: 'sqlalchemy', label: 'SQLAlchemy (PostgreSQL)', value: uri },
    {
      key: 'async',
      label: 'SQLAlchemy async (asyncpg)',
      value: `postgresql+asyncpg://${user}:${secret}@${server}/${database}`,
    },
    {
      key: 'jdbc',
      label: 'JDBC (PostgreSQL)',
      value: `jdbc:postgresql://${server}/${database}?user=${user}&password=${secret}`,
    },
    { key: 'env', label: 'KAVALAI_DB_URI', value: `KAVALAI_DB_URI=${uri}` },
  ];
}

/** The user's active project id, `''` for none (the API may send `'None'`). */
export function activeProjectOf(user: UserDetails): string {
  const id = user.active_project_id;
  return id && id !== 'None' ? id : '';
}

/** Where a project's runs are: `host:port / database / schema`, or the SQLite file. */
export function databaseLabel(project: Project): string {
  if (project.db_type === 'sqlite') return project.db_name || 'SQLite file';
  const server = [project.db_host, project.db_port].filter(Boolean).join(':');
  return [server, project.db_name, project.db_schema].filter(Boolean).join(' / ');
}

/**
 * The overview of the active project: its figures, its activity over the last
 * days, and how to connect to its agent database. The project is the one the
 * sidebar's switcher made active; the page follows the user details, so
 * switching there reloads it here. Settings live on their own page.
 */
@Component({
  selector: 'app-projects-page',
  imports: [
    DecimalPipe,
    RouterLink,
    BaseChartDirective,
    BarChart,
    UiAlert,
    UiBadge,
    UiButton,
    UiCard,
    UiCopyButton,
    UiEmptyState,
    UiLoading,
    UiPageHeader,
    UiSpinner,
    UiStat,
    UiStats,
  ],
  templateUrl: './projects-page.html',
  styleUrl: './projects-page.css',
})
export class ProjectsPage implements OnInit {
  private readonly projectService = inject(ProjectService);
  private readonly agentService = inject(AgentService);
  private readonly userService = inject(UserService);
  private readonly destroyRef = inject(DestroyRef);

  readonly activitySeries = ACTIVITY_SERIES;
  readonly tokenSeries = TOKEN_SERIES;
  readonly lineOptions = durationChartOptions();

  readonly loading = signal(true);
  readonly loadError = signal('');
  readonly project = signal<Project | null>(null);
  /** Whether the user is a member of any project, active or not. */
  readonly hasProjects = signal(false);
  private activeId = '';

  readonly statsLoading = signal(false);
  readonly statsError = signal('');
  readonly summary = signal<SummaryStats | null>(null);
  readonly daily = signal<DailyStats | null>(null);

  /** The password an administrator asked to see; listings never carry it. */
  readonly revealedPassword = signal<string | null>(null);
  readonly showPassword = signal(false);
  readonly revealing = signal(false);
  readonly revealError = signal('');

  readonly isAdmin = signal(false);

  readonly activity = computed(() => {
    const stats = this.daily();
    return stats ? activityBars(stats) : [];
  });
  readonly tokens = computed(() => {
    const stats = this.daily();
    return stats ? tokenBars(stats) : [];
  });
  readonly callDurations = computed(() => {
    const stats = this.daily();
    return stats ? callDurationLines(stats) : null;
  });
  readonly runDurations = computed(() => {
    const stats = this.daily();
    return stats ? runDurationLines(stats) : null;
  });

  readonly database = computed(() => {
    const project = this.project();
    return project ? databaseLabel(project) : '';
  });

  readonly strings = computed(() => {
    const revealed = this.revealedPassword();
    const password = this.showPassword() && revealed !== null ? revealed : MASK;
    // Read only inside the template's branch for a loaded project.
    return connectionStrings(this.project()!, password);
  });

  /** Only a PostgreSQL project with a stored password has one to show. */
  readonly canRevealPassword = computed(() => {
    const project = this.project();
    return this.isAdmin() && !!project && project.db_type !== 'sqlite' && !!project.has_password;
  });

  ngOnInit(): void {
    this.userService.userDetails
      .pipe(
        map((user) => (user ? activeProjectOf(user) : null)),
        distinctUntilChanged(),
        takeUntilDestroyed(this.destroyRef)
      )
      .subscribe((activeId) => {
        if (activeId === null) return;
        this.activeId = activeId;
        this.isAdmin.set(this.userService.getIsAdmin());
        this.load();
      });
  }

  /** The user's projects, of which the active one is shown. */
  load(): void {
    this.loading.set(true);
    this.loadError.set('');
    this.projectService.getAll().subscribe({
      next: (projects) => {
        const project = projects.find((p) => p.id === this.activeId) ?? null;
        this.hasProjects.set(projects.length > 0);
        this.loading.set(false);
        this.select(project);
      },
      error: (err) => {
        this.loading.set(false);
        this.loadError.set(errorMessage(err, 'The projects could not be loaded.'));
      },
    });
  }

  private select(project: Project | null): void {
    this.project.set(project);
    this.revealedPassword.set(null);
    this.showPassword.set(false);
    this.revealError.set('');
    this.summary.set(null);
    this.daily.set(null);
    if (project) this.loadStats();
  }

  /** The figures and the per-day statistics, read from the project's agent database. */
  loadStats(): void {
    const project = this.project()!;
    this.statsLoading.set(true);
    this.statsError.set('');
    forkJoin({
      summary: this.agentService.getSummaryStats(project.id),
      daily: this.agentService.getDailyStats(project.id, CHART_DAYS),
    }).subscribe({
      next: ({ summary, daily }) => {
        this.summary.set(summary as SummaryStats);
        this.daily.set(daily as DailyStats);
        this.statsLoading.set(false);
      },
      error: (err) => {
        this.statsError.set(errorMessage(err, "The project's database could not be read."));
        this.statsLoading.set(false);
      },
    });
  }

  /**
   * Show or hide the password in the connection strings. Project listings do
   * not carry it, so the first show asks the server, which answers only
   * administrators.
   */
  togglePassword(): void {
    const project = this.project()!;
    if (this.showPassword()) {
      this.showPassword.set(false);
      return;
    }
    if (this.revealedPassword() !== null) {
      this.showPassword.set(true);
      return;
    }
    this.revealing.set(true);
    this.revealError.set('');
    this.projectService.revealPassword(project.id).subscribe({
      next: ({ db_password }) => {
        this.revealedPassword.set(db_password ?? '');
        this.showPassword.set(true);
        this.revealing.set(false);
      },
      error: (err) => {
        this.revealing.set(false);
        this.revealError.set(errorMessage(err, 'The password could not be shown.'));
      },
    });
  }
}
