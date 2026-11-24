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
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { DatePipe, DecimalPipe } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ActivatedRoute, RouterLink } from '@angular/router';
import { AgentService } from '../../services/agent-service';
import { UserService } from '../../services/user-service';
import { SessionSummary } from '../../models/session';
import { Agent } from '../../models/agent';
import { errorMessage } from '../../core/http-error';
import {
  UiAlert,
  UiBadge,
  UiButton,
  UiCard,
  UiChip,
  UiCopyButton,
  UiEmptyState,
  UiField,
  UiInfo,
  UiLoading,
  UiPageHeader,
  UiSpinner,
  UiStat,
  UiStats,
  UiTable,
  UiToolbar,
} from '../../ui';

/** Sessions fetched per request; "Load more" asks for the next page. */
const PAGE_SIZE = 20;

/** A `datetime-local` value (local time, minutes) for `date`. */
function toLocalInput(date: Date): string {
  const offset = date.getTimezoneOffset() * 60000;
  return new Date(date.getTime() - offset).toISOString().slice(0, 16);
}

/** The default window: from the start of the day a week ago to the end of today. */
function defaultWindow(): { start: string; end: string } {
  const start = new Date();
  start.setDate(start.getDate() - 7);
  start.setHours(0, 0, 0, 0);
  const end = new Date();
  end.setHours(23, 59, 59, 999);
  return { start: toLocalInput(start), end: toLocalInput(end) };
}

/**
 * The conversations of the active project, latest activity first. The filters
 * narrow the list on the server; a selected row opens its summary beside the
 * table, and "Open conversation" leads to the transcript.
 *
 * The list endpoint has no total for the current page window, so paging is
 * "Load more": a page shorter than `PAGE_SIZE` is the last one.
 */
@Component({
  selector: 'app-conversations-page',
  imports: [
    DatePipe,
    DecimalPipe,
    FormsModule,
    RouterLink,
    UiAlert,
    UiBadge,
    UiButton,
    UiCard,
    UiChip,
    UiCopyButton,
    UiEmptyState,
    UiField,
    UiInfo,
    UiLoading,
    UiPageHeader,
    UiSpinner,
    UiStat,
    UiStats,
    UiTable,
    UiToolbar,
  ],
  templateUrl: './conversations-page.html',
  styleUrl: './conversations-page.css',
})
export class ConversationsPage implements OnInit {
  private readonly agentService = inject(AgentService);
  private readonly userService = inject(UserService);
  private readonly route = inject(ActivatedRoute);
  private readonly destroyRef = inject(DestroyRef);

  readonly projectId = signal<string | null>(null);
  readonly agents = signal<Agent[]>([]);
  readonly agentsError = signal('');
  readonly sessions = signal<SessionSummary[]>([]);
  readonly total = signal(0);
  readonly loading = signal(false);
  readonly error = signal('');
  readonly hasMore = signal(false);
  readonly selected = signal<SessionSummary | null>(null);

  /** The agent the list is narrowed to, shown as a chip in the header. */
  readonly agent = computed(() => {
    const id = this.agentFilter();
    return id ? (this.agents().find((a) => a.id === id) ?? null) : null;
  });

  agentId = '';
  search = '';
  /** Prefix match on the session's external id — the click-through from an eval result. */
  externalId = '';
  startDate = '';
  endDate = '';
  private readonly agentFilter = signal('');
  private offset = 0;

  constructor() {
    const window = defaultWindow();
    this.startDate = window.start;
    this.endDate = window.end;
  }

  ngOnInit(): void {
    this.route.queryParams.pipe(takeUntilDestroyed(this.destroyRef)).subscribe((params) => {
      if (params['agentId']) this.agentId = params['agentId'];
      if (params['externalId']) this.externalId = params['externalId'];
      this.agentFilter.set(this.agentId);
    });

    this.userService.userDetails.pipe(takeUntilDestroyed(this.destroyRef)).subscribe((user) => {
      const id = user?.active_project_id && user.active_project_id !== 'None' ? user.active_project_id : null;
      if (id === this.projectId()) return;
      this.projectId.set(id);
      if (id) {
        this.loadAgents();
        this.load();
      }
    });
  }

  loadAgents(): void {
    const projectId = this.projectId();
    if (!projectId) return;
    this.agentsError.set('');
    this.agentService.getAgentsByProject(projectId).subscribe({
      next: (agents) => this.agents.set(agents),
      error: (err) => this.agentsError.set(errorMessage(err, 'The agents could not be loaded.')),
    });
  }

  /** Load the first page (`reset`) or append the next one. */
  load(reset = true): void {
    const projectId = this.projectId();
    if (!projectId) return;
    if (reset) {
      this.offset = 0;
      this.selected.set(null);
    }
    this.agentFilter.set(this.agentId);
    this.loading.set(true);
    this.error.set('');
    this.agentService
      .getSessions(
        projectId,
        this.agentId || undefined,
        this.search.trim() || undefined,
        this.startDate ? new Date(this.startDate).toISOString() : undefined,
        this.endDate ? new Date(this.endDate).toISOString() : undefined,
        PAGE_SIZE,
        this.offset,
        this.externalId.trim() || undefined
      )
      .subscribe({
        next: (data: { sessions: SessionSummary[]; total_count: number }) => {
          this.sessions.set(reset ? data.sessions : [...this.sessions(), ...data.sessions]);
          this.total.set(data.total_count);
          this.hasMore.set(data.sessions.length >= PAGE_SIZE);
          this.loading.set(false);
        },
        error: (err) => {
          this.error.set(errorMessage(err, 'The conversations could not be loaded.'));
          this.loading.set(false);
        },
      });
  }

  /** Retry what failed: the first page, or the page "Load more" asked for. */
  retry(): void {
    this.load(this.offset === 0);
  }

  loadMore(): void {
    if (this.loading() || !this.hasMore()) return;
    this.offset += PAGE_SIZE;
    this.load(false);
  }

  clearFilters(): void {
    const window = defaultWindow();
    this.agentId = '';
    this.search = '';
    this.externalId = '';
    this.startDate = window.start;
    this.endDate = window.end;
    this.load();
  }

  select(session: SessionSummary): void {
    this.selected.set(this.selected()?.session_id === session.session_id ? null : session);
  }

  close(): void {
    this.selected.set(null);
  }

  /** Whether a conversation came from an evaluation run rather than a real user. */
  isEval(session: SessionSummary): boolean {
    return !!session.external_id?.startsWith('eval:');
  }
}
