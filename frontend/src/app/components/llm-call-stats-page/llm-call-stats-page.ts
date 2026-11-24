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
import {
  Component,
  DestroyRef,
  ElementRef,
  OnInit,
  computed,
  inject,
  signal,
  viewChild,
} from '@angular/core';
import { DatePipe, DecimalPipe } from '@angular/common';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { FormsModule } from '@angular/forms';
import { ActivatedRoute, Params, Router, RouterLink } from '@angular/router';
import { Subscription } from 'rxjs';
import { AgentService } from '../../services/agent-service';
import { UserService } from '../../services/user-service';
import { LLMCallStat, LLMCallStatFilters, LLMCallSummary } from '../../models/llm-call-stat';
import { TOKEN_SERIES } from '../../core/daily-stats';
import { errorMessage } from '../../core/http-error';
import {
  BarChart,
  ChartBar,
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
  UiStatusBadge,
  UiTable,
  UiToolbar,
  fillDays,
} from '../../ui';
import { JsonTreeComponent } from '../json-tree/json-tree';
import { formatDuration, shortId } from '../../core/format';
import { ChatLine, parsePayload, requestMessages, responseText } from './call-text';

/** The query parameters that narrow the list, in the order their chips are shown. */
const FILTERS = [
  { param: 'call_type', label: 'Type' },
  { param: 'agent_id', label: 'Agent' },
  { param: 'session_id', label: 'Conversation' },
  { param: 'run_id', label: 'Run' },
] as const;

/** Below this width the detail panel sits under the table (see the stylesheet). */
const NARROW = '(max-width: 1100px)';

type FilterParam = (typeof FILTERS)[number]['param'];

/** The search box and the time window, kept in the query as `search`, `from` and `to`. */
export interface TextFilters {
  search: string;
  /** `datetime-local` values: local time, no zone. */
  from: string;
  to: string;
}

/** A gap between days with calls is drawn as quiet days only up to this many days. */
const MAX_FILLED_DAYS = 92;

/** The chart shows at least this many days, ending on the last day with calls. */
const MIN_DAYS = 7;

/**
 * The tokens per day of a summary as chart bars: LLM input, LLM output and
 * embedding tokens. Days without calls between the first and the last are
 * drawn empty while the span is short enough to read, and the chart covers
 * at least a week, so one busy day is one bar among seven.
 */
export function tokenDays(summary: LLMCallSummary): ChartBar[] {
  const rows = summary.by_day.map((d) => ({
    day: d.date,
    values: [d.prompt_tokens, d.completion_tokens, d.embedding_tokens],
  }));
  if (rows.length === 0) return rows;
  const last = Date.parse(`${rows[rows.length - 1].day}T12:00:00Z`);
  const first = Math.min(Date.parse(`${rows[0].day}T12:00:00Z`), last - (MIN_DAYS - 1) * 86_400_000);
  if ((last - first) / 86_400_000 > MAX_FILLED_DAYS) return rows;
  return fillDays(rows, new Date(first), new Date(last), () => ({ day: '', values: [0, 0, 0] }));
}

/** A `datetime-local` value as an ISO time for the API; undefined when empty or invalid. */
function isoOf(local: string): string | undefined {
  if (!local) return undefined;
  const time = new Date(local);
  return Number.isNaN(time.getTime()) ? undefined : time.toISOString();
}

export interface ActiveFilter {
  param: FilterParam;
  label: string;
  value: string;
}

/** A selected call, with its payloads read once rather than on every render. */
interface CallView {
  call: LLMCallStat;
  request: unknown;
  response: unknown;
  messages: ChatLine[] | null;
  text: string | null;
}

/**
 * Lists model calls, newest first. The `call_type`, `agent_id`,
 * `session_id` and `run_id` query parameters narrow the list, so a run's
 * calls are one link away from the run and from any call it made; `call`
 * opens that call when it is on the first page. A selected row opens beside
 * the table with the call's figures, links and its request and response,
 * readable as messages with "Show as text".
 */
@Component({
  selector: 'app-llm-call-stats-page',
  standalone: true,
  imports: [
    BarChart,
    DatePipe,
    DecimalPipe,
    FormsModule,
    RouterLink,
    JsonTreeComponent,
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
    UiStatusBadge,
    UiTable,
    UiToolbar,
  ],
  templateUrl: './llm-call-stats-page.html',
  styleUrl: './llm-call-stats-page.css',
})
export class LlmCallStatsPage implements OnInit {
  private route = inject(ActivatedRoute);
  private router = inject(Router);
  private agentService = inject(AgentService);
  private userService = inject(UserService);
  private destroyRef = inject(DestroyRef);

  readonly limit = 20;

  readonly projectId = signal<string | null>(null);
  readonly filterValues = signal<Partial<Record<FilterParam, string>>>({});
  readonly stats = signal<LLMCallStat[]>([]);
  readonly loading = signal(false);
  readonly error = signal<string | null>(null);
  readonly hasMore = signal(false);
  readonly selectedId = signal<string | null>(null);
  readonly showText = signal(true);

  /** The search and time window the list and the summary are narrowed by. */
  readonly textFilters = signal<TextFilters>({ search: '', from: '', to: '' });
  /** The toolbar's fields, applied to the query with Search. */
  form: TextFilters = { search: '', from: '', to: '' };

  readonly summary = signal<LLMCallSummary | null>(null);
  readonly summaryLoading = signal(false);
  readonly summaryError = signal<string | null>(null);
  readonly tokenSeries = TOKEN_SERIES;
  readonly tokenBars = computed(() => {
    const summary = this.summary();
    return summary ? tokenDays(summary) : [];
  });
  readonly embeddingTokens = computed(
    () => this.summary()?.by_model.filter((m) => m.call_type === 'embedding').reduce((t, m) => t + m.total_tokens, 0) ?? 0
  );
  /** Whether the search or the time window narrows the list. */
  readonly textFiltered = computed(() => {
    const f = this.textFilters();
    return !!(f.search || f.from || f.to);
  });

  readonly formatDuration = formatDuration;
  readonly shortId = shortId;

  readonly filters = computed<ActiveFilter[]>(() => {
    const values = this.filterValues();
    return FILTERS.filter((f) => values[f.param]).map((f) => ({
      param: f.param,
      label: f.label,
      value: values[f.param]!,
    }));
  });

  readonly selected = computed<CallView | null>(() => {
    const call = this.stats().find((s) => s.id === this.selectedId());
    if (!call) return null;
    return {
      call,
      request: parsePayload(call.request_data),
      response: parsePayload(call.response_data),
      messages: requestMessages(call.request_data),
      text: responseText(call.response_data),
    };
  });

  /** Whether the selected call has anything the text view can show. */
  readonly hasText = computed(() => {
    const view = this.selected();
    return !!view && (view.messages !== null || view.text !== null);
  });

  private readonly detailPanel = viewChild<ElementRef<HTMLElement>>('detail');
  private summarySubscription: Subscription | null = null;
  private offset = 0;
  private requestedCall: string | null = null;
  private loadSubscription: Subscription | null = null;

  ngOnInit(): void {
    this.route.queryParams.pipe(takeUntilDestroyed(this.destroyRef)).subscribe((params) => {
      this.requestedCall = params['call'] || null;
      this.filterValues.set(this.readFilters(params));
      const text: TextFilters = { search: params['search'] || '', from: params['from'] || '', to: params['to'] || '' };
      this.textFilters.set(text);
      this.form = { ...text };
      this.reload();
    });
    this.userService.userDetails.pipe(takeUntilDestroyed(this.destroyRef)).subscribe((user) => {
      const projectId =
        user?.active_project_id && user.active_project_id !== 'None' ? user.active_project_id : null;
      if (projectId !== this.projectId()) {
        this.projectId.set(projectId);
        this.reload();
      }
    });
    this.destroyRef.onDestroy(() => {
      this.loadSubscription?.unsubscribe();
      this.summarySubscription?.unsubscribe();
    });
  }

  /** The query parameters that keep the other filters and drop this one. */
  without(param: FilterParam): Params {
    return { [param]: null, call: null };
  }

  reload(): void {
    this.offset = 0;
    this.stats.set([]);
    this.selectedId.set(null);
    this.hasMore.set(false);
    this.loadStats();
    this.loadSummary();
  }

  /** Put the toolbar's search and time window into the query, which reloads the page. */
  applyText(): void {
    const { search, from, to } = this.form;
    this.router.navigate([], {
      relativeTo: this.route,
      queryParams: { search: search.trim() || null, from: from || null, to: to || null, call: null },
      queryParamsHandling: 'merge',
    });
  }

  clearText(): void {
    this.form = { search: '', from: '', to: '' };
    this.applyText();
  }

  loadSummary(): void {
    const projectId = this.projectId();
    if (!projectId) return;
    this.summarySubscription?.unsubscribe();
    this.summaryLoading.set(true);
    this.summaryError.set(null);
    const values = this.filterValues();
    this.summarySubscription = this.agentService
      .getLLMCallSummary(projectId, values.call_type, this.requestFilters())
      .subscribe({
        next: (summary) => {
          this.summary.set(summary);
          this.summaryLoading.set(false);
        },
        error: (err) => {
          this.summary.set(null);
          this.summaryError.set(errorMessage(err, 'Could not load the token usage.'));
          this.summaryLoading.set(false);
        },
      });
  }

  loadStats(): void {
    const projectId = this.projectId();
    if (!projectId) return;

    // A filter change supersedes a page still in flight.
    this.loadSubscription?.unsubscribe();
    this.loading.set(true);
    this.error.set(null);
    const values = this.filterValues();
    this.loadSubscription = this.agentService
      .getLLMCallStats(projectId, values.call_type, this.limit, this.offset, this.requestFilters())
      .subscribe({
        next: (data) => {
          this.stats.update((stats) => [...stats, ...data]);
          this.loading.set(false);
          this.hasMore.set(data.length === this.limit);
          if (this.requestedCall && data.some((s) => s.id === this.requestedCall)) {
            this.selectedId.set(this.requestedCall);
            this.requestedCall = null;
          }
        },
        error: (err) => {
          this.error.set(errorMessage(err, 'Could not load the model calls.'));
          this.loading.set(false);
        },
      });
  }

  loadMore(): void {
    if (this.loading() || !this.hasMore()) return;
    this.offset += this.limit;
    this.loadStats();
  }

  select(call: LLMCallStat): void {
    this.selectedId.set(this.selectedId() === call.id ? null : call.id);
    // On a narrow screen the panel opens below the table, out of sight.
    if (this.selectedId() && window.matchMedia(NARROW).matches) {
      setTimeout(() => this.detailPanel()?.nativeElement.scrollIntoView({ block: 'start' }));
    }
  }

  close(): void {
    this.selectedId.set(null);
  }

  toggleText(event: Event): void {
    this.showText.set((event.target as HTMLInputElement).checked);
  }

  status(call: LLMCallStat): string {
    if (call.response_code === null || call.response_code === undefined) return 'unknown';
    return call.response_code >= 200 && call.response_code < 300 ? 'ok' : 'error';
  }

  /** The id filters, the search and the time window, as the API takes them; unset ones are left out. */
  private requestFilters(): LLMCallStatFilters {
    const values = this.filterValues();
    const text = this.textFilters();
    const filters: LLMCallStatFilters = {
      agentId: values.agent_id,
      sessionId: values.session_id,
      runId: values.run_id,
      search: text.search || undefined,
      start: isoOf(text.from),
      end: isoOf(text.to),
    };
    return Object.fromEntries(Object.entries(filters).filter(([, v]) => v !== undefined));
  }

  private readFilters(params: Params): Partial<Record<FilterParam, string>> {
    const values: Partial<Record<FilterParam, string>> = {};
    for (const f of FILTERS) {
      if (params[f.param]) values[f.param] = params[f.param];
    }
    return values;
  }
}
