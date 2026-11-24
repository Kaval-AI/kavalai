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
import { Component, DestroyRef, OnInit, computed, inject, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { DatePipe, DecimalPipe, NgTemplateOutlet } from '@angular/common';
import { ActivatedRoute, RouterLink } from '@angular/router';
import { AgentService } from '../../services/agent-service';
import { UserService } from '../../services/user-service';
import { ChatMessage } from '../../models/chat-message';
import { Run } from '../../models/run';
import { SessionDetails } from '../../models/session';
import { Task } from '../../models/task';
import { JsonTreeComponent } from '../json-tree/json-tree';
import { errorMessage } from '../../core/http-error';
import {
  UiAlert,
  UiBadge,
  UiButton,
  UiCard,
  UiCopyButton,
  UiDialog,
  UiEmptyState,
  UiInfo,
  UiLoading,
  UiPageHeader,
  UiStat,
  UiStats,
} from '../../ui';

/**
 * One run of the conversation as the page shows it: the run, its position
 * (1-based, in the order the runs started), the messages it recorded, the
 * names of the steps it ran (top-level tasks, in execution order) and the
 * number of errors its tasks recorded.
 */
export interface RunBlock {
  run: Run;
  index: number;
  messages: ChatMessage[];
  steps: string[];
  errors: number;
}

/** The JSON shown in the dialog: which of a run's payloads, and the value. */
interface JsonView {
  title: string;
  data: unknown;
}

function byCreated(a: { created_at: string }, b: { created_at: string }): number {
  return new Date(a.created_at).getTime() - new Date(b.created_at).getTime();
}

function hasText(message: ChatMessage): boolean {
  return !!message.content && message.content.trim().length > 0;
}

/**
 * The transcript of one conversation, grouped by run. Each run is a card with
 * its metadata, the messages it recorded and the way to what lies behind it:
 * its tasks, its model calls and its input, output and context as JSON.
 */
@Component({
  selector: 'app-session-detail-page',
  imports: [
    DatePipe,
    DecimalPipe,
    NgTemplateOutlet,
    RouterLink,
    JsonTreeComponent,
    UiAlert,
    UiBadge,
    UiButton,
    UiCard,
    UiCopyButton,
    UiDialog,
    UiEmptyState,
    UiInfo,
    UiLoading,
    UiPageHeader,
    UiStat,
    UiStats,
  ],
  templateUrl: './session-detail-page.html',
  styleUrl: './session-detail-page.css',
})
export class SessionDetailPage implements OnInit {
  private readonly route = inject(ActivatedRoute);
  private readonly agentService = inject(AgentService);
  private readonly userService = inject(UserService);
  private readonly destroyRef = inject(DestroyRef);

  readonly sessionId = signal<string | null>(null);
  readonly projectId = signal<string | null>(null);
  readonly runs = signal<RunBlock[]>([]);
  readonly unassigned = signal<ChatMessage[]>([]);
  readonly loading = signal(false);
  readonly error = signal('');
  readonly json = signal<JsonView | null>(null);

  readonly shortId = computed(() => this.sessionId()?.slice(0, 8) ?? '');
  readonly empty = computed(() => this.runs().length === 0 && this.unassigned().length === 0);
  readonly formatDuration = formatDuration;

  ngOnInit(): void {
    this.route.paramMap.pipe(takeUntilDestroyed(this.destroyRef)).subscribe((params) => {
      this.sessionId.set(params.get('sessionId'));
      this.load();
    });

    this.userService.userDetails.pipe(takeUntilDestroyed(this.destroyRef)).subscribe((user) => {
      const id = user?.active_project_id && user.active_project_id !== 'None' ? user.active_project_id : null;
      if (id === this.projectId()) return;
      this.projectId.set(id);
      this.load();
    });
  }

  load(): void {
    const projectId = this.projectId();
    const sessionId = this.sessionId();
    if (!projectId || !sessionId) return;
    this.loading.set(true);
    this.error.set('');
    this.agentService.getSessionDetails(projectId, sessionId).subscribe({
      next: (details) => {
        this.group(details);
        this.loading.set(false);
      },
      error: (err) => {
        this.error.set(errorMessage(err, 'The conversation could not be loaded.'));
        this.loading.set(false);
      },
    });
  }

  showJson(title: string, data: unknown): void {
    this.json.set({ title, data });
  }

  closeJson(): void {
    this.json.set(null);
  }

  /**
   * A message's content as indented JSON when it is a JSON object or array
   * (a workflow's structured answer), otherwise `null`.
   */
  prettyJson(content: string): string | null {
    const text = content.trim();
    if (!text.startsWith('{') && !text.startsWith('[')) return null;
    try {
      return JSON.stringify(JSON.parse(text), null, 2);
    } catch {
      return null;
    }
  }

  private group(details: SessionDetails): void {
    const runIds = new Set(details.runs.map((r) => r.id));
    const messages = details.messages.filter(hasText).sort(byCreated);
    this.runs.set(
      [...details.runs].sort(byCreated).map((run, i) => {
        const tasks = details.tasks.filter((t) => t.run_id === run.id);
        return {
          run,
          index: i + 1,
          messages: messages.filter((m) => m.run_id === run.id),
          steps: stepNames(tasks),
          errors: tasks.reduce((sum, t) => sum + (t.errors?.length ?? 0), 0),
        };
      })
    );
    this.unassigned.set(messages.filter((m) => !m.run_id || !runIds.has(m.run_id)));
  }
}

/** The distinct names of a run's top-level steps, in the order they ran. */
function stepNames(tasks: Task[]): string[] {
  const order = (t: Task) => t.seq ?? Number.MAX_SAFE_INTEGER;
  const names = tasks
    .filter((t) => !t.parent_task_name)
    .sort((a, b) => order(a) - order(b) || byCreated(a, b))
    .map((t) => t.name || t.id);
  return [...new Set(names)];
}
