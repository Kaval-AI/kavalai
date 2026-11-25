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
import { DatePipe, DecimalPipe } from '@angular/common';
import { Component, DestroyRef, effect, inject, input, output, signal, untracked } from '@angular/core';
import { Router } from '@angular/router';
import { Subscription } from 'rxjs';
import { errorMessage } from '../../core/http-error';
import { formatDuration } from '../../core/format';
import { LLMCallStat } from '../../models/llm-call-stat';
import { AgentService } from '../../services/agent-service';
import { UiAlert, UiBadge, UiButton, UiLoading, UiStatusBadge, UiTable } from '../../ui';

/** A run's model calls are listed up to this many, oldest first. */
export const RUN_CALL_LIMIT = 100;

/** `ok` or `error` from a call's HTTP status; `unknown` when none was recorded. */
export function callStatus(call: LLMCallStat): string {
  if (call.response_code === null || call.response_code === undefined) return 'unknown';
  return call.response_code >= 200 && call.response_code < 300 ? 'ok' : 'error';
}

/**
 * The model calls one run made, as a table whose rows open the call on the
 * Model calls page. It loads them itself, from the model call list filtered to
 * the run, and hands what it loaded to `loaded` so a page can total them.
 */
@Component({
  selector: 'app-run-model-calls',
  imports: [DatePipe, DecimalPipe, UiAlert, UiBadge, UiButton, UiLoading, UiStatusBadge, UiTable],
  templateUrl: './run-model-calls.html',
  styleUrl: './run-model-calls.css',
})
export class RunModelCalls {
  private readonly agentService = inject(AgentService);
  private readonly router = inject(Router);

  readonly projectId = input.required<string>();
  readonly runId = input.required<string>();
  readonly loaded = output<LLMCallStat[]>();

  readonly calls = signal<LLMCallStat[]>([]);
  readonly loading = signal(false);
  readonly error = signal('');

  readonly formatDuration = formatDuration;
  readonly callStatus = callStatus;
  readonly limit = RUN_CALL_LIMIT;

  private subscription: Subscription | null = null;

  constructor() {
    effect(() => {
      const projectId = this.projectId();
      const runId = this.runId();
      untracked(() => this.load(projectId, runId));
    });
    inject(DestroyRef).onDestroy(() => this.subscription?.unsubscribe());
  }

  load(projectId = this.projectId(), runId = this.runId()): void {
    this.subscription?.unsubscribe();
    this.loading.set(true);
    this.error.set('');
    this.subscription = this.agentService
      .getLLMCallStats(projectId, undefined, RUN_CALL_LIMIT, 0, { runId })
      .subscribe({
        next: (calls) => {
          this.calls.set([...calls].reverse());
          this.loading.set(false);
          this.loaded.emit(this.calls());
        },
        error: (err) => {
          this.calls.set([]);
          this.error.set(errorMessage(err, 'Could not load the model calls.'));
          this.loading.set(false);
          this.loaded.emit([]);
        },
      });
  }

  open(call: LLMCallStat): void {
    this.router.navigate(['/llm-call-stats'], { queryParams: { run_id: this.runId(), call: call.id } });
  }
}
