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

export type BadgeTone = 'neutral' | 'success' | 'warning' | 'error' | 'info' | 'muted' | 'primary';

/** A small chamfered mono label: a plan, a role, a state. */
@Component({
  selector: 'ui-badge',
  templateUrl: './badge.html',
  styleUrl: './badge.css',
  host: { '[class]': 'classes()' },
})
export class UiBadge {
  readonly tone = input<BadgeTone>('neutral');
  readonly classes = computed(() => `ui-badge ui-badge-${this.tone()}`);
}

const STATUS_LABELS: Record<string, string> = {
  running: `Running`,
  completed: `Completed`,
  failed: `Failed`,
  ok: `OK`,
  error: `Error`,
};

const STATUS_TONES: Record<string, BadgeTone> = {
  running: 'warning',
  completed: 'success',
  failed: 'error',
  ok: 'success',
  error: 'error',
};

/**
 * A badge worded and coloured by the state of a run, a task or a model call:
 * `running`, `completed` and `failed` for a run, `ok` and `error` for a single
 * step. An unknown status is shown as it is, in the neutral tone.
 */
@Component({
  selector: 'ui-status-badge',
  imports: [UiBadge],
  templateUrl: './status-badge.html',
})
export class UiStatusBadge {
  readonly status = input.required<string>();
  readonly label = computed(() => STATUS_LABELS[this.status()] ?? this.status());
  readonly tone = computed(() => STATUS_TONES[this.status()] ?? 'neutral');
}
