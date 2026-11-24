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

export type AlertTone = 'error' | 'success' | 'warning' | 'info';

const ICONS: Record<AlertTone, string> = {
  error: 'fa-circle-exclamation',
  success: 'fa-circle-check',
  warning: 'fa-triangle-exclamation',
  info: 'fa-circle-info',
};

/**
 * An inline message: an error with its retry button, a success note, a
 * warning about a limit. Content marked `alert-actions` sits at the right.
 */
@Component({
  selector: 'ui-alert',
  templateUrl: './alert.html',
  styleUrl: './alert.css',
  host: { '[attr.tone]': 'tone()', '[attr.role]': 'tone() === "error" ? "alert" : "status"' },
})
export class UiAlert {
  readonly tone = input<AlertTone>('error');
  readonly icon = computed(() => ICONS[this.tone()]);
}
