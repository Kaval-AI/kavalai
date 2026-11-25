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

import { Component, booleanAttribute, computed, input } from '@angular/core';

/** The initials of a name (or a word's first letters) in the brand's charcoal box, or a picture. */
export function initialsOf(name: string): string {
  const parts = name
    .split(/[\s@.]+/)
    .filter(Boolean)
    .slice(0, 2);
  if (parts.length === 1) return parts[0]!.substring(0, 2).toUpperCase();
  return parts.map((p) => p[0]!.toUpperCase()).join('');
}

@Component({
  selector: 'ui-avatar',
  templateUrl: './avatar.html',
  styleUrl: './avatar.css',
  host: { style: 'display: inline-flex' },
})
export class UiAvatar {
  readonly name = input.required<string>();
  readonly src = input<string | null>(null);
  readonly size = input<'sm' | 'md' | 'lg'>('md');
  /** A round avatar for people; the chamfered box is for chatbots. */
  readonly round = input(false, { transform: booleanAttribute });

  readonly initials = computed(() => initialsOf(this.name() || '?'));
  readonly classes = computed(() =>
    ['ui-avatar', `ui-avatar-${this.size()}`, this.round() ? 'ui-avatar-round' : ''].join(' ').trim(),
  );
}
