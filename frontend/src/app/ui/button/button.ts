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

import { Directive, booleanAttribute, computed, input } from '@angular/core';

export type ButtonVariant = 'primary' | 'secondary' | 'ghost' | 'danger' | 'icon';
export type ButtonSize = 'sm' | 'md' | 'lg';

/**
 * The console's button. A directive rather than a component so the native
 * `<button>` (type, disabled, form submission) and `<a routerLink>` keep
 * their semantics: `<button uiButton="primary" type="submit">Save</button>`.
 * Its styles are in button.css, which ui.css imports, because a directive
 * cannot carry a stylesheet of its own.
 *
 * Variants: `primary` (lime, chamfered, the one call to action of a view),
 * `secondary` (outlined, the default), `ghost` (text only), `danger`
 * (destructive, confirms in place), `icon` (a square icon-only button; give
 * it an aria-label). Put a `<ui-spinner />` inside while the action runs.
 */
@Directive({
  selector: 'button[uiButton], a[uiButton]',
  host: { '[class]': 'classes()' },
})
export class UiButton {
  readonly uiButton = input<ButtonVariant | ''>('');
  readonly size = input<ButtonSize>('md');
  readonly block = input(false, { transform: booleanAttribute });

  readonly variant = computed<ButtonVariant>(() => this.uiButton() || 'secondary');

  readonly classes = computed(() =>
    ['ui-btn', `ui-btn-${this.variant()}`, `ui-btn-${this.size()}`, this.block() ? 'ui-btn-block' : '']
      .join(' ')
      .trim(),
  );
}
