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

import { Component, Directive, booleanAttribute, computed, input } from '@angular/core';

/**
 * A dropdown sheet anchored to its positioned parent; items carry `uiMenuItem`.
 *
 * `align` is the parent's edge the sheet lines up with: `end` (the user menu, at the right of the top bar) or
 * `start` (a picker at the left of a form). `role` is `menu` for a list of entries, `dialog` for a sheet holding
 * a form of its own, such as a filter box over checkboxes.
 */
@Component({
  selector: 'ui-menu',
  templateUrl: './menu.html',
  styleUrl: './menu.css',
  host: { '[attr.role]': 'role()', '[class.ui-menu-start]': "align() === 'start'" },
})
export class UiMenu {
  readonly align = input<'start' | 'end'>('end');
  readonly role = input('menu');
}

/**
 * An entry of a `ui-menu`: an `<a>` or a `<button>`; `danger` for sign-out and deletions.
 * A directive, so its styles live in menu-item.css, which ui.css imports.
 */
@Directive({
  selector: 'a[uiMenuItem], button[uiMenuItem]',
  host: { '[class]': 'classes()', role: 'menuitem' },
})
export class UiMenuItem {
  readonly danger = input(false, { transform: booleanAttribute });
  readonly classes = computed(() => (this.danger() ? 'ui-menu-item ui-menu-item-danger' : 'ui-menu-item'));
}
