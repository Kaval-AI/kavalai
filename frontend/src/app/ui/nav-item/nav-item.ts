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

import { Component, ViewEncapsulation, booleanAttribute, input } from '@angular/core';
import { RouterLink, RouterLinkActive } from '@angular/router';

/** A sidebar destination: icon, label, and the active indicator when its route is open. */
@Component({
  selector: 'ui-nav-item',
  imports: [RouterLink, RouterLinkActive],
  templateUrl: './nav-item.html',
  styleUrl: './nav-item.css',
  // .ui-nav-badge marks content the sidebar projects into the item.
  encapsulation: ViewEncapsulation.None,
  host: { '[attr.compact]': 'compact() ? "" : null' },
})
export class UiNavItem {
  readonly link = input.required<string | unknown[]>();
  readonly label = input.required<string>();
  readonly icon = input('');
  readonly exact = input(false, { transform: booleanAttribute });
  /** The smaller row of a nested section. */
  readonly compact = input(false, { transform: booleanAttribute });
}
