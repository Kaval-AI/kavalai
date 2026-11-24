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

import { Component, ViewEncapsulation, computed, input, output } from '@angular/core';
import { UiButton } from '../button/button';

/**
 * The scroll container and styling for a plain `<table>`; the markup stays
 * the page's own. `dense` tightens the rows, `top` aligns cells to the top,
 * `clickable` gives rows a pointer; a row with class `selected` is
 * highlighted and one with `muted` dimmed.
 */
@Component({
  selector: 'ui-table',
  templateUrl: './table.html',
  styleUrl: './table.css',
  // The table, its rows and its cells are projected by the page.
  encapsulation: ViewEncapsulation.None,
})
export class UiTable {}

/** "Showing 1–50 of 120" and the Previous / Next pair under a paged table. */
@Component({
  selector: 'ui-pager',
  imports: [UiButton],
  templateUrl: './pager.html',
  styleUrl: './pager.css',
})
export class UiPager {
  /** Zero-based offset of the first row on the page. */
  readonly offset = input.required<number>();
  /** Rows on this page. */
  readonly count = input.required<number>();
  readonly total = input.required<number>();
  /** The plural noun for the whole set, shown when the set fits on one page. */
  readonly noun = input(`rows`);
  readonly busy = input(false);

  readonly hasPrevious = computed(() => this.offset() > 0);
  readonly hasNext = computed(() => this.offset() + this.count() < this.total());

  readonly range = computed(() => {
    const total = this.total().toLocaleString();
    if (this.count() === 0) return `0 ${this.noun()}`;
    if (!this.hasPrevious() && !this.hasNext()) return `${total} ${this.noun()}`;
    return `Showing ${this.offset() + 1}–${this.offset() + this.count()} of ${total}`;
  });

  readonly previous = output<void>();
  readonly next = output<void>();
}
