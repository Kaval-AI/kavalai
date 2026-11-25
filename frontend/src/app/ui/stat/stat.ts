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

import { Component, input } from '@angular/core';
import { Params, RouterLink } from '@angular/router';

/** A grid of `ui-stat` tiles; `inset` for tiles that sit inside a card. */
@Component({
  selector: 'ui-stats',
  templateUrl: './stats.html',
  styleUrl: './stats.css',
  host: { role: 'list' },
})
export class UiStats {}

/**
 * One figure with its label; `link` makes the whole tile a navigation target
 * (a stretched overlay link, so the value stays a single projected slot), with
 * `queryParams` when the page it opens takes some. The content is the value.
 */
@Component({
  selector: 'ui-stat',
  imports: [RouterLink],
  templateUrl: './stat.html',
  styleUrl: './stat.css',
  host: { role: 'listitem', '[class.ui-stat-linked]': 'link()' },
})
export class UiStat {
  readonly label = input.required<string>();
  readonly hint = input('');
  readonly link = input<string | unknown[] | null>(null);
  readonly queryParams = input<Params | null>(null);
}
