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
import { RouterLink } from '@angular/router';

export type CardTone = 'default' | 'danger' | 'dark';

/**
 * A white, chamfered surface: the section of a page. `title` and `icon`
 * draw the lime-ruled heading; content marked `card-actions` sits at the
 * heading's right end. `tone="danger"` marks a destructive section and
 * `tone="dark"` gives the charcoal panel with the lime outline. `selected`
 * marks the one card of a list the rest of the console is working on.
 * `titleLink` turns the heading into a link to the thing the card is about,
 * Content marked `card-footer` closes the card; with `fill` the card takes
 * the height of its grid row and the footer sits at its bottom, so cards
 * side by side end on one line.
 */
@Component({
  selector: 'ui-card',
  imports: [RouterLink],
  templateUrl: './card.html',
  styleUrl: './card.css',
  // The dark tone recolours the headings and links a page projects in.
  encapsulation: ViewEncapsulation.None,
  host: {
    '[attr.tone]': 'tone()',
    '[attr.selected]': 'selected() ? "" : null',
    '[attr.fill]': 'fill() ? "" : null',
  },
})
export class UiCard {
  readonly title = input('');
  /** Where the heading navigates, when the card stands for a thing with its own page. */
  readonly titleLink = input<string | unknown[] | null>(null);
  readonly icon = input('');
  readonly tone = input<CardTone>('default');
  readonly selected = input(false, { transform: booleanAttribute });
  /** As tall as its grid row, with the footer at the bottom. */
  readonly fill = input(false, { transform: booleanAttribute });
}
