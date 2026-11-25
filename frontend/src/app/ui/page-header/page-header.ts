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

import { Component, ViewEncapsulation, input } from '@angular/core';
import { RouterLink } from '@angular/router';

/**
 * The heading of a page: the section tag, the title, an optional back link
 * and a subtitle, with actions (a button, a range picker, the chatbot chip)
 * at the right. Extra title content, such as a status badge, is projected
 * with `page-title`; the default content is the subtitle.
 */
@Component({
  selector: 'ui-page-header',
  imports: [RouterLink],
  templateUrl: './page-header.html',
  styleUrl: './page-header.css',
  // The subtitle's paragraphs are the page's markup, projected in.
  encapsulation: ViewEncapsulation.None,
})
export class UiPageHeader {
  readonly title = input.required<string>();
  /** The tracked mono label above the title, e.g. the area of the console the page belongs to. */
  readonly tag = input('');
  /** A sub-page of the titled thing, shown after a slash: "Acme / Overview". */
  readonly section = input('');
  readonly backLink = input<string | unknown[] | null>(null);
  readonly backLabel = input('Back');
}
