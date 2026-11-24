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

import { Component, booleanAttribute, input } from '@angular/core';

/** A label/value row; `mono` for identifiers, addresses and model names. */
@Component({
  selector: 'ui-info',
  templateUrl: './info.html',
  styleUrl: './info.css',
  host: { '[attr.mono]': 'mono() ? "" : null' },
})
export class UiInfo {
  readonly label = input.required<string>();
  readonly mono = input(false, { transform: booleanAttribute });
}
