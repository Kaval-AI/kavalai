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

/** A small rotating ring for in-progress work; `size="lg"` for a whole panel. */
@Component({
  selector: 'ui-spinner',
  // Nothing to render: the ring is the host element, drawn by spinner.css.
  template: '',
  styleUrl: './spinner.css',
  host: {
    class: 'ui-spinner',
    '[class.ui-spinner-lg]': 'size() === "lg"',
    role: 'status',
    'aria-label': `Loading`,
  },
})
export class UiSpinner {
  readonly size = input<'md' | 'lg'>('md');
}

/** A centred spinner with an optional caption, for a view that has nothing to show yet. */
@Component({
  selector: 'ui-loading',
  imports: [UiSpinner],
  templateUrl: './loading.html',
  styleUrl: './loading.css',
})
export class UiLoading {
  readonly label = input('');
}
