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

import { Component, computed, input, signal } from '@angular/core';
import { UiButton } from '../button/button';

/** Copies `text` to the clipboard and confirms briefly. */
@Component({
  selector: 'ui-copy-button',
  imports: [UiButton],
  templateUrl: './copy-button.html',
})
export class UiCopyButton {
  readonly text = input.required<string>();
  readonly label = input(`to clipboard`);
  readonly copied = signal(false);
  readonly copyLabel = computed(() => `Copy ${this.label()}`);

  async copy(): Promise<void> {
    try {
      await navigator.clipboard.writeText(this.text());
      this.copied.set(true);
      setTimeout(() => this.copied.set(false), 1500);
    } catch {
      // Clipboard access denied: the text is visible to select by hand.
    }
  }
}
