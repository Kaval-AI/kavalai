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

import { Component, ElementRef, booleanAttribute, effect, input, output, viewChild } from '@angular/core';

/**
 * A modal box over the page for a decision that needs explaining before it is
 * made: what a removal takes with it, say. The native `<dialog>` in its modal
 * state, so the rest of the page is inert and Escape closes it; clicking the
 * backdrop closes it too. The parent owns the state: `open` shows it, and
 * `closed` fires when the person dismissed it, so the parent sets `open` back.
 *
 *   <ui-dialog [open]="removing() !== null" (closed)="removing.set(null)" title="Remove user">
 *     <p>...</p>
 *     <button dialog-actions uiButton="ghost">Cancel</button>
 *     <button dialog-actions uiButton="danger">Remove</button>
 *   </ui-dialog>
 */
@Component({
  selector: 'ui-dialog',
  templateUrl: './dialog.html',
  styleUrl: './dialog.css',
})
export class UiDialog {
  readonly open = input(false, { transform: booleanAttribute });
  readonly title = input('');
  /** `md` for a decision; `lg` and `xl` for a document — JSON, a log. */
  readonly size = input<'md' | 'lg' | 'xl'>('md');
  /** The person dismissed it: Escape, or a click on the backdrop. */
  readonly closed = output<void>();

  private readonly dialog = viewChild.required<ElementRef<HTMLDialogElement>>('dialog');

  constructor() {
    effect(() => {
      const element = this.dialog().nativeElement;
      if (this.open()) {
        if (!element.open) element.showModal();
      } else if (element.open) {
        element.close();
      }
    });
  }

  /** The native close: only a dismissal is reported, not the close `open` asked for. */
  onClose(): void {
    if (this.open()) this.closed.emit();
  }

  onClick(event: MouseEvent): void {
    if (event.target === this.dialog().nativeElement) this.dialog().nativeElement.close();
  }
}
