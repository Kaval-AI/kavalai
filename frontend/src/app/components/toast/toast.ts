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
import { Component, inject } from '@angular/core';
import { ToastService } from '../../services/toast-service';
import { UiAlert } from '../../ui';

/** The toasts of `ToastService`, stacked in the bottom-right corner. */
@Component({
  selector: 'app-toast',
  imports: [UiAlert],
  template: `
    <div class="toasts" aria-live="polite">
      @for (toast of toastService.toasts(); track toast.id) {
        <ui-alert class="toast" [tone]="toast.type" [attr.data-testid]="'toast-' + toast.type">
          {{ toast.message }}
          <button
            alert-actions
            type="button"
            class="toast-close"
            aria-label="Dismiss"
            (click)="toastService.remove(toast.id)"
          >
            <i class="fa-solid fa-xmark" aria-hidden="true"></i>
          </button>
        </ui-alert>
      }
    </div>
  `,
  styles: `
    .toasts {
      position: fixed;
      right: var(--space-5);
      bottom: var(--space-5);
      z-index: 2000;
      display: flex;
      flex-direction: column;
      align-items: flex-end;
      gap: var(--space-2);
      max-width: calc(100vw - 2 * var(--space-5));
      pointer-events: none;
    }

    .toast {
      pointer-events: auto;
      min-width: 220px;
      box-shadow: var(--elevation-2);
    }

    .toast-close {
      color: inherit;
      opacity: 0.7;
      cursor: pointer;
    }

    .toast-close:hover {
      opacity: 1;
    }
  `,
})
export class Toast {
  public toastService = inject(ToastService);
}
