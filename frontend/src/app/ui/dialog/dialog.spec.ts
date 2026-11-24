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

import { Component, provideZonelessChangeDetection, signal } from '@angular/core';
import { TestBed } from '@angular/core/testing';
import { UiDialog } from './dialog';

@Component({
  imports: [UiDialog],
  template: `
    <ui-dialog [open]="open()" (closed)="open.set(false); dismissed = dismissed + 1" title="Sure?">
      <p>Body</p>
      <button dialog-actions type="button" (click)="open.set(false)">Cancel</button>
    </ui-dialog>
  `,
})
class Host {
  readonly open = signal(false);
  dismissed = 0;
}

describe('UiDialog', () => {
  beforeEach(() => {
    TestBed.configureTestingModule({ providers: [provideZonelessChangeDetection()] });
  });

  function create() {
    const fixture = TestBed.createComponent(Host);
    fixture.detectChanges();
    const dialog = fixture.nativeElement.querySelector('dialog') as HTMLDialogElement;
    return { fixture, dialog };
  }

  /** The browser fires a dialog's close event from a task of its own, after the close itself. */
  async function closed(fixture: { whenStable(): Promise<unknown> }, dialog: HTMLDialogElement): Promise<void> {
    await new Promise((resolve) => dialog.addEventListener('close', resolve, { once: true }));
    await fixture.whenStable();
  }

  it('shows the modal while open and closes it when open goes back', async () => {
    const { fixture, dialog } = create();
    expect(dialog.open).toBeFalse();
    fixture.componentInstance.open.set(true);
    await fixture.whenStable();
    expect(dialog.open).toBeTrue();
    expect(dialog.querySelector('.ui-dialog-title')?.textContent).toBe('Sure?');
    expect(dialog.querySelector('.ui-dialog-content')?.textContent).toContain('Body');
    expect(dialog.querySelector('.ui-dialog-actions button')?.textContent).toBe('Cancel');
    fixture.componentInstance.open.set(false);
    await fixture.whenStable();
    expect(dialog.open).toBeFalse();
    // A close the parent asked for is not a dismissal.
    expect(fixture.componentInstance.dismissed).toBe(0);
  });

  it('opens centred in the window, whatever the global reset does to dialogs', async () => {
    const { fixture, dialog } = create();
    fixture.componentInstance.open.set(true);
    await fixture.whenStable();
    const box = dialog.getBoundingClientRect();
    const left = box.left;
    const right = document.documentElement.clientWidth - box.right;
    expect(Math.abs(left - right)).toBeLessThan(2);
    expect(left).toBeGreaterThan(0);
    fixture.componentInstance.open.set(false);
    await fixture.whenStable();
  });

  it('carries its size for the wider document dialogs', () => {
    const fixture = TestBed.createComponent(UiDialog);
    fixture.detectChanges();
    const dialog = fixture.nativeElement.querySelector('dialog') as HTMLDialogElement;
    expect(dialog.getAttribute('data-size')).toBe('md');
    fixture.componentRef.setInput('size', 'xl');
    fixture.detectChanges();
    expect(dialog.getAttribute('data-size')).toBe('xl');
  });

  it('reports a dismissal by Escape or a click on the backdrop', async () => {
    const { fixture, dialog } = create();
    fixture.componentInstance.open.set(true);
    await fixture.whenStable();
    // Escape: the browser closes the dialog and fires close.
    dialog.close();
    await closed(fixture, dialog);
    expect(fixture.componentInstance.dismissed).toBe(1);
    fixture.componentInstance.open.set(true);
    await fixture.whenStable();
    expect(dialog.open).toBeTrue();
    dialog.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await closed(fixture, dialog);
    expect(dialog.open).toBeFalse();
    expect(fixture.componentInstance.dismissed).toBe(2);
    // A click inside the box is not a click on the backdrop.
    fixture.componentInstance.open.set(true);
    await fixture.whenStable();
    dialog.querySelector('p')?.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await fixture.whenStable();
    expect(dialog.open).toBeTrue();
  });
});
