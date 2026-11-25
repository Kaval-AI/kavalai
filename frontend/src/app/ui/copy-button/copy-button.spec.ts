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

import { provideZonelessChangeDetection } from '@angular/core';
import { TestBed } from '@angular/core/testing';
import { UiCopyButton } from './copy-button';

describe('UiCopyButton', () => {
  beforeEach(() => TestBed.configureTestingModule({ providers: [provideZonelessChangeDetection()] }));

  it('copies the text and confirms', async () => {
    const writeText = spyOn(navigator.clipboard, 'writeText').and.resolveTo();
    const fixture = TestBed.createComponent(UiCopyButton);
    fixture.componentRef.setInput('text', '<script></script>');
    fixture.componentRef.setInput('label', 'script tag');
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('button').getAttribute('aria-label')).toBe('Copy script tag');
    await fixture.componentInstance.copy();
    fixture.detectChanges();
    expect(writeText).toHaveBeenCalledWith('<script></script>');
    expect(fixture.nativeElement.textContent).toContain('Copied');
  });

  it('shows the icon alone when compact, keeping its label', () => {
    const fixture = TestBed.createComponent(UiCopyButton);
    fixture.componentRef.setInput('text', 'x');
    fixture.componentRef.setInput('label', 'session id');
    fixture.componentRef.setInput('compact', true);
    fixture.detectChanges();
    const button = fixture.nativeElement.querySelector('button') as HTMLButtonElement;
    expect(button.textContent!.trim()).toBe('');
    expect(button.getAttribute('aria-label')).toBe('Copy session id');
    expect(button.title).toBe('Copy session id');
    expect(button.classList).toContain('ui-btn-icon');
  });

  it('goes back to Copy after a moment', async () => {
    spyOn(navigator.clipboard, 'writeText').and.resolveTo();
    jasmine.clock().install();
    try {
      const fixture = TestBed.createComponent(UiCopyButton);
      fixture.componentRef.setInput('text', 'x');
      await fixture.componentInstance.copy();
      expect(fixture.componentInstance.copied()).toBeTrue();
      jasmine.clock().tick(1500);
      expect(fixture.componentInstance.copied()).toBeFalse();
    } finally {
      jasmine.clock().uninstall();
    }
  });

  it('stays quiet when the clipboard is unavailable', async () => {
    spyOn(navigator.clipboard, 'writeText').and.rejectWith(new Error('denied'));
    const fixture = TestBed.createComponent(UiCopyButton);
    fixture.componentRef.setInput('text', 'x');
    await fixture.componentInstance.copy();
    expect(fixture.componentInstance.copied()).toBeFalse();
  });
});
