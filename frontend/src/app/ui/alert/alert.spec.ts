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

import { Component, provideZonelessChangeDetection } from '@angular/core';
import { TestBed } from '@angular/core/testing';
import { UiAlert } from './alert';

@Component({
  imports: [UiAlert],
  template: `
    <ui-alert data-testid="error">Failed. <button alert-actions data-testid="retry">Retry</button></ui-alert>
    <ui-alert tone="success" data-testid="ok">Saved.</ui-alert>
  `,
})
class AlertHost {}

describe('UiAlert', () => {
  beforeEach(() => TestBed.configureTestingModule({ providers: [provideZonelessChangeDetection()] }));

  it('is an error alert by default, a status otherwise, with its actions at the end', () => {
    const fixture = TestBed.createComponent(AlertHost);
    fixture.detectChanges();
    const el: HTMLElement = fixture.nativeElement;
    const error = el.querySelector('[data-testid="error"]')!;
    expect(error.getAttribute('tone')).toBe('error');
    expect(error.getAttribute('role')).toBe('alert');
    expect(error.querySelector('i')!.className).toContain('fa-circle-exclamation');
    expect(error.querySelector('.ui-alert-actions [data-testid="retry"]')).toBeTruthy();
    const ok = el.querySelector('[data-testid="ok"]')!;
    expect(ok.getAttribute('tone')).toBe('success');
    expect(ok.getAttribute('role')).toBe('status');
    expect(ok.querySelector('i')!.className).toContain('fa-circle-check');
  });
});
