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
import { UiBadge, UiStatusBadge } from './badge';

describe('UiBadge and UiStatusBadge', () => {
  beforeEach(() => TestBed.configureTestingModule({ providers: [provideZonelessChangeDetection()] }));

  it('classes a badge by tone', () => {
    const fixture = TestBed.createComponent(UiBadge);
    fixture.componentRef.setInput('tone', 'warning');
    fixture.detectChanges();
    expect(Array.from((fixture.nativeElement as HTMLElement).classList).sort()).toEqual([
      'ui-badge',
      'ui-badge-warning',
    ]);
  });

  it('words and colours the known statuses', () => {
    const fixture = TestBed.createComponent(UiStatusBadge);
    fixture.componentRef.setInput('status', 'completed');
    fixture.detectChanges();
    const badge = fixture.nativeElement.querySelector('ui-badge') as HTMLElement;
    expect(badge.textContent!.trim()).toBe('Completed');
    expect(badge.className).toContain('ui-badge-success');
    expect(badge.getAttribute('data-status')).toBe('completed');
    fixture.componentRef.setInput('status', 'failed');
    fixture.detectChanges();
    expect(badge.textContent!.trim()).toBe('Failed');
    expect(badge.className).toContain('ui-badge-error');
    fixture.componentRef.setInput('status', 'running');
    fixture.detectChanges();
    expect(badge.className).toContain('ui-badge-warning');
  });

  it('passes unknown statuses through as neutral', () => {
    const fixture = TestBed.createComponent(UiStatusBadge);
    fixture.componentRef.setInput('status', 'weird');
    fixture.detectChanges();
    const badge = fixture.nativeElement.querySelector('ui-badge') as HTMLElement;
    expect(badge.textContent!.trim()).toBe('weird');
    expect(badge.className).toContain('ui-badge-neutral');
  });
});
