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
import { UiProgress } from './progress';

describe('UiProgress', () => {
  beforeEach(() => TestBed.configureTestingModule({ providers: [provideZonelessChangeDetection()] }));

  it('scales the bar to max and clamps it', () => {
    const fixture = TestBed.createComponent(UiProgress);
    fixture.componentRef.setInput('value', 30);
    fixture.componentRef.setInput('max', 120);
    fixture.detectChanges();
    const el: HTMLElement = fixture.nativeElement;
    expect(el.getAttribute('role')).toBe('progressbar');
    expect(el.getAttribute('aria-valuenow')).toBe('30');
    expect(el.getAttribute('aria-valuemax')).toBe('120');
    expect((el.querySelector('.ui-progress-bar') as HTMLElement).style.width).toBe('25%');
    fixture.componentRef.setInput('value', 500);
    fixture.detectChanges();
    expect((el.querySelector('.ui-progress-bar') as HTMLElement).style.width).toBe('100%');
    fixture.componentRef.setInput('max', 0);
    fixture.detectChanges();
    expect((el.querySelector('.ui-progress-bar') as HTMLElement).style.width).toBe('0%');
  });
});
