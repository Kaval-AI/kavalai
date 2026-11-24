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
import { provideRouter } from '@angular/router';
import { UiChip } from './chip';

describe('UiChip', () => {
  beforeEach(() =>
    TestBed.configureTestingModule({ providers: [provideZonelessChangeDetection(), provideRouter([])] }),
  );

  it('is a link with the switch icon when it has one, a plain marker otherwise', () => {
    const fixture = TestBed.createComponent(UiChip);
    fixture.componentRef.setInput('text', 'Acme');
    fixture.componentRef.setInput('detail', 'www.acme.example');
    fixture.componentRef.setInput('link', '/bots');
    fixture.detectChanges();
    const el: HTMLElement = fixture.nativeElement;
    expect(el.querySelector('a.ui-chip')!.getAttribute('href')).toBe('/bots');
    expect(el.querySelector('.ui-chip-text')!.textContent).toBe('Acme');
    expect(el.querySelector('.ui-chip-detail')!.textContent).toBe('www.acme.example');
    expect(el.querySelector('ui-avatar')).toBeTruthy();
    fixture.componentRef.setInput('link', null);
    fixture.detectChanges();
    expect(el.querySelector('a')).toBeNull();
    expect(el.querySelector('span.ui-chip')).toBeTruthy();
  });
});
