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
import { UiAvatar, initialsOf } from './avatar';

describe('initialsOf', () => {
  it('takes the first letters of the first two words, or two letters of one', () => {
    expect(initialsOf('Ann Example')).toBe('AE');
    expect(initialsOf('bob@example.com')).toBe('BE');
    expect(initialsOf('Acme')).toBe('AC');
    expect(initialsOf('x')).toBe('X');
  });
});

describe('UiAvatar', () => {
  beforeEach(() => TestBed.configureTestingModule({ providers: [provideZonelessChangeDetection()] }));

  it('shows initials in the chamfered box by default', () => {
    const fixture = TestBed.createComponent(UiAvatar);
    fixture.componentRef.setInput('name', 'Acme helper');
    fixture.componentRef.setInput('size', 'sm');
    fixture.detectChanges();
    const span = fixture.nativeElement.querySelector('span') as HTMLElement;
    expect(span.textContent).toBe('AH');
    expect(Array.from(span.classList).sort()).toEqual(['ui-avatar', 'ui-avatar-sm']);
  });

  it('shows the picture, round, when there is one', () => {
    const fixture = TestBed.createComponent(UiAvatar);
    fixture.componentRef.setInput('name', 'Ann');
    fixture.componentRef.setInput('src', 'https://img.example/a.png');
    fixture.componentRef.setInput('round', true);
    fixture.detectChanges();
    const img = fixture.nativeElement.querySelector('img') as HTMLImageElement;
    expect(img.src).toBe('https://img.example/a.png');
    expect(img.className).toContain('ui-avatar-round');
    expect(fixture.nativeElement.querySelector('span')).toBeNull();
  });
});
