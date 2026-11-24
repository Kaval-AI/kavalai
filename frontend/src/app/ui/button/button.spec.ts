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
import { provideRouter } from '@angular/router';
import { UiButton } from './button';

@Component({
  imports: [UiButton],
  template: `
    <button uiButton="primary" type="submit" data-testid="primary">Save</button>
    <button uiButton data-testid="default">Cancel</button>
    <button uiButton="icon" size="sm" aria-label="Remove" data-testid="icon"><i class="fa-solid fa-trash"></i></button>
    <a uiButton="ghost" block href="/x" data-testid="link">Open</a>
  `,
})
class Host {}

describe('UiButton', () => {
  beforeEach(() =>
    TestBed.configureTestingModule({
      imports: [Host],
      providers: [provideZonelessChangeDetection(), provideRouter([])],
    }),
  );

  function render() {
    const fixture = TestBed.createComponent(Host);
    fixture.detectChanges();
    return fixture.nativeElement as HTMLElement;
  }

  /** The host class binding does not keep the order of the string, so compare the set. */
  function classesOf(el: HTMLElement, testid: string): string[] {
    return Array.from(el.querySelector(`[data-testid="${testid}"]`)!.classList).sort();
  }

  it('classes the variants and sizes', () => {
    const el = render();
    expect(classesOf(el, 'primary')).toEqual(['ui-btn', 'ui-btn-md', 'ui-btn-primary']);
    expect(classesOf(el, 'icon')).toEqual(['ui-btn', 'ui-btn-icon', 'ui-btn-sm']);
    expect(classesOf(el, 'link')).toEqual(['ui-btn', 'ui-btn-block', 'ui-btn-ghost', 'ui-btn-md']);
  });

  it('is an outlined secondary button by default', () => {
    expect(classesOf(render(), 'default')).toEqual(['ui-btn', 'ui-btn-md', 'ui-btn-secondary']);
  });

  it('keeps the native element and its attributes', () => {
    const el = render();
    expect((el.querySelector('[data-testid="primary"]') as HTMLButtonElement).type).toBe('submit');
    expect(el.querySelector('[data-testid="icon"]')!.getAttribute('aria-label')).toBe('Remove');
    expect(el.querySelector('[data-testid="link"]')!.tagName).toBe('A');
  });
});
