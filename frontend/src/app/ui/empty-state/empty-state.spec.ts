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
import { UiEmptyState } from './empty-state';

@Component({
  imports: [UiEmptyState],
  template: `
    <ui-empty-state icon="fa-robot" title="No chatbots yet">
      Create one.
      <a empty-actions href="/bots/new" data-testid="cta">Create</a>
    </ui-empty-state>
  `,
})
class EmptyHost {}

describe('UiEmptyState', () => {
  beforeEach(() => TestBed.configureTestingModule({ providers: [provideZonelessChangeDetection()] }));

  it('renders the icon, title, help and call to action', () => {
    const fixture = TestBed.createComponent(EmptyHost);
    fixture.detectChanges();
    const empty = (fixture.nativeElement as HTMLElement).querySelector('ui-empty-state')!;
    expect(empty.querySelector('i')!.className).toContain('fa-robot');
    expect(empty.querySelector('h2')!.textContent).toBe('No chatbots yet');
    expect(empty.querySelector('p')!.textContent).toContain('Create one.');
    expect(empty.querySelector('.ui-empty-actions [data-testid="cta"]')).toBeTruthy();
  });
});
