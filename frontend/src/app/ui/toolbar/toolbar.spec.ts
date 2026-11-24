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
import { UiToolbar } from './toolbar';

@Component({
  imports: [UiToolbar],
  template: `
    <ui-toolbar>
      <input type="search" data-testid="search" />
      <button class="ui-toolbar-end" data-testid="action">New</button>
    </ui-toolbar>
  `,
})
class ToolbarHost {}

describe('UiToolbar', () => {
  beforeEach(() => TestBed.configureTestingModule({ providers: [provideZonelessChangeDetection()] }));

  it('is a toolbar landmark around the filters and actions it is given', () => {
    const fixture = TestBed.createComponent(ToolbarHost);
    fixture.detectChanges();
    const toolbar = (fixture.nativeElement as HTMLElement).querySelector('ui-toolbar')!;
    expect(toolbar.getAttribute('role')).toBe('toolbar');
    expect(toolbar.querySelector('[data-testid="search"]')).toBeTruthy();
    expect(toolbar.querySelector('[data-testid="action"]')).toBeTruthy();
  });
});
