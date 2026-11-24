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
import { UiMenu, UiMenuItem } from './menu';

@Component({
  imports: [UiMenu, UiMenuItem],
  template: `
    <div style="position: relative">
      <ui-menu>
        <div menu-header data-testid="header">ann@example.com</div>
        <a uiMenuItem href="/account">Account</a>
        <button uiMenuItem danger data-testid="logout">Log out</button>
      </ui-menu>
    </div>
  `,
})
class MenuHost {}

@Component({
  imports: [UiMenu],
  template: `
    <div style="position: relative">
      <ui-menu align="start" role="dialog" aria-label="Websites to search"><input type="search" /></ui-menu>
    </div>
  `,
})
class PickerHost {}

describe('UiMenu', () => {
  beforeEach(() => TestBed.configureTestingModule({ providers: [provideZonelessChangeDetection()] }));

  it("opens under its parent's right edge by default", () => {
    const fixture = TestBed.createComponent(MenuHost);
    fixture.detectChanges();
    const menu: HTMLElement = fixture.nativeElement.querySelector('ui-menu');
    expect(menu.classList).not.toContain('ui-menu-start');
    expect(getComputedStyle(menu).right).toBe('0px');
  });

  it("holds a picker's form under its parent's left edge", () => {
    const fixture = TestBed.createComponent(PickerHost);
    fixture.detectChanges();
    const menu: HTMLElement = fixture.nativeElement.querySelector('ui-menu');
    expect(menu.getAttribute('role')).toBe('dialog');
    expect(menu.getAttribute('aria-label')).toBe('Websites to search');
    expect(menu.classList).toContain('ui-menu-start');
    expect(getComputedStyle(menu).left).toBe('0px');
  });

  it('has the menu roles and a danger entry', () => {
    const fixture = TestBed.createComponent(MenuHost);
    fixture.detectChanges();
    const el: HTMLElement = fixture.nativeElement;
    expect(el.querySelector('ui-menu')!.getAttribute('role')).toBe('menu');
    expect(el.querySelector('.ui-menu-header [data-testid="header"]')).toBeTruthy();
    const items = el.querySelectorAll('[role="menuitem"]');
    expect(items.length).toBe(2);
    expect(Array.from(items[0]!.classList).sort()).toEqual(['ui-menu-item']);
    expect(Array.from(items[1]!.classList).sort()).toEqual(['ui-menu-item', 'ui-menu-item-danger']);
  });
});
