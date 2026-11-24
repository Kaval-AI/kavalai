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
import { Router, provideRouter } from '@angular/router';
import { RouterTestingHarness } from '@angular/router/testing';
import { UiNavItem } from './nav-item';

@Component({ template: '' })
class Blank {}

@Component({
  imports: [UiNavItem],
  template: `
    <ui-nav-item link="/bots" icon="fa-robot" label="Chatbots" exact />
    <ui-nav-item link="/bots/b1/settings" label="Prompt & model" compact />
  `,
})
class NavHost {}

describe('UiNavItem', () => {
  beforeEach(() =>
    TestBed.configureTestingModule({
      providers: [
        provideZonelessChangeDetection(),
        provideRouter([
          { path: 'bots', component: Blank },
          { path: 'bots/:id/settings', component: Blank },
        ]),
      ],
    }),
  );

  it('renders the link with its icon and marks the open route active', async () => {
    const harness = await RouterTestingHarness.create();
    const fixture = TestBed.createComponent(NavHost);
    fixture.detectChanges();
    const el: HTMLElement = fixture.nativeElement;
    const links = el.querySelectorAll('a');
    expect(links[0]!.getAttribute('href')).toBe('/bots');
    expect(links[0]!.querySelector('i')!.className).toContain('fa-robot');
    expect(links[0]!.textContent).toContain('Chatbots');
    expect(links[1]!.querySelector('i')).toBeNull();
    expect(el.querySelectorAll('ui-nav-item')[1]!.hasAttribute('compact')).toBeTrue();

    await harness.navigateByUrl('/bots/b1/settings');
    fixture.detectChanges();
    expect(links[0]!.classList).not.toContain('active');
    expect(links[1]!.classList).toContain('active');

    await TestBed.inject(Router).navigateByUrl('/bots');
    fixture.detectChanges();
    expect(links[0]!.classList).toContain('active');
    expect(links[1]!.classList).not.toContain('active');
  });
});
