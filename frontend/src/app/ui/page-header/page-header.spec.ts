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
import { UiPageHeader } from './page-header';

@Component({
  imports: [UiPageHeader],
  template: `
    <ui-page-header tag="Chatbots" title="Acme" section="Overview" backLink="/bots" backLabel="Chatbots">
      <span page-title data-testid="extra">badge</span>
      <span page-actions data-testid="action">New</span>
      A subtitle.
    </ui-page-header>
  `,
})
class HeaderHost {}

describe('UiPageHeader', () => {
  beforeEach(() =>
    TestBed.configureTestingModule({ providers: [provideZonelessChangeDetection(), provideRouter([])] }),
  );

  it('shows the tag, the back link, the title with its section, the subtitle and the actions', () => {
    const fixture = TestBed.createComponent(HeaderHost);
    fixture.detectChanges();
    const el: HTMLElement = fixture.nativeElement;
    expect(el.querySelector('.section-tag')!.textContent).toBe('Chatbots');
    const back = el.querySelector('.ui-back-link') as HTMLAnchorElement;
    expect(back.getAttribute('href')).toBe('/bots');
    expect(back.textContent).toContain('Chatbots');
    expect(el.querySelector('h1')!.textContent).toContain('Acme');
    expect(el.querySelector('[data-testid="section-title"]')!.textContent).toBe('Overview');
    expect(el.querySelector('.ui-page-title [data-testid="extra"]')).toBeTruthy();
    expect(el.querySelector('.ui-page-subtitle')!.textContent).toContain('A subtitle.');
    expect(el.querySelector('.ui-page-header-actions [data-testid="action"]')).toBeTruthy();
  });

  it('leaves out the tag, section and back link when not given', () => {
    const fixture = TestBed.createComponent(UiPageHeader);
    fixture.componentRef.setInput('title', 'Usage');
    fixture.detectChanges();
    const el: HTMLElement = fixture.nativeElement;
    expect(el.querySelector('.section-tag')).toBeNull();
    expect(el.querySelector('.ui-back-link')).toBeNull();
    expect(el.querySelector('[data-testid="section-title"]')).toBeNull();
    expect(el.querySelector('h1')!.textContent!.trim()).toBe('Usage');
  });
});
