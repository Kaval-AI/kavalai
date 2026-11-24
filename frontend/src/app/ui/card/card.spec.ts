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
import { UiCard } from './card';

@Component({
  imports: [UiCard],
  template: `
    <ui-card title="Details" icon="fa-circle-info" tone="danger">
      <button card-actions data-testid="action">Edit</button>
      <p data-testid="body">Body</p>
    </ui-card>
    <ui-card data-testid="plain"><p>Untitled</p></ui-card>
    <ui-card selected data-testid="selected"><p>Current</p></ui-card>
    <ui-card title="Acme" titleLink="/websites/s1" data-testid="linked"><p>A website</p></ui-card>
  `,
})
class CardHost {}

@Component({
  imports: [UiCard],
  template: `
    <div style="display: grid; grid-template-columns: 1fr 1fr; width: 600px">
      <ui-card fill title="Short" data-testid="short">
        <p>One line</p>
        <button card-footer data-testid="short-action">Go</button>
      </ui-card>
      <ui-card fill title="Tall" data-testid="tall">
        <p>One line</p>
        <p>Two lines</p>
        <p>Three lines</p>
        <button card-footer>Go</button>
      </ui-card>
    </div>
  `,
})
class FillHost {}

describe('UiCard', () => {
  beforeEach(() =>
    TestBed.configureTestingModule({ providers: [provideZonelessChangeDetection(), provideRouter([])] }),
  );

  it('renders the lime-ruled title with its icon and actions, then the body', () => {
    const fixture = TestBed.createComponent(CardHost);
    fixture.detectChanges();
    const el: HTMLElement = fixture.nativeElement;
    const card = el.querySelector('ui-card')!;
    expect(card.getAttribute('tone')).toBe('danger');
    const title = card.querySelector('.ui-card-title')!;
    expect(title.textContent).toContain('Details');
    expect(title.querySelector('i')!.className).toContain('fa-circle-info');
    expect(title.querySelector('.ui-card-actions [data-testid="action"]')).toBeTruthy();
    expect(card.querySelector('.ui-card-body [data-testid="body"]')).toBeTruthy();
  });

  it('rules the selected card in lime', () => {
    const fixture = TestBed.createComponent(CardHost);
    fixture.detectChanges();
    const el: HTMLElement = fixture.nativeElement;
    // The rule is unencapsulated and reaches the body, so assert what it paints:
    // a page stylesheet cannot reach in here, which is how it once went dead.
    const body = el.querySelector('[data-testid="selected"] .ui-card-body')!;
    const plain = el.querySelector('[data-testid="plain"] .ui-card-body')!;
    expect(getComputedStyle(body).boxShadow).toContain('inset');
    expect(getComputedStyle(plain).boxShadow).toBe('none');
  });

  it('links the heading of a card that stands for a thing with its own page', () => {
    const fixture = TestBed.createComponent(CardHost);
    fixture.detectChanges();
    const el: HTMLElement = fixture.nativeElement;
    const link = el.querySelector('[data-testid="linked"] .ui-card-title-link')!;
    expect(link.textContent).toContain('Acme');
    expect(link.getAttribute('href')).toBe('/websites/s1');
    expect(el.querySelector('[data-testid="plain"] .ui-card-title-link')).toBeNull();
  });

  it('closes in its footer, and has none without footer content', () => {
    const fixture = TestBed.createComponent(CardHost);
    fixture.detectChanges();
    const el: HTMLElement = fixture.nativeElement;
    const footer = el.querySelector('[data-testid="plain"] .ui-card-footer')!;
    expect(getComputedStyle(footer).display).toBe('none');
  });

  it('fills its grid row and ends side by side cards on one line', () => {
    const fixture = TestBed.createComponent(FillHost);
    document.body.appendChild(fixture.nativeElement);
    fixture.detectChanges();
    const el: HTMLElement = fixture.nativeElement;
    const short = el.querySelector('[data-testid="short"]')!.getBoundingClientRect();
    const tall = el.querySelector('[data-testid="tall"]')!.getBoundingClientRect();
    expect(el.querySelector('[data-testid="short"]')!.hasAttribute('fill')).toBeTrue();
    expect(short.height).toBe(tall.height);
    const buttons = el.querySelectorAll('.ui-card-footer button');
    expect(buttons[0].getBoundingClientRect().bottom).toBe(buttons[1].getBoundingClientRect().bottom);
    fixture.nativeElement.remove();
  });

  it('has no heading without a title', () => {
    const fixture = TestBed.createComponent(CardHost);
    fixture.detectChanges();
    const plain = (fixture.nativeElement as HTMLElement).querySelector('[data-testid="plain"]')!;
    expect(plain.querySelector('.ui-card-title')).toBeNull();
    expect(plain.getAttribute('tone')).toBe('default');
  });
});
