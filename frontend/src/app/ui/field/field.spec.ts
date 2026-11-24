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

import { Component, provideZonelessChangeDetection, signal } from '@angular/core';
import { TestBed } from '@angular/core/testing';
import { UiField } from './field';

@Component({
  imports: [UiField],
  template: `
    <ui-field label="Name" for="name" hint="Shown in the header." [error]="error()">
      <input id="name" />
    </ui-field>
  `,
})
class FieldHost {
  readonly error = signal('');
}

describe('UiField', () => {
  beforeEach(() => TestBed.configureTestingModule({ providers: [provideZonelessChangeDetection()] }));

  it('labels the control and shows the hint, then the error when there is one', () => {
    const fixture = TestBed.createComponent(FieldHost);
    fixture.detectChanges();
    const el: HTMLElement = fixture.nativeElement;
    expect(el.querySelector('label')!.getAttribute('for')).toBe('name');
    expect(el.querySelector('label')!.textContent).toBe('Name');
    expect(el.querySelector('.ui-field-hint')!.textContent).toBe('Shown in the header.');
    expect(el.querySelector('.ui-field-error')).toBeNull();
    fixture.componentInstance.error.set('Required');
    fixture.detectChanges();
    expect(el.querySelector('.ui-field-error')!.textContent).toBe('Required');
  });
});
