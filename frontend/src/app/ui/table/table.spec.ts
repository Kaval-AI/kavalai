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
import { UiPager, UiTable } from './table';

@Component({
  imports: [UiTable, UiPager],
  template: `
    <ui-table clickable dense>
      <table>
        <tbody>
          <tr class="selected">
            <td>a</td>
          </tr>
        </tbody>
      </table>
    </ui-table>
    <ui-pager
      [offset]="offset()"
      [count]="count()"
      [total]="total()"
      noun="pages"
      (previous)="log.push('prev')"
      (next)="log.push('next')"
    />
  `,
})
class TableHost {
  readonly offset = signal(50);
  readonly count = signal(50);
  readonly total = signal(120);
  readonly log: string[] = [];
}

describe('UiTable and UiPager', () => {
  beforeEach(() => TestBed.configureTestingModule({ providers: [provideZonelessChangeDetection()] }));

  it('wraps the table and shows the page range with working buttons', () => {
    const fixture = TestBed.createComponent(TableHost);
    fixture.detectChanges();
    const el: HTMLElement = fixture.nativeElement;
    expect(el.querySelector('ui-table[clickable][dense] table tr.selected')).toBeTruthy();
    expect(el.querySelector('[data-testid="pager-range"]')!.textContent).toBe('Showing 51–100 of 120');
    const buttons = el.querySelectorAll<HTMLButtonElement>('ui-pager button');
    expect(buttons[0]!.disabled).toBeFalse();
    expect(buttons[1]!.disabled).toBeFalse();
    buttons[0]!.click();
    buttons[1]!.click();
    expect(fixture.componentInstance.log).toEqual(['prev', 'next']);
  });

  it('describes a single page by its total and disables both buttons', () => {
    const fixture = TestBed.createComponent(TableHost);
    fixture.componentInstance.offset.set(0);
    fixture.componentInstance.count.set(7);
    fixture.componentInstance.total.set(7);
    fixture.detectChanges();
    const el: HTMLElement = fixture.nativeElement;
    expect(el.querySelector('[data-testid="pager-range"]')!.textContent).toBe('7 pages');
    const buttons = el.querySelectorAll<HTMLButtonElement>('ui-pager button');
    expect(buttons[0]!.disabled).toBeTrue();
    expect(buttons[1]!.disabled).toBeTrue();
  });

  it('says 0 when there is nothing', () => {
    const fixture = TestBed.createComponent(TableHost);
    fixture.componentInstance.offset.set(0);
    fixture.componentInstance.count.set(0);
    fixture.componentInstance.total.set(0);
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('[data-testid="pager-range"]').textContent).toBe('0 pages');
  });
});
