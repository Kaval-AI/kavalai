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
import { BarChart, fillDays } from './bar-chart';

describe('fillDays', () => {
  it('gives every day of the range a row, keeping the ones that exist', () => {
    const rows = fillDays([{ day: '2026-09-02', n: 2 }], '2026-09-01T12:00:00Z', '2026-09-03T12:00:00Z', () => ({
      day: '',
      n: 0,
    }));
    expect(rows).toEqual([
      { day: '2026-09-01', n: 0 },
      { day: '2026-09-02', n: 2 },
      { day: '2026-09-03', n: 0 },
    ]);
  });
});

describe('BarChart', () => {
  it('stacks the series per day and scales to the tallest bar', async () => {
    await TestBed.configureTestingModule({ providers: [provideZonelessChangeDetection()] }).compileComponents();
    const fixture = TestBed.createComponent(BarChart);
    fixture.componentRef.setInput('bars', [
      { day: '2026-09-01', values: [2, 2] },
      { day: '2026-09-02', values: [1, 0] },
    ]);
    fixture.componentRef.setInput('series', [
      { label: 'a', color: 'red' },
      { label: 'b', color: 'blue' },
    ]);
    fixture.detectChanges();
    const chart = fixture.componentInstance;
    expect(chart.max()).toBe(4);
    expect(chart.stack({ day: '2026-09-01', values: [2, 2] })).toEqual([
      { y: 50, h: 50, color: 'red' },
      { y: 0, h: 50, color: 'blue' },
    ]);
    expect(chart.describe({ day: '2026-09-02', values: [1, 0] })).toBe('1 a, 0 b');
    const el: HTMLElement = fixture.nativeElement;
    expect(el.querySelectorAll('svg g').length).toBe(2);
    expect(el.querySelectorAll('rect').length).toBe(4);
    expect(el.querySelector('.legend')?.textContent).toContain('b');
  });

  it('says so when there is nothing to draw', async () => {
    await TestBed.configureTestingModule({ providers: [provideZonelessChangeDetection()] }).compileComponents();
    const fixture = TestBed.createComponent(BarChart);
    fixture.componentRef.setInput('bars', []);
    fixture.componentRef.setInput('series', []);
    fixture.componentRef.setInput('empty', 'Quiet.');
    fixture.detectChanges();
    expect(fixture.nativeElement.textContent).toContain('Quiet.');
  });
});
