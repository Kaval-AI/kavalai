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
import { UiStat, UiStats } from './stat';

@Component({
  imports: [UiStats, UiStat],
  template: `
    <ui-stats>
      <ui-stat label="Messages" hint="today">12</ui-stat>
      <ui-stat label="Accounts" link="/staff/accounts">3</ui-stat>
      <ui-stat label="Queued" link="/staff/queues/crawl" [queryParams]="{ status: 'queued' }">2</ui-stat>
    </ui-stats>
  `,
})
class StatsHost {}

describe('UiStats', () => {
  beforeEach(() =>
    TestBed.configureTestingModule({ providers: [provideZonelessChangeDetection(), provideRouter([])] }),
  );

  it('renders tiles with their label, value and hint, linking when asked', () => {
    const fixture = TestBed.createComponent(StatsHost);
    fixture.detectChanges();
    const tiles = (fixture.nativeElement as HTMLElement).querySelectorAll('ui-stat');
    expect(tiles.length).toBe(3);
    expect(tiles[0]!.querySelector('.ui-stat-label')!.textContent).toBe('Messages');
    expect(tiles[0]!.querySelector('.ui-stat-value')!.textContent!.trim()).toBe('12');
    expect(tiles[0]!.querySelector('.ui-stat-hint')!.textContent).toBe('today');
    expect(tiles[0]!.querySelector('a')).toBeNull();
    expect(tiles[0]!.classList).not.toContain('ui-stat-linked');
    const linked = tiles[1]!.querySelector('a.ui-stat-link')!;
    expect(linked.getAttribute('href')).toBe('/staff/accounts');
    expect(linked.getAttribute('aria-label')).toBe('Accounts');
    expect(tiles[1]!.querySelector('.ui-stat-value')!.textContent!.trim()).toBe('3');
    expect(tiles[2]!.querySelector('a.ui-stat-link')!.getAttribute('href')).toBe('/staff/queues/crawl?status=queued');
  });
});
