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

import { DatePipe } from '@angular/common';
import { Component, computed, input } from '@angular/core';

/** One bar: stacked `values`, one per series. */
export interface ChartBar {
  day: string;
  values: number[];
}

export interface ChartSeries {
  label: string;
  /** Any CSS colour. */
  color: string;
}

/** Fill a day range so quiet days show as gaps: one bar per day between start and end (UTC dates). */
export function fillDays<T extends { day: string }>(
  rows: T[],
  start: string | Date,
  end: string | Date,
  zero: () => T,
): T[] {
  const byDay = new Map(rows.map((r) => [r.day, r]));
  const s = new Date(start);
  const e = new Date(end);
  const cursor = new Date(Date.UTC(s.getUTCFullYear(), s.getUTCMonth(), s.getUTCDate()));
  const last = new Date(Date.UTC(e.getUTCFullYear(), e.getUTCMonth(), e.getUTCDate()));
  const out: T[] = [];
  while (cursor <= last) {
    const key = cursor.toISOString().slice(0, 10);
    out.push(byDay.get(key) ?? { ...zero(), day: key });
    cursor.setUTCDate(cursor.getUTCDate() + 1);
  }
  return out;
}

/** A small stacked bar chart per day, drawn as SVG so it needs no library. */
@Component({
  selector: 'app-bar-chart',
  imports: [DatePipe],
  templateUrl: './bar-chart.html',
  styleUrl: './bar-chart.css',
})
export class BarChart {
  readonly bars = input.required<ChartBar[]>();
  readonly series = input.required<ChartSeries[]>();
  readonly label = input('Per day');
  readonly empty = input('Nothing in this range yet.');

  readonly max = computed(() => Math.max(0, ...this.bars().map((b) => b.values.reduce((a, v) => a + v, 0))));

  /** Segments from the bottom up, the first series at the bottom. */
  stack(bar: ChartBar): { y: number; h: number; color: string }[] {
    const max = this.max() || 1;
    let top = 100;
    return bar.values.map((v, i) => {
      const h = (v / max) * 100;
      top -= h;
      return { y: top, h, color: this.series()[i]?.color ?? '#999' };
    });
  }

  describe(bar: ChartBar): string {
    return bar.values.map((v, i) => `${v} ${this.series()[i]?.label ?? ''}`).join(', ');
  }
}
