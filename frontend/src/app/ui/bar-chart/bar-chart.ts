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

const COMPACT = new Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 1 });

/**
 * Gridlines at round values: the smallest step of 1, 2 or 5 times a power of
 * ten, and never below 1 since the bars count things, that reaches `max` in at
 * most five steps. The last tick is the top of the scale.
 */
export function niceTicks(max: number): number[] {
  if (max <= 0) return [0, 1];
  const magnitude = Math.pow(10, Math.floor(Math.log10(max)));
  const step =
    [1, 2, 5, 10].map((m) => m * magnitude).find((s) => s >= 1 && max / s <= 5) ?? 10 * magnitude;
  const count = Math.ceil(max / step - 1e-9);
  return Array.from({ length: count + 1 }, (_, i) => Number((i * step).toPrecision(12)));
}

/**
 * A small stacked bar chart per day, drawn as SVG so it needs no library. The
 * scale on the left runs from zero to a round value at or above the tallest
 * bar; `unit` names what it counts ("tokens").
 */
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
  /** What the scale counts, shown above it: `tokens`, `records`. */
  readonly unit = input('');

  readonly max = computed(() => Math.max(0, ...this.bars().map((b) => b.values.reduce((a, v) => a + v, 0))));

  readonly ticks = computed(() => niceTicks(this.max()));
  readonly top = computed(() => this.ticks()[this.ticks().length - 1]);

  /** A tick's label: `0`, `500`, `4K`, `1.2M`. */
  tickLabel(value: number): string {
    return COMPACT.format(value);
  }

  /** A tick's height on the plot, in percent of it. */
  tickHeight(value: number): number {
    return (value / this.top()) * 100;
  }

  /** Segments from the bottom up, the first series at the bottom, on the scale's 0–100. */
  stack(bar: ChartBar): { y: number; h: number; color: string }[] {
    const max = this.top();
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
