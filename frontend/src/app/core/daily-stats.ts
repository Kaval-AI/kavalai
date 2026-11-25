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
import { ChartData, ChartOptions } from 'chart.js';
import { ChartBar, ChartSeries } from '../ui';

/** One day of one series of `GET /agents/stats/{project}`. */
export interface StatsDay {
  date: string;
  count: number;
  /** The day's total, not an average. */
  duration_seconds?: number | null;
  prompt_tokens?: number | null;
  completion_tokens?: number | null;
  total_tokens?: number | null;
}

/**
 * What `GET /agents/stats/{project}` answers: a list per counted record, and
 * per agent (`runs`) or per model (`llm`, `embedding`) a list of the same
 * days. Every list covers the same days, oldest first.
 */
export interface DailyStats {
  sessions: StatsDay[];
  messages: StatsDay[];
  tasks: StatsDay[];
  runs: Record<string, StatsDay[]>;
  llm: Record<string, StatsDay[]>;
  embedding: Record<string, StatsDay[]>;
}

/**
 * Series colours from the visual toolkit. Hex values, not CSS properties: the
 * charts write them into SVG attributes and a canvas, where `var()` does not
 * resolve.
 */
export const CHART_COLORS = {
  lime: '#acc12f',
  verdigris: '#1b998b',
  amethyst: '#b185a7',
  aqua: '#86bbbd',
  amber: '#e67f0d',
  raspberry: '#82204a',
  lime700: '#809537',
  charcoal: '#273e47',
  slate: '#606c74',
};

/** One colour per agent or model, in this order. */
export const CHART_PALETTE = [
  CHART_COLORS.lime,
  CHART_COLORS.verdigris,
  CHART_COLORS.amethyst,
  CHART_COLORS.amber,
  CHART_COLORS.raspberry,
  CHART_COLORS.aqua,
  CHART_COLORS.lime700,
  CHART_COLORS.slate,
];

export const ACTIVITY_SERIES: ChartSeries[] = [
  { label: 'Sessions', color: CHART_COLORS.lime },
  { label: 'Runs', color: CHART_COLORS.verdigris },
  { label: 'Messages', color: CHART_COLORS.amethyst },
];

export const TOKEN_SERIES: ChartSeries[] = [
  { label: 'Input tokens', color: CHART_COLORS.amber },
  { label: 'Output tokens', color: CHART_COLORS.raspberry },
  { label: 'Embedding tokens', color: CHART_COLORS.lime700 },
];

function day(entry: StatsDay): string {
  return entry.date.slice(0, 10);
}

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

/** A chart's day label, `24 Nov`, read from the date itself so no time zone shifts it. */
export function dayLabel(entry: StatsDay): string {
  const [, month, date] = day(entry).split('-').map(Number);
  return `${date} ${MONTHS[month - 1]}`;
}

/** The sum over every agent or model of one field, per day. */
function sumPerDay(
  groups: Record<string, StatsDay[]>,
  days: number,
  field: keyof StatsDay
): number[] {
  return Array.from({ length: days }, (_, i) =>
    Object.values(groups).reduce((total, list) => total + Number(list[i]?.[field] ?? 0), 0)
  );
}

/** The average of a day's total duration over its count, `null` on a day without any. */
function average(entry: StatsDay | undefined): number | null {
  if (!entry?.count) return null;
  return (entry.duration_seconds ?? 0) / entry.count;
}

/** Sessions, runs and messages per day, stacked. */
export function activityBars(stats: DailyStats): ChartBar[] {
  const runs = sumPerDay(stats.runs, stats.sessions.length, 'count');
  return stats.sessions.map((entry, i) => ({
    day: day(entry),
    values: [entry.count, runs[i], stats.messages[i]?.count ?? 0],
  }));
}

/** Input, output and embedding tokens per day, summed over the models. */
export function tokenBars(stats: DailyStats): ChartBar[] {
  const days = stats.sessions.length;
  const input = sumPerDay(stats.llm, days, 'prompt_tokens');
  const output = sumPerDay(stats.llm, days, 'completion_tokens');
  const embedding = sumPerDay(stats.embedding, days, 'total_tokens');
  return stats.sessions.map((entry, i) => ({
    day: day(entry),
    values: [input[i], output[i], embedding[i]],
  }));
}

function line(label: string, data: (number | null)[], color: string) {
  return {
    label,
    data,
    borderColor: color,
    backgroundColor: color,
    borderWidth: 2,
    pointRadius: 3,
    pointHoverRadius: 5,
    tension: 0.2,
    spanGaps: true,
  };
}

function averageOver(groups: Record<string, StatsDay[]>, days: number): (number | null)[] {
  return Array.from({ length: days }, (_, i) => {
    const entries = Object.values(groups).map((list) => list[i]);
    const count = entries.reduce((total, e) => total + (e?.count ?? 0), 0);
    const seconds = entries.reduce((total, e) => total + (e?.duration_seconds ?? 0), 0);
    return count ? seconds / count : null;
  });
}

/** The average model call and embedding call, in seconds, per day. */
export function callDurationLines(stats: DailyStats): ChartData<'line', (number | null)[], string> {
  const days = stats.sessions.length;
  return {
    labels: stats.sessions.map(dayLabel),
    datasets: [
      line('Model call', averageOver(stats.llm, days), CHART_COLORS.amber),
      line('Embedding call', averageOver(stats.embedding, days), CHART_COLORS.lime700),
    ],
  };
}

/** The average run of each agent, in seconds, per day. */
export function runDurationLines(stats: DailyStats): ChartData<'line', (number | null)[], string> {
  return {
    labels: stats.sessions.map(dayLabel),
    datasets: Object.entries(stats.runs).map(([agent, list], i) =>
      line(agent, list.map(average), CHART_PALETTE[i % CHART_PALETTE.length])
    ),
  };
}

/** Whether any day of any series counted something. */
export function hasActivity(stats: DailyStats | null): boolean {
  return !!stats && stats.sessions.some((entry) => entry.count > 0);
}

/** Line chart options in the toolkit's type and colours, values in seconds. */
export function durationChartOptions(): ChartOptions<'line'> {
  const mono = { family: "'Space Mono', monospace", size: 11 };
  return {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: 'index', intersect: false },
    scales: {
      x: {
        grid: { display: false },
        ticks: { font: mono, color: CHART_COLORS.slate },
      },
      y: {
        beginAtZero: true,
        grid: { color: '#e5e5e8' },
        border: { display: false },
        ticks: { font: mono, color: CHART_COLORS.slate, callback: (value) => `${value} s` },
      },
    },
    plugins: {
      legend: {
        position: 'bottom',
        labels: {
          boxWidth: 8,
          usePointStyle: true,
          pointStyle: 'rect',
          font: { family: "'Space Grotesk', sans-serif", size: 12 },
          color: CHART_COLORS.charcoal,
        },
      },
      tooltip: {
        backgroundColor: CHART_COLORS.charcoal,
        titleFont: mono,
        bodyFont: { family: "'Space Grotesk', sans-serif", size: 12 },
        callbacks: {
          label: (item) =>
            `${item.dataset.label}: ${item.parsed.y === null ? '—' : item.parsed.y.toFixed(2)} s`,
        },
      },
    },
  };
}
