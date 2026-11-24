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
import {
  DailyStats,
  activityBars,
  callDurationLines,
  dayLabel,
  durationChartOptions,
  hasActivity,
  runDurationLines,
  tokenBars,
} from './daily-stats';

function stats(): DailyStats {
  return {
    sessions: [
      { date: '2026-11-23', count: 0 },
      { date: '2026-11-24', count: 2 },
    ],
    messages: [
      { date: '2026-11-23', count: 0 },
      { date: '2026-11-24', count: 4 },
    ],
    tasks: [
      { date: '2026-11-23', count: 0 },
      { date: '2026-11-24', count: 9 },
    ],
    runs: {
      pulse: [
        { date: '2026-11-23', count: 0, duration_seconds: 0 },
        { date: '2026-11-24', count: 2, duration_seconds: 10 },
      ],
      other: [
        { date: '2026-11-23', count: 0 },
        { date: '2026-11-24', count: 1, duration_seconds: 3 },
      ],
    },
    llm: {
      a: [
        { date: '2026-11-23', count: 0 },
        { date: '2026-11-24', count: 3, duration_seconds: 6, prompt_tokens: 100, completion_tokens: 10 },
      ],
      b: [
        { date: '2026-11-23', count: 0 },
        { date: '2026-11-24', count: 1, duration_seconds: 2, prompt_tokens: 50, completion_tokens: 5 },
      ],
    },
    embedding: {},
  };
}

describe('daily stats', () => {
  it('stacks sessions, runs over every agent, and messages per day', () => {
    expect(activityBars(stats())).toEqual([
      { day: '2026-11-23', values: [0, 0, 0] },
      { day: '2026-11-24', values: [2, 3, 4] },
    ]);
  });

  it('sums the tokens of every model per day', () => {
    expect(tokenBars(stats())[1]).toEqual({ day: '2026-11-24', values: [150, 15, 0] });
  });

  it('averages a call over every model, and leaves a day without calls empty', () => {
    const lines = callDurationLines(stats());
    expect(lines.labels).toEqual(['23 Nov', '24 Nov']);
    expect(lines.datasets[0].data).toEqual([null, 2]);
    expect(lines.datasets[1].data).toEqual([null, null]);
  });

  it('averages the runs of each agent', () => {
    const lines = runDurationLines(stats());
    expect(lines.datasets.map((d) => d.label)).toEqual(['pulse', 'other']);
    expect(lines.datasets[0].data).toEqual([null, 5]);
    expect(lines.datasets[0].borderColor).not.toEqual(lines.datasets[1].borderColor);
  });

  it('labels a day by its date and month, without a time zone shift', () => {
    expect(dayLabel({ date: '2026-01-01', count: 0 })).toBe('1 Jan');
    expect(dayLabel({ date: '2026-12-31T00:00:00', count: 0 })).toBe('31 Dec');
  });

  it('knows whether anything happened', () => {
    expect(hasActivity(null)).toBeFalse();
    expect(hasActivity(stats())).toBeTrue();
    const quiet = stats();
    quiet.sessions = quiet.sessions.map((s) => ({ ...s, count: 0 }));
    expect(hasActivity(quiet)).toBeFalse();
  });

  it('labels durations in seconds', () => {
    const options = durationChartOptions();
    const ticks = options.scales!['y']!.ticks as { callback: (v: number) => string };
    expect(ticks.callback(1.5)).toBe('1.5 s');
    const label = options.plugins!.tooltip!.callbacks!.label as (item: unknown) => string;
    expect(label({ dataset: { label: 'Model call' }, parsed: { y: 2 } })).toBe('Model call: 2.00 s');
    expect(label({ dataset: { label: 'Model call' }, parsed: { y: null } })).toBe('Model call: — s');
  });
});
