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
import { formatDuration, shortId } from './format';

describe('formatDuration', () => {
  it('writes milliseconds below a second', () => {
    expect(formatDuration(0)).toBe('0 ms');
    expect(formatDuration(0.2494)).toBe('249 ms');
  });

  it('writes seconds with one decimal below a minute', () => {
    expect(formatDuration(1)).toBe('1.0 s');
    expect(formatDuration(7.169)).toBe('7.2 s');
  });

  it('writes minutes and whole seconds from a minute on', () => {
    expect(formatDuration(60)).toBe('1 min 0 s');
    expect(formatDuration(125.4)).toBe('2 min 5 s');
  });

  it('writes a dash for a missing value', () => {
    expect(formatDuration(null)).toBe('—');
    expect(formatDuration(undefined)).toBe('—');
    expect(formatDuration(NaN)).toBe('—');
  });
});

describe('shortId', () => {
  it('keeps the first eight characters', () => {
    expect(shortId('d333962d-475e-4523-9555-fea1d6611e52')).toBe('d333962d');
  });

  it('is empty for a missing id', () => {
    expect(shortId(null)).toBe('');
    expect(shortId(undefined)).toBe('');
  });
});
