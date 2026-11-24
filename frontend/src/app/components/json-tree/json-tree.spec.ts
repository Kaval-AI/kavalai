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
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { JsonTreeComponent } from './json-tree';
import { settle } from '../../../testing';

describe('JsonTreeComponent', () => {
  let fixture: ComponentFixture<JsonTreeComponent>;
  let el: HTMLElement;

  async function render(data: unknown, isExpanded: boolean | null = null): Promise<void> {
    fixture = TestBed.createComponent(JsonTreeComponent);
    el = fixture.nativeElement;
    fixture.componentRef.setInput('data', data);
    if (isExpanded !== null) fixture.componentRef.setInput('isExpanded', isExpanded);
    fixture.detectChanges();
    await settle(fixture);
  }

  function query<T extends Element = HTMLElement>(testId: string): T | null {
    return el.querySelector<T>(`[data-testid="${testId}"]`);
  }

  /** The visible lines, each as its key and value text. */
  function lines(): string[] {
    return Array.from(el.querySelectorAll('.json-line')).map((line) => {
      const key = line.querySelector('.json-key')?.textContent!.trim();
      const value = line.querySelector('.json-value')!.textContent!.trim();
      return key ? `${key} ${value}` : value;
    });
  }

  function lineOf(key: string): HTMLElement {
    return Array.from(el.querySelectorAll<HTMLElement>('.json-line')).find((line) =>
      line.querySelector('.json-key')?.textContent === `${key}:`,
    )!;
  }

  beforeEach(async () => {
    await TestBed.configureTestingModule({ imports: [JsonTreeComponent] }).compileComponents();
  });

  it('opens the top level and keeps nested levels closed', async () => {
    await render({ name: 'trip_minutes', periods: 2, ok: true, unit: null, facts: ['a', 'b'] });
    expect(lines()).toEqual([
      '{5 keys}',
      'name: "trip_minutes"',
      'periods: 2',
      'ok: true',
      'unit: null',
      'facts: [2 items]',
    ]);
    expect(lineOf('facts').getAttribute('aria-expanded')).toBe('false');
    expect(lineOf('facts').querySelector('.fa-chevron-right')).not.toBeNull();
  });

  it('colours each value by its type', async () => {
    await render({ s: 'x', n: 1, b: false, z: null, o: { k: 1 } });
    expect(lineOf('s').querySelector('.json-value')!.classList).toContain('type-string');
    expect(lineOf('n').querySelector('.json-value')!.classList).toContain('type-number');
    expect(lineOf('b').querySelector('.json-value')!.classList).toContain('type-boolean');
    expect(lineOf('z').querySelector('.json-value')!.classList).toContain('type-null');
    expect(lineOf('o').querySelector('.json-value')!.classList).toContain('type-object');
    expect(lineOf('o').textContent).toContain('{1 key}');
  });

  it('opens and closes a level on a click and from the keyboard', async () => {
    await render({ facts: ['first'] });
    lineOf('facts').click();
    await settle(fixture);
    expect(lines()).toContain('0: "first"');
    expect(lineOf('facts').getAttribute('aria-expanded')).toBe('true');
    expect(lineOf('facts').querySelector('.fa-chevron-down')).not.toBeNull();

    lineOf('facts').dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter' }));
    await settle(fixture);
    expect(lines()).not.toContain('0: "first"');
  });

  it('does nothing on a click on a plain value', async () => {
    await render({ n: 1 });
    const line = lineOf('n');
    expect(line.getAttribute('role')).toBeNull();
    line.click();
    await settle(fixture);
    expect(lines()).toEqual(['{1 key}', 'n: 1']);
  });

  it('expands and collapses every level', async () => {
    await render({ a: { b: { c: [1] } } });
    // Each level opens its children a tick after it renders them, as in the app.
    fixture.autoDetectChanges(true);
    query<HTMLButtonElement>('json-expand-all')!.click();
    for (let i = 0; i < 4; i++) await settle(fixture);
    expect(lines()).toEqual(['{1 key}', 'a: {1 key}', 'b: {1 key}', 'c: [1 item]', '0: 1']);

    query<HTMLButtonElement>('json-collapse-all')!.click();
    for (let i = 0; i < 4; i++) await settle(fixture);
    expect(lines()).toEqual(['{1 key}']);
  });

  it('copies the value as indented JSON', async () => {
    const write = spyOn(navigator.clipboard, 'writeText').and.resolveTo();
    await render({ a: [1] });
    const copy = query('json-controls')!.querySelector<HTMLButtonElement>('button')!;
    expect(copy.getAttribute('aria-label')).toBe('Copy JSON');
    copy.click();
    // Not settle: it would wait out the timer that turns "Copied" back into "Copy".
    await new Promise((resolve) => setTimeout(resolve, 0));
    fixture.detectChanges();
    expect(write).toHaveBeenCalledOnceWith('{\n  "a": [\n    1\n  ]\n}');
    expect(copy.textContent).toContain('Copied');
  });

  it('shows no controls for a value with nothing to open', async () => {
    await render('plain');
    expect(query('json-controls')).toBeNull();
    expect(lines()).toEqual(['"plain"']);

    fixture.componentRef.setInput('data', []);
    await settle(fixture);
    expect(query('json-controls')).toBeNull();
    expect(lines()).toEqual(['[0 items]']);
  });

  it('shows a missing value as null', async () => {
    await render(undefined);
    expect(lines()).toEqual(['null']);
    expect(fixture.componentInstance.jsonText).toBe('null');
  });

  it('starts closed when asked to', async () => {
    await render({ a: 1 }, false);
    expect(lines()).toEqual(['{1 key}']);
  });
});
