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
import { HttpErrorResponse } from '@angular/common/http';
import { Router, provideRouter } from '@angular/router';
import { NEVER, of, throwError } from 'rxjs';
import { RUN_CALL_LIMIT, RunModelCalls, callStatus } from './run-model-calls';
import { AgentService } from '../../services/agent-service';
import { LLMCallStat } from '../../models/llm-call-stat';
import { makeModelCall, settle } from '../../../testing';

describe('RunModelCalls', () => {
  let fixture: ComponentFixture<RunModelCalls>;
  let el: HTMLElement;
  let agentService: jasmine.SpyObj<AgentService>;
  let loaded: LLMCallStat[][];

  async function render(runId = 'r1'): Promise<void> {
    fixture = TestBed.createComponent(RunModelCalls);
    el = fixture.nativeElement;
    fixture.componentRef.setInput('projectId', 'p1');
    fixture.componentRef.setInput('runId', runId);
    fixture.componentInstance.loaded.subscribe((calls) => loaded.push(calls));
    fixture.detectChanges();
    await settle(fixture);
  }

  function query(testId: string): HTMLElement | null {
    return el.querySelector(`[data-testid="${testId}"]`);
  }

  beforeEach(async () => {
    loaded = [];
    agentService = jasmine.createSpyObj('AgentService', ['getLLMCallStats']);
    await TestBed.configureTestingModule({
      imports: [RunModelCalls],
      providers: [provideRouter([]), { provide: AgentService, useValue: agentService }],
    }).compileComponents();
  });

  it('loads the calls of its run, oldest first, and hands them over', async () => {
    agentService.getLLMCallStats.and.returnValue(
      of([makeModelCall({ id: 'c2', model: 'b' }), makeModelCall({ id: 'c1', model: 'a' })])
    );
    await render();
    expect(agentService.getLLMCallStats).toHaveBeenCalledWith('p1', undefined, RUN_CALL_LIMIT, 0, { runId: 'r1' });
    const rows = el.querySelectorAll('[data-testid="model-call-row"]');
    expect(Array.from(rows).map((r) => r.querySelector('.mono')!.textContent)).toEqual(['a', 'b']);
    expect(loaded.at(-1)!.map((c) => c.id)).toEqual(['c1', 'c2']);
    expect(query('model-calls-capped')).toBeNull();
  });

  it('shows a loading state, then an empty one', async () => {
    agentService.getLLMCallStats.and.returnValue(NEVER);
    await render();
    expect(el.querySelector('ui-loading')).not.toBeNull();

    TestBed.resetTestingModule();
    await TestBed.configureTestingModule({
      imports: [RunModelCalls],
      providers: [provideRouter([]), { provide: AgentService, useValue: agentService }],
    }).compileComponents();
    agentService.getLLMCallStats.and.returnValue(of([]));
    await render();
    expect(query('model-calls-empty')!.textContent).toContain('no model calls');
  });

  it('says when the list is cut at the limit', async () => {
    agentService.getLLMCallStats.and.returnValue(
      of(Array.from({ length: RUN_CALL_LIMIT }, (_, i) => makeModelCall({ id: `c${i}` })))
    );
    await render();
    expect(query('model-calls-capped')!.textContent).toContain(`first ${RUN_CALL_LIMIT}`);
  });

  it("shows the server's reason and loads again on Retry", async () => {
    agentService.getLLMCallStats.and.returnValue(
      throwError(() => new HttpErrorResponse({ status: 500, error: { detail: 'Stats table missing' } }))
    );
    await render();
    expect(query('model-calls-error')!.textContent).toContain('Stats table missing');
    expect(loaded.at(-1)).toEqual([]);

    agentService.getLLMCallStats.and.returnValue(of([makeModelCall()]));
    (query('model-calls-retry') as HTMLButtonElement).click();
    await settle(fixture);
    expect(el.querySelectorAll('[data-testid="model-call-row"]').length).toBe(1);
  });

  it('opens a call on the Model calls page, narrowed to the run', async () => {
    agentService.getLLMCallStats.and.returnValue(of([makeModelCall({ id: 'c9' })]));
    await render('r7');
    const navigate = spyOn(TestBed.inject(Router), 'navigate').and.resolveTo(true);
    (el.querySelector('[data-testid="model-call-row"]') as HTMLElement).click();
    expect(navigate).toHaveBeenCalledWith(['/llm-call-stats'], { queryParams: { run_id: 'r7', call: 'c9' } });
    (el.querySelector('[data-testid="model-call-row"]') as HTMLElement).dispatchEvent(
      new KeyboardEvent('keydown', { key: 'Enter' })
    );
    expect(navigate).toHaveBeenCalledTimes(2);
  });

  it('reads a call as ok, error or unknown from its status code', () => {
    expect(callStatus(makeModelCall({ response_code: 200 }))).toBe('ok');
    expect(callStatus(makeModelCall({ response_code: 429 }))).toBe('error');
    expect(callStatus(makeModelCall({ response_code: null }))).toBe('unknown');
  });
});
