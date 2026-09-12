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
import { ActivatedRoute, provideRouter } from '@angular/router';
import { BehaviorSubject, of, throwError } from 'rxjs';

import { LlmCallStatsPage } from './llm-call-stats-page';
import { AgentService } from '../../services/agent-service';
import { UserService } from '../../services/user-service';
import { LLMCallStat } from '../../models/llm-call-stat';

describe('LlmCallStatsPage', () => {
  let component: LlmCallStatsPage;
  let fixture: ComponentFixture<LlmCallStatsPage>;
  let agentServiceSpy: jasmine.SpyObj<AgentService>;
  let queryParams: BehaviorSubject<any>;
  let userDetails: BehaviorSubject<any>;

  const call = (id: string, overrides: Partial<LLMCallStat> = {}): LLMCallStat => ({
    id, call_type: 'llm', model: 'openai/gpt-5-mini', agent_id: null,
    session_id: 'sess-1234-5678', run_id: 'run-abcd-efgh', response_code: 200,
    prompt_tokens: 10, completion_tokens: 5, total_tokens: 15,
    cached_prompt_tokens: null, reasoning_tokens: null, duration_seconds: 0.5,
    request_data: null, response_data: null,
    created_at: '2026-09-01T10:00:00Z', updated_at: '2026-09-01T10:00:00Z',
    ...overrides
  });
  const page = (size: number) => Array.from({ length: size }, (_, i) => call(`c${i}`));

  beforeEach(async () => {
    agentServiceSpy = jasmine.createSpyObj('AgentService', ['getLLMCallStats']);
    agentServiceSpy.getLLMCallStats.and.returnValue(of([call('c1')]));
    queryParams = new BehaviorSubject<any>({});
    userDetails = new BehaviorSubject<any>({ active_project_id: 'proj1' });

    await TestBed.configureTestingModule({
      imports: [LlmCallStatsPage],
      providers: [
        provideRouter([]),
        { provide: AgentService, useValue: agentServiceSpy },
        { provide: UserService, useValue: { userDetails: userDetails.asObservable() } },
        { provide: ActivatedRoute, useValue: { queryParams: queryParams.asObservable() } },
      ]
    }).compileComponents();

    fixture = TestBed.createComponent(LlmCallStatsPage);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('loads the project\'s calls once, unfiltered', () => {
    expect(agentServiceSpy.getLLMCallStats).toHaveBeenCalledOnceWith(
      'proj1', undefined, 20, 0, { sessionId: undefined, runId: undefined }
    );
    expect(component.stats.length).toBe(1);
    expect(component.isFiltered).toBeFalse();
  });

  it('forwards the type, conversation and run filters of the query parameters', () => {
    queryParams.next({ call_type: 'embedding', session_id: 'sess-1', run_id: 'run-1' });
    fixture.detectChanges();

    expect(agentServiceSpy.getLLMCallStats).toHaveBeenCalledWith(
      'proj1', 'embedding', 20, 0, { sessionId: 'sess-1', runId: 'run-1' }
    );
    expect(component.stats.length).toBe(1);
    expect(component.isFiltered).toBeTrue();
    const header: HTMLElement = fixture.nativeElement.querySelector('.header-section');
    expect(header.textContent).toContain('Filtered by run: run-1');
    expect(header.textContent).toContain('Filtered by conversation:');
    expect(header.textContent).toContain('Show all calls');
  });

  it('links a call to its run and conversation', () => {
    const card: HTMLElement = fixture.nativeElement.querySelector('.stat-card');
    expect(card.querySelector('.run-filter-link')?.textContent?.trim()).toBe('run-abcd');
    expect(card.querySelector('.conversation-link')?.getAttribute('href')).toBe('/conversations/sess-1234-5678');
  });

  it('shows no attribution for a call made outside a run', () => {
    agentServiceSpy.getLLMCallStats.and.returnValue(of([call('c2', { session_id: null, run_id: null })]));
    component.reload();
    fixture.detectChanges();

    const card: HTMLElement = fixture.nativeElement.querySelector('.stat-card');
    expect(card.querySelector('.run-filter-link')).toBeNull();
    expect(card.querySelector('.conversation-link')).toBeNull();
  });

  it('pages until a short page arrives', () => {
    agentServiceSpy.getLLMCallStats.and.returnValue(of(page(20)));
    component.reload();
    expect(component.hasMore).toBeTrue();

    agentServiceSpy.getLLMCallStats.and.returnValue(of(page(3)));
    component.loadMore();
    expect(agentServiceSpy.getLLMCallStats).toHaveBeenCalledWith(
      'proj1', undefined, 20, 20, { sessionId: undefined, runId: undefined }
    );
    expect(component.stats.length).toBe(23);
    expect(component.hasMore).toBeFalse();

    const calls = agentServiceSpy.getLLMCallStats.calls.count();
    component.loadMore();
    expect(agentServiceSpy.getLLMCallStats.calls.count()).toBe(calls);
  });

  it('reports a failed load', () => {
    spyOn(console, 'error');
    agentServiceSpy.getLLMCallStats.and.returnValue(throwError(() => new Error('boom')));
    component.reload();

    expect(component.error).toBe('Failed to load model calls');
    expect(component.loading).toBeFalse();
  });

  it('reloads only when the active project changes', () => {
    userDetails.next({ active_project_id: 'proj1' });
    expect(agentServiceSpy.getLLMCallStats).toHaveBeenCalledTimes(1);

    userDetails.next({ active_project_id: 'None' });
    expect(component.projectId).toBeNull();
    expect(component.stats).toEqual([]);
    expect(agentServiceSpy.getLLMCallStats).toHaveBeenCalledTimes(1);
  });

  it('stops listening once destroyed', () => {
    fixture.destroy();
    queryParams.next({ run_id: 'run-2' });
    expect(agentServiceSpy.getLLMCallStats).toHaveBeenCalledTimes(1);
  });
});
