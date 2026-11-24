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
import { ActivatedRoute, Params, Router, provideRouter } from '@angular/router';
import { BehaviorSubject, NEVER, of, throwError } from 'rxjs';
import { LlmCallStatsPage, tokenDays } from './llm-call-stats-page';
import { AgentService } from '../../services/agent-service';
import { UserService } from '../../services/user-service';
import { LLMCallStat, LLMCallSummary } from '../../models/llm-call-stat';
import { UserDetails } from '../../models/user-details';
import { makeModelCall, makeUser, settle } from '../../../testing';

const SUMMARY: LLMCallSummary = {
  totals: {
    calls: 3,
    errors: 1,
    prompt_tokens: 3729,
    completion_tokens: 394,
    cached_prompt_tokens: 120,
    reasoning_tokens: 31,
    total_tokens: 4130,
    average_seconds: 2.1,
  },
  by_model: [
    { model: 'scaleway/glm-5.3', call_type: 'llm', calls: 2, prompt_tokens: 3729, completion_tokens: 394, cached_prompt_tokens: 120, reasoning_tokens: 31, total_tokens: 4123, errors: 1, average_seconds: 2.5 },
    { model: 'fastembed/bge', call_type: 'embedding', calls: 1, prompt_tokens: 7, completion_tokens: 0, cached_prompt_tokens: 0, reasoning_tokens: 0, total_tokens: 7, errors: 0, average_seconds: 0.4 },
  ],
  by_day: [
    { date: '2026-11-22', calls: 1, prompt_tokens: 1000, completion_tokens: 100, embedding_tokens: 0 },
    { date: '2026-11-24', calls: 2, prompt_tokens: 2729, completion_tokens: 294, embedding_tokens: 7 },
  ],
};

describe('LlmCallStatsPage', () => {
  let fixture: ComponentFixture<LlmCallStatsPage>;
  let el: HTMLElement;
  let agentService: jasmine.SpyObj<AgentService>;
  let user: BehaviorSubject<UserDetails | null>;
  let params: BehaviorSubject<Params>;

  const chatCall = makeModelCall({
    id: 'c1-aaaaaaaa-bbbb',
    run_id: 'r1-run-0000-1111',
    session_id: 's1-conv-0000-2222',
    agent_id: 'a1-agent-000-3333',
    cached_prompt_tokens: 512,
    reasoning_tokens: 31,
    request_data: JSON.stringify({
      model: 'glm-5.3',
      messages: [
        { role: 'system', content: 'Write the answer as cards.' },
        { role: 'user', content: 'How many trips last month?' },
      ],
    }),
    response_data: '{"cards": []}',
  });
  const embeddingCall = makeModelCall({
    id: 'c2',
    call_type: 'embedding',
    model: 'fastembed/bge-small',
    session_id: null,
    run_id: 'r2',
    agent_id: null,
    response_code: 429,
    cached_prompt_tokens: null,
    reasoning_tokens: null,
    prompt_tokens: null,
    completion_tokens: null,
    total_tokens: null,
    request_data: { input: [3] },
    response_data: { model: 'bge', usage: { tokens: 3 } },
  });

  const httpError = (detail: string) => new HttpErrorResponse({ status: 500, error: { detail } });

  async function render(): Promise<void> {
    fixture = TestBed.createComponent(LlmCallStatsPage);
    el = fixture.nativeElement;
    fixture.detectChanges();
    await settle(fixture);
  }

  function query<T extends Element = HTMLElement>(testId: string): T | null {
    return el.querySelector<T>(`[data-testid="${testId}"]`);
  }

  function all(testId: string): HTMLElement[] {
    return Array.from(el.querySelectorAll<HTMLElement>(`[data-testid="${testId}"]`));
  }

  function stat(label: string): { value: string; hint: string } {
    const tile = Array.from(query('call-figures')!.querySelectorAll('ui-stat')).find(
      (t) => t.querySelector('.ui-stat-label')!.textContent === label,
    )!;
    return {
      value: tile.querySelector('.ui-stat-value')!.textContent!.trim(),
      hint: tile.querySelector('.ui-stat-hint')?.textContent?.trim() ?? '',
    };
  }

  async function open(row = 0): Promise<void> {
    all('call-row')[row].click();
    await settle(fixture);
  }

  beforeEach(async () => {
    agentService = jasmine.createSpyObj<AgentService>('AgentService', ['getLLMCallStats', 'getLLMCallSummary']);
    agentService.getLLMCallStats.and.returnValue(of([chatCall, embeddingCall]));
    agentService.getLLMCallSummary.and.returnValue(of(SUMMARY));
    user = new BehaviorSubject<UserDetails | null>(makeUser({ active_project_id: 'p1' }));
    params = new BehaviorSubject<Params>({});

    await TestBed.configureTestingModule({
      imports: [LlmCallStatsPage],
      providers: [
        provideRouter([]),
        { provide: AgentService, useValue: agentService },
        { provide: UserService, useValue: { userDetails: user.asObservable() } },
        { provide: ActivatedRoute, useValue: { queryParams: params.asObservable() } },
      ],
    }).compileComponents();
  });

  it('opens with its page header', async () => {
    await render();
    expect(el.querySelector('h1')!.textContent).toContain('Model calls');
    expect(query('active-filters')).toBeNull();
  });

  it('shows a spinner while the calls load', async () => {
    agentService.getLLMCallStats.and.returnValue(NEVER);
    await render();
    expect(el.querySelector('ui-loading')).not.toBeNull();
    expect(query('calls-table')).toBeNull();
  });

  it('shows the server error with Retry when the calls fail to load', async () => {
    agentService.getLLMCallStats.and.returnValue(throwError(() => httpError('Stats table missing')));
    await render();
    expect(query('calls-error')!.textContent).toContain('Stats table missing');

    agentService.getLLMCallStats.and.returnValue(of([chatCall]));
    query<HTMLButtonElement>('calls-retry')!.click();
    await settle(fixture);
    expect(query('calls-error')).toBeNull();
    expect(all('call-row').length).toBe(1);
  });

  it('says so when the project has no model calls', async () => {
    agentService.getLLMCallStats.and.returnValue(of([]));
    await render();
    expect(query('calls-empty')!.textContent).toContain('recorded no model calls');
    expect(query('load-more')).toBeNull();
  });

  it('asks for a project when none is selected', async () => {
    user.next(null);
    await render();
    expect(query('no-project')).not.toBeNull();
    expect(agentService.getLLMCallStats).not.toHaveBeenCalled();
  });

  it('lists the calls with when, model, type, status, duration and tokens', async () => {
    await render();
    expect(agentService.getLLMCallStats).toHaveBeenCalledOnceWith('p1', undefined, 20, 0, {
    });
    const rows = all('call-row');
    expect(rows.length).toBe(2);
    const cells = Array.from(rows[0].querySelectorAll('td')).map((td) => td.textContent!.trim());
    expect(cells.slice(1)).toEqual(['scaleway/glm-5.3', 'llm', 'OK', '2.5 s', '861', '189']);
    const other = Array.from(rows[1].querySelectorAll('td')).map((td) => td.textContent!.trim());
    expect(other.slice(1)).toEqual(['fastembed/bge-small', 'embedding', 'Error', '2.5 s', '—', '—']);
  });

  it('shows a call without a status code as unknown', async () => {
    agentService.getLLMCallStats.and.returnValue(of([makeModelCall({ response_code: null })]));
    await render();
    expect(all('call-row')[0].querySelectorAll('td')[3].textContent!.trim()).toBe('unknown');
  });

  it('narrows the list by the filters in the address and shows them with a way to clear', async () => {
    params.next({ call_type: 'llm', agent_id: 'a1-agent-000-3333', session_id: 's1-conv-0000-2222', run_id: 'r1-run-0000-1111' });
    await render();
    expect(agentService.getLLMCallStats).toHaveBeenCalledOnceWith('p1', 'llm', 20, 0, {
      agentId: 'a1-agent-000-3333',
      sessionId: 's1-conv-0000-2222',
      runId: 'r1-run-0000-1111',
    });
    const chips = ['call_type', 'agent_id', 'session_id', 'run_id'].map((p) => query(`filter-${p}`)!.textContent!.trim());
    expect(chips[0]).toContain('Type');
    expect(chips[0]).toContain('llm');
    expect(chips[1]).toContain('Agent');
    expect(chips[1]).toContain('a1-agent');
    expect(chips[2]).toContain('Conversation');
    expect(chips[3]).toContain('Run');
    expect(chips[3]).toContain('r1-run-0');
    const clear = query<HTMLAnchorElement>('clear-run_id')!;
    expect(clear.getAttribute('aria-label')).toBe('Clear the run filter');
    // The route is not the router's here, so the merge has no current parameters to keep.
    expect(clear.getAttribute('href')).not.toContain('run_id');
    expect(query<HTMLAnchorElement>('clear-filters')!.getAttribute('href')).toBe('/llm-call-stats');
  });

  it('says so when no call matches the filters', async () => {
    params.next({ run_id: 'r9' });
    agentService.getLLMCallStats.and.returnValue(of([]));
    await render();
    expect(query('calls-empty')!.textContent).toContain('No call matches these filters.');
  });

  it('loads again, from the start, when the filters change', async () => {
    await render();
    await open();
    params.next({ run_id: 'r2' });
    await settle(fixture);
    expect(agentService.getLLMCallStats).toHaveBeenCalledWith('p1', undefined, 20, 0, {
      runId: 'r2',
    });
    expect(query('call-detail')).toBeNull();
  });

  it('loads again for another project', async () => {
    await render();
    user.next(makeUser({ active_project_id: 'p2' }));
    await settle(fixture);
    expect(agentService.getLLMCallStats.calls.mostRecent().args[0]).toBe('p2');
    user.next(makeUser({ active_project_id: 'p2', name: 'Renamed' }));
    expect(agentService.getLLMCallStats).toHaveBeenCalledTimes(2);
  });

  it('opens a selected call beside the list with its figures and links', async () => {
    await render();
    expect(query('call-detail')).toBeNull();
    await open();
    const rows = all('call-row');
    expect(rows[0].classList).toContain('selected');
    expect(el.querySelector('.calls-layout')!.classList).toContain('with-detail');
    expect(query('call-detail')!.textContent).toContain('scaleway/glm-5.3');
    expect(stat('Duration').value).toBe('2.5 s');
    expect(stat('Tokens in')).toEqual({ value: '861', hint: '512 cached' });
    expect(stat('Tokens out')).toEqual({ value: '189', hint: '31 reasoning' });
    expect(stat('Total').value).toBe('1,050');
    expect(query<HTMLAnchorElement>('run-link')!.getAttribute('href')).toBe(
      '/conversations/s1-conv-0000-2222/runs/r1-run-0000-1111/tasks',
    );
    expect(query<HTMLAnchorElement>('run-filter-link')!.getAttribute('href')).toContain('run_id=r1-run-0000-1111');
    expect(query<HTMLAnchorElement>('conversation-link')!.getAttribute('href')).toBe('/conversations/s1-conv-0000-2222');
    expect(query<HTMLAnchorElement>('agent-filter-link')!.getAttribute('href')).toContain('agent_id=a1-agent-000-3333');

    query<HTMLButtonElement>('close-detail')!.click();
    await settle(fixture);
    expect(query('call-detail')).toBeNull();
  });

  it('closes the panel on a second click and opens a row from the keyboard', async () => {
    await render();
    await open();
    await open();
    expect(query('call-detail')).toBeNull();
    all('call-row')[1].dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter' }));
    await settle(fixture);
    expect(query('call-detail')!.textContent).toContain('fastembed/bge-small');
  });

  it('shows the request as role-labelled messages and the response as text', async () => {
    await render();
    await open();
    expect(query<HTMLInputElement>('show-text')!.checked).toBeTrue();
    const messages = all('request-message');
    expect(messages.map((m) => m.getAttribute('data-role'))).toEqual(['system', 'user']);
    expect(messages[1].querySelector('.msg-role')!.textContent!.trim()).toBe('user');
    expect(messages[1].querySelector('.msg-text')!.textContent).toBe('How many trips last month?');
    expect(query('response-text')!.textContent).toContain('"cards": []');
    expect(query('call-request')!.querySelector('app-json-tree')).toBeNull();
  });

  it('shows the request and response as JSON trees without the text view', async () => {
    await render();
    await open();
    const toggle = query<HTMLInputElement>('show-text')!;
    toggle.click();
    await settle(fixture);
    expect(query('request-messages')).toBeNull();
    expect(query('response-text')).toBeNull();
    // The recorded JSON text is shown as the structure it holds.
    expect(query('call-request')!.textContent).toContain('messages:');
    expect(query('call-response')!.textContent).toContain('cards:');
  });

  it('names the tool a message belongs to', async () => {
    agentService.getLLMCallStats.and.returnValue(
      of([makeModelCall({ request_data: { messages: [{ role: 'tool', tool_call_id: 'call_7', content: '42' }] } })]),
    );
    await render();
    await open();
    expect(all('request-message')[0].querySelector('.msg-role')!.textContent).toContain('tool · call_7');
  });

  it('offers no text view for a call with no messages or text', async () => {
    await render();
    await open(1);
    expect(query('show-text')).toBeNull();
    expect(query('call-request')!.querySelector('app-json-tree')).not.toBeNull();
    expect(query('call-response')!.querySelector('app-json-tree')).not.toBeNull();
    expect(query('call-detail')!.textContent).toContain('HTTP 429');
    expect(stat('Tokens in')).toEqual({ value: '—', hint: '' });
    expect(stat('Total').value).toBe('—');
    expect(query('run-link')).toBeNull();
    expect(query('conversation-link')).toBeNull();
    expect(query('agent-filter-link')).toBeNull();
    expect(query('call-detail')!.textContent).toContain('r2');
  });

  it('leaves out the filter links the list is already narrowed by', async () => {
    params.next({ run_id: 'r1-run-0000-1111', agent_id: 'a1-agent-000-3333' });
    await render();
    await open();
    expect(query('run-link')).not.toBeNull();
    expect(query('run-filter-link')).toBeNull();
    expect(query('agent-filter-link')).toBeNull();
  });

  it('opens the call named in the address', async () => {
    params.next({ run_id: 'r2', call: 'c2' });
    await render();
    expect(query('call-detail')!.textContent).toContain('fastembed/bge-small');
  });

  it('opens nothing when the named call is not on the page', async () => {
    params.next({ call: 'elsewhere' });
    await render();
    expect(query('call-detail')).toBeNull();
  });

  describe('Load more', () => {
    const page = (start: number, count: number): LLMCallStat[] =>
      Array.from({ length: count }, (_, i) => makeModelCall({ id: `c${start + i}` }));

    it('loads the next page and appends it', async () => {
      agentService.getLLMCallStats.and.returnValues(of(page(0, 20)), of(page(20, 5)));
      await render();
      expect(all('call-row').length).toBe(20);

      query<HTMLButtonElement>('load-more')!.click();
      await settle(fixture);
      expect(agentService.getLLMCallStats.calls.mostRecent().args.slice(2, 4)).toEqual([20, 20]);
      expect(all('call-row').length).toBe(25);
      expect(query('load-more')).toBeNull();
    });

    it('keeps the button busy while the next page loads', async () => {
      agentService.getLLMCallStats.and.returnValues(of(page(0, 20)), NEVER);
      await render();
      query<HTMLButtonElement>('load-more')!.click();
      await settle(fixture);
      const button = query<HTMLButtonElement>('load-more')!;
      expect(button.disabled).toBeTrue();
      expect(button.querySelector('ui-spinner')).not.toBeNull();
      expect(all('call-row').length).toBe(20);

      fixture.componentInstance.loadMore();
      expect(agentService.getLLMCallStats).toHaveBeenCalledTimes(2);
    });

    it('shows a failed page under the list with Retry', async () => {
      agentService.getLLMCallStats.and.returnValues(
        of(page(0, 20)),
        throwError(() => httpError('Timed out')),
        of(page(20, 3)),
      );
      await render();
      query<HTMLButtonElement>('load-more')!.click();
      await settle(fixture);
      expect(query('more-error')!.textContent).toContain('Timed out');
      expect(all('call-row').length).toBe(20);

      query<HTMLButtonElement>('more-retry')!.click();
      await settle(fixture);
      expect(query('more-error')).toBeNull();
      expect(all('call-row').length).toBe(23);
    });
  });

  describe('token usage', () => {
    it('adds up every matching call: figures, tokens per day and models', async () => {
      await render();
      const totals = query('usage-totals')!.textContent!;
      expect(totals).toContain('3,729');
      expect(totals).toContain('120 cached');
      expect(totals).toContain('31 reasoning');
      expect(totals).toContain('1 failed');
      expect(totals).toContain('2.1 s');
      // Embedding tokens come from the embedding models.
      expect(totals).toContain('Embedding tokens');
      expect(fixture.componentInstance.embeddingTokens()).toBe(7);
      const models = all('usage-model').map((r) => r.textContent!);
      expect(models[0]).toContain('scaleway/glm-5.3');
      expect(models[1]).toContain('fastembed/bge');
      // A week ending on the last day with calls, quiet days drawn as gaps.
      expect(fixture.componentInstance.tokenBars().map((b) => b.day)).toEqual([
        '2026-11-18',
        '2026-11-19',
        '2026-11-20',
        '2026-11-21',
        '2026-11-22',
        '2026-11-23',
        '2026-11-24',
      ]);
      expect(query('usage')!.querySelector('[data-testid="chart-unit"]')!.textContent).toBe('tokens');
    });

    it('says when no call matches', async () => {
      agentService.getLLMCallSummary.and.returnValue(
        of({ totals: { ...SUMMARY.totals, calls: 0, average_seconds: null }, by_model: [], by_day: [] })
      );
      await render();
      expect(query('usage-empty')).not.toBeNull();
      expect(query('usage-totals')).toBeNull();
    });

    it('shows a loading state, then the server refusal with Retry', async () => {
      agentService.getLLMCallSummary.and.returnValue(NEVER);
      await render();
      expect(query('usage')!.querySelector('ui-loading')).not.toBeNull();

      agentService.getLLMCallSummary.and.returnValue(throwError(() => httpError('Summary failed')));
      fixture.componentInstance.loadSummary();
      await settle(fixture);
      expect(query('usage-error')!.textContent).toContain('Summary failed');

      agentService.getLLMCallSummary.and.returnValue(of(SUMMARY));
      query<HTMLButtonElement>('usage-retry')!.click();
      await settle(fixture);
      expect(query('usage-totals')).not.toBeNull();
    });
  });

  describe('search and time window', () => {
    it('narrows the list and the summary by the query', async () => {
      params.next({ search: 'claims', from: '2026-11-23T00:00', to: '2026-11-24T23:59', run_id: 'r1' });
      await render();
      const expected = jasmine.objectContaining({
        runId: 'r1',
        search: 'claims',
        start: new Date('2026-11-23T00:00').toISOString(),
        end: new Date('2026-11-24T23:59').toISOString(),
      });
      expect(agentService.getLLMCallStats.calls.mostRecent().args[4]).toEqual(expected);
      expect(agentService.getLLMCallSummary.calls.mostRecent().args[2]).toEqual(expected);
      expect(query<HTMLInputElement>('search')!.value).toBe('claims');
      expect(query('clear-search')).not.toBeNull();
    });

    it('ignores a time it cannot read', async () => {
      params.next({ from: 'not a time' });
      await render();
      expect(agentService.getLLMCallStats.calls.mostRecent().args[4]).toEqual({});
    });

    it('puts the typed search into the query, and clears it', async () => {
      await render();
      const navigate = spyOn(TestBed.inject(Router), 'navigate').and.resolveTo(true);
      expect(query('clear-search')).toBeNull();
      fixture.componentInstance.form = { search: '  claims ', from: '2026-11-23T00:00', to: '' };
      query<HTMLButtonElement>('apply-search')!.click();
      expect(navigate.calls.mostRecent().args[1]!.queryParams).toEqual({
        search: 'claims',
        from: '2026-11-23T00:00',
        to: null,
        call: null,
      });

      fixture.componentInstance.clearText();
      expect(navigate.calls.mostRecent().args[1]!.queryParams).toEqual({ search: null, from: null, to: null, call: null });
    });
  });

  it('draws a long span between days without filling it', () => {
    const bars = tokenDays({
      ...SUMMARY,
      by_day: [
        { date: '2026-01-01', calls: 1, prompt_tokens: 1, completion_tokens: 0, embedding_tokens: 0 },
        { date: '2026-11-24', calls: 1, prompt_tokens: 2, completion_tokens: 0, embedding_tokens: 0 },
      ],
    });
    expect(bars.length).toBe(2);
    expect(tokenDays({ ...SUMMARY, by_day: [] })).toEqual([]);
  });
});
