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
import { ActivatedRoute, Params, provideRouter } from '@angular/router';
import { BehaviorSubject, NEVER, of, throwError } from 'rxjs';
import { ConversationsPage } from './conversations-page';
import { AgentService } from '../../services/agent-service';
import { UserService } from '../../services/user-service';
import { SessionSummary } from '../../models/session';
import { Agent } from '../../models/agent';
import { makeSession, settle, signedInAs } from '../../../testing';

describe('ConversationsPage', () => {
  let fixture: ComponentFixture<ConversationsPage>;
  let el: HTMLElement;
  let agentService: jasmine.SpyObj<AgentService>;
  let userService: jasmine.SpyObj<UserService>;
  let queryParams: BehaviorSubject<Params>;

  const agents: Agent[] = [
    { id: 'a1', name: 'pulse_answer' },
    { id: 'a2', name: 'pulse_report' },
  ];

  function page(sessions: SessionSummary[], total = sessions.length) {
    return of({ sessions, total_count: total });
  }

  function many(count: number, from = 0): SessionSummary[] {
    return Array.from({ length: count }, (_, i) => makeSession({ session_id: `s${from + i}` }));
  }

  async function render(): Promise<void> {
    fixture = TestBed.createComponent(ConversationsPage);
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

  /** The arguments of the last `getSessions` call, by name. */
  function lastCall() {
    const [projectId, agentId, search, start, end, limit, offset, externalId] =
      agentService.getSessions.calls.mostRecent().args;
    return { projectId, agentId, search, start, end, limit, offset, externalId };
  }

  beforeEach(async () => {
    agentService = jasmine.createSpyObj('AgentService', ['getAgentsByProject', 'getSessions']);
    agentService.getAgentsByProject.and.returnValue(of(agents));
    userService = signedInAs('owner');
    queryParams = new BehaviorSubject<Params>({});

    await TestBed.configureTestingModule({
      imports: [ConversationsPage],
      providers: [
        provideRouter([]),
        { provide: AgentService, useValue: agentService },
        { provide: UserService, useValue: userService },
        { provide: ActivatedRoute, useValue: { queryParams } },
      ],
    }).compileComponents();
  });

  it('opens with its page header', async () => {
    agentService.getSessions.and.returnValue(page([]));
    await render();
    expect(el.querySelector('ui-page-header h1')!.textContent).toContain('Conversations');
  });

  it('shows a loading state while the sessions are fetched', async () => {
    agentService.getSessions.and.returnValue(NEVER);
    await render();
    expect(query('loading')).not.toBeNull();
    expect(query('sessions-table')).toBeNull();
  });

  it('asks for the last week of the active project, twenty at a time', async () => {
    agentService.getSessions.and.returnValue(page([]));
    await render();
    const call = lastCall();
    expect(agentService.getAgentsByProject).toHaveBeenCalledWith('p1');
    expect(call.projectId).toBe('p1');
    expect(call.agentId).toBeUndefined();
    expect(call.search).toBeUndefined();
    expect(call.externalId).toBeUndefined();
    expect(call.limit).toBe(20);
    expect(call.offset).toBe(0);
    const days = (new Date(call.end!).getTime() - new Date(call.start!).getTime()) / 86400000;
    expect(days).toBeGreaterThan(7);
    expect(days).toBeLessThan(8.1);
  });

  it('shows the server error with a Retry that loads again', async () => {
    agentService.getSessions.and.returnValue(
      throwError(() => new HttpErrorResponse({ status: 500, error: { detail: 'Database unreachable' } }))
    );
    await render();
    expect(query('error')!.textContent).toContain('Database unreachable');
    expect(query('sessions-table')).toBeNull();

    agentService.getSessions.and.returnValue(page([makeSession()]));
    query<HTMLButtonElement>('retry')!.click();
    await settle(fixture);
    expect(query('error')).toBeNull();
    expect(all('session-row').length).toBe(1);
  });

  it('names the next step when no conversation matches, and clears the filters', async () => {
    queryParams.next({ agentId: 'a1', externalId: 'eval:suite:' });
    agentService.getSessions.and.returnValue(page([]));
    await render();
    expect(query('empty')!.textContent).toContain('No conversations match');
    expect(lastCall().agentId).toBe('a1');

    query<HTMLButtonElement>('clear-filters')!.click();
    await settle(fixture);
    const call = lastCall();
    expect(call.agentId).toBeUndefined();
    expect(call.externalId).toBeUndefined();
    expect(call.search).toBeUndefined();
  });

  it('lists the sessions with their counts, eval marker and errors', async () => {
    agentService.getSessions.and.returnValue(
      page(
        [
          makeSession({ session_id: 's1', errors_count: 0, first_message: 'Hello there' }),
          makeSession({ session_id: 's2', errors_count: 3, external_id: 'eval:suite:tag:case', first_message: null }),
        ],
        57
      )
    );
    await render();
    const rows = all('session-row');
    expect(rows.length).toBe(2);
    expect(rows[0].textContent).toContain('pulse_answer');
    expect(rows[0].textContent).toContain('Hello there');
    expect(rows[0].querySelector('[data-testid="errors-badge"]')).toBeNull();
    expect(rows[0].querySelector('ui-badge')).toBeNull();
    expect(rows[1].querySelector('[data-testid="errors-badge"]')!.textContent).toContain('3');
    expect(rows[1].textContent).toContain('eval');
    expect(rows[1].textContent).toContain('—');
    expect(query('count')!.textContent).toContain('2 of 57');
    expect(query('load-more')).toBeNull();
  });

  it('opens the summary of a selected row beside the table, and closes it', async () => {
    const session = makeSession({
      session_id: 'abcdef1234567890',
      external_id: 'customer-42',
      runs_count: 4,
      first_message: 'How many claims?',
      last_message: '{"cards": []}',
    });
    agentService.getSessions.and.returnValue(page([session, makeSession({ session_id: 's2' })]));
    await render();
    expect(query('session-detail')).toBeNull();

    all('session-row')[0].click();
    await settle(fixture);
    const detail = query('session-detail')!;
    expect(all('session-row')[0].classList).toContain('selected');
    expect(detail.textContent).toContain('pulse_answer');
    expect(detail.textContent).toContain('abcdef12');
    expect(query('detail-external')!.textContent).toContain('customer-42');
    expect(query('detail-first')!.textContent).toContain('How many claims?');
    expect(query('detail-last')!.textContent).toContain('{"cards": []}');
    expect(query<HTMLAnchorElement>('open-conversation')!.getAttribute('href')).toBe(
      '/conversations/abcdef1234567890'
    );

    all('session-row')[0].click();
    await settle(fixture);
    expect(query('session-detail')).toBeNull();

    all('session-row')[1].dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter' }));
    await settle(fixture);
    expect(query('session-detail')).not.toBeNull();
    expect(query('detail-external')).toBeNull();
    expect(query('detail-first')!.textContent).toContain('How many');

    query<HTMLButtonElement>('close-detail')!.click();
    await settle(fixture);
    expect(query('session-detail')).toBeNull();
  });

  it('shows a dash for a session with no messages', async () => {
    agentService.getSessions.and.returnValue(page([makeSession({ first_message: null, last_message: null })]));
    await render();
    all('session-row')[0].click();
    await settle(fixture);
    expect(query('detail-first')!.textContent!.trim()).toBe('—');
    expect(query('detail-last')!.textContent!.trim()).toBe('—');
  });

  it('narrows the list by agent, message text and external id', async () => {
    agentService.getSessions.and.returnValue(page([]));
    await render();

    const select = query<HTMLSelectElement>('filter-agent')!;
    expect(Array.from(select.options).map((o) => o.textContent!.trim())).toEqual([
      'All agents',
      'pulse_answer',
      'pulse_report',
    ]);
    select.value = select.options[2].value;
    select.dispatchEvent(new Event('change'));
    await settle(fixture);
    expect(lastCall().agentId).toBe('a2');
    expect(query('agent-chip')!.textContent).toContain('pulse_report');

    const search = query<HTMLInputElement>('filter-search')!;
    search.value = '  claims  ';
    search.dispatchEvent(new Event('input'));
    search.dispatchEvent(new KeyboardEvent('keyup', { key: 'Enter' }));
    await settle(fixture);
    expect(lastCall().search).toBe('claims');

    const external = query<HTMLInputElement>('filter-external')!;
    external.value = 'eval:suite:';
    external.dispatchEvent(new Event('input'));
    query<HTMLButtonElement>('apply')!.click();
    await settle(fixture);
    expect(lastCall().externalId).toBe('eval:suite:');
  });

  it('applies a changed date range', async () => {
    agentService.getSessions.and.returnValue(page([]));
    await render();
    const from = el.querySelector<HTMLInputElement>('#filter-from')!;
    from.value = '2026-01-01T00:00';
    from.dispatchEvent(new Event('input'));
    from.dispatchEvent(new Event('change'));
    await settle(fixture);
    expect(lastCall().start).toBe(new Date('2026-01-01T00:00').toISOString());

    const to = el.querySelector<HTMLInputElement>('#filter-to')!;
    to.value = '';
    to.dispatchEvent(new Event('input'));
    to.dispatchEvent(new Event('change'));
    await settle(fixture);
    expect(lastCall().end).toBeUndefined();
  });

  it('loads more sessions after a full page, and retries a failed page', async () => {
    agentService.getSessions.and.returnValue(page(many(20), 45));
    await render();
    expect(all('session-row').length).toBe(20);

    agentService.getSessions.and.returnValue(
      throwError(() => new HttpErrorResponse({ status: 502, error: { detail: 'Bad gateway' } }))
    );
    query<HTMLButtonElement>('load-more')!.click();
    await settle(fixture);
    expect(lastCall().offset).toBe(20);
    expect(query('error')!.textContent).toContain('Bad gateway');
    expect(all('session-row').length).toBe(20);

    agentService.getSessions.and.returnValue(page(many(5, 20), 45));
    query<HTMLButtonElement>('retry')!.click();
    await settle(fixture);
    expect(lastCall().offset).toBe(20);
    expect(all('session-row').length).toBe(25);
    expect(query('count')!.textContent).toContain('25 of 45');
    expect(query('load-more')).toBeNull();
  });

  it('holds a busy Load more button while the next page loads', async () => {
    agentService.getSessions.and.returnValue(page(many(20), 45));
    await render();
    agentService.getSessions.and.returnValue(NEVER);
    query<HTMLButtonElement>('load-more')!.click();
    await settle(fixture);
    const button = query<HTMLButtonElement>('load-more')!;
    expect(button.disabled).toBeTrue();
    expect(button.querySelector('ui-spinner')).not.toBeNull();

    agentService.getSessions.calls.reset();
    fixture.componentInstance.loadMore();
    expect(agentService.getSessions).not.toHaveBeenCalled();
  });

  it('warns when the agents cannot be loaded, with a Retry', async () => {
    agentService.getAgentsByProject.and.returnValue(throwError(() => new Error('offline')));
    agentService.getSessions.and.returnValue(page([]));
    await render();
    expect(query('agents-error')!.textContent).toContain('offline');

    agentService.getAgentsByProject.and.returnValue(of(agents));
    query('agents-error')!.querySelector<HTMLButtonElement>('button')!.click();
    await settle(fixture);
    expect(query('agents-error')).toBeNull();
  });

  it('asks for a project when none is active, and loads nothing', async () => {
    userService = signedInAs('owner', { active_project_id: null as never });
    TestBed.overrideProvider(UserService, { useValue: userService });
    agentService.getSessions.and.returnValue(page([]));
    await render();
    expect(query('no-project')).not.toBeNull();
    expect(query('filters')).toBeNull();
    expect(agentService.getSessions).not.toHaveBeenCalled();
    fixture.componentInstance.load();
    fixture.componentInstance.loadAgents();
    expect(agentService.getSessions).not.toHaveBeenCalled();
    expect(agentService.getAgentsByProject).not.toHaveBeenCalled();
  });
});
