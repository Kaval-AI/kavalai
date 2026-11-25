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
import { ActivatedRoute, Router, convertToParamMap, provideRouter } from '@angular/router';
import { BehaviorSubject, NEVER, of, throwError } from 'rxjs';
import { RunTasksPage } from './run-tasks-page';
import { AgentService } from '../../services/agent-service';
import { UserService } from '../../services/user-service';
import { SessionDetails } from '../../models/session';
import { UserDetails } from '../../models/user-details';
import { makeModelCall, makeRun, makeTask, makeUser, settle } from '../../../testing';

describe('RunTasksPage', () => {
  let fixture: ComponentFixture<RunTasksPage>;
  let el: HTMLElement;
  let agentService: jasmine.SpyObj<AgentService>;
  let user: BehaviorSubject<UserDetails | null>;
  let router: Router;

  const details = (overrides: Partial<SessionDetails> = {}): SessionDetails => ({
    session_id: 's1',
    messages: [],
    runs: [makeRun({ id: 'r1', duration_seconds: 4.2 }), makeRun({ id: 'r2' })],
    tasks: [
      makeTask({ id: 't1', name: 'fetch', run_id: 'r1', seq: 0 }),
      makeTask({ id: 't2', name: 'compose', run_id: 'r1', seq: 1, node_type: 'llm', errors: ['Timed out'] }),
      makeTask({ id: 't3', name: 'other run', run_id: 'r2' }),
    ],
    ...overrides,
  });

  const httpError = (detail: string) => new HttpErrorResponse({ status: 500, error: { detail } });

  async function render(): Promise<void> {
    fixture = TestBed.createComponent(RunTasksPage);
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
    const tile = Array.from(el.querySelectorAll('ui-stat')).find(
      (t) => t.querySelector('.ui-stat-label')!.textContent === label,
    )!;
    return {
      value: tile.querySelector('.ui-stat-value')!.textContent!.trim(),
      hint: tile.querySelector('.ui-stat-hint')?.textContent?.trim() ?? '',
    };
  }

  beforeEach(async () => {
    agentService = jasmine.createSpyObj<AgentService>('AgentService', ['getSessionDetails', 'getLLMCallStats']);
    agentService.getSessionDetails.and.returnValue(of(details()));
    agentService.getLLMCallStats.and.returnValue(
      of([
        makeModelCall({ id: 'c2', created_at: '2026-11-24T10:00:03Z', prompt_tokens: 1200, completion_tokens: 50, total_tokens: 1250 }),
        makeModelCall({ id: 'c1', created_at: '2026-11-24T10:00:01Z', response_code: 500 }),
      ]),
    );
    user = new BehaviorSubject<UserDetails | null>(makeUser({ active_project_id: 'p1' }));

    await TestBed.configureTestingModule({
      imports: [RunTasksPage],
      providers: [
        provideRouter([]),
        { provide: AgentService, useValue: agentService },
        { provide: UserService, useValue: { userDetails: user.asObservable() } },
        {
          provide: ActivatedRoute,
          useValue: { paramMap: of(convertToParamMap({ sessionId: 's1', runId: 'r1' })) },
        },
      ],
    }).compileComponents();
    router = TestBed.inject(Router);
  });

  it('is headed by the run, with a way back to its conversation', async () => {
    await render();
    expect(el.querySelector('h1')!.textContent).toContain('Run');
    expect(query('section-title')!.textContent).toBe('r1');
    expect(el.querySelector<HTMLAnchorElement>('.ui-back-link')!.getAttribute('href')).toBe('/conversations/s1');
    expect(query('run-started')!.textContent).toContain('Started');
    expect(query<HTMLAnchorElement>('run-model-calls-link')!.getAttribute('href')).toBe('/llm-call-stats?run_id=r1');
  });

  it('shows a spinner while the run loads', async () => {
    agentService.getSessionDetails.and.returnValue(NEVER);
    agentService.getLLMCallStats.and.returnValue(NEVER);
    await render();
    expect(el.querySelector('ui-loading')).not.toBeNull();
    expect(query('run-stats')).toBeNull();
  });

  it('shows the server error with Retry when the run fails to load', async () => {
    agentService.getSessionDetails.and.returnValue(throwError(() => httpError('Database unreachable')));
    await render();
    expect(query('run-error')!.textContent).toContain('Database unreachable');

    agentService.getSessionDetails.and.returnValue(of(details()));
    query<HTMLButtonElement>('run-retry')!.click();
    await settle(fixture);
    expect(query('run-error')).toBeNull();
    expect(all('task-row').length).toBe(2);
  });

  it('says so when the conversation has no such run', async () => {
    agentService.getSessionDetails.and.returnValue(of(details({ runs: [], tasks: [] })));
    await render();
    expect(query('run-missing')!.textContent).toContain('No such run');
    expect(query('run-missing')!.querySelector('a')!.getAttribute('href')).toBe('/conversations/s1');
  });

  it('shows a run that recorded no steps', async () => {
    agentService.getSessionDetails.and.returnValue(of(details({ tasks: [] })));
    await render();
    expect(stat('Steps').value).toBe('0');
    expect(el.textContent).toContain('No steps to place on the timeline.');
    expect(query('tasks-empty')).not.toBeNull();
  });

  it("shows the run's figures, timeline and steps", async () => {
    await render();
    expect(stat('Duration')).toEqual({ value: '4.2 s', hint: 'wall-clock time' });
    expect(stat('Steps')).toEqual({ value: '2', hint: '1 failed' });
    expect(stat('Model calls').value).toBe('2');
    expect(stat('Tokens')).toEqual({ value: '2,300', hint: '2,061 in / 239 out' });
    expect(all('timeline-row').length).toBe(2);
    expect(all('task-row').map((r) => r.querySelectorAll('td')[1].textContent!.trim())).toEqual(['fetch', 'compose']);
    expect(agentService.getSessionDetails).toHaveBeenCalledWith('p1', 's1');
  });

  it("lists the run's model calls oldest first and opens one on the Model calls page", async () => {
    const navigate = spyOn(router, 'navigate').and.resolveTo(true);
    await render();
    expect(agentService.getLLMCallStats).toHaveBeenCalledWith('p1', undefined, 100, 0, { runId: 'r1' });
    const rows = all('model-call-row');
    expect(rows.length).toBe(2);
    const first = rows[0].querySelectorAll('td');
    expect(first[1].textContent!.trim()).toBe('scaleway/glm-5.3');
    expect(first[3].textContent!.trim()).toBe('Error');
    expect(first[4].textContent!.trim()).toBe('2.5 s');
    const second = rows[1].querySelectorAll('td');
    expect(second[3].textContent!.trim()).toBe('OK');
    expect(second[5].textContent!.trim()).toBe('1,200');
    expect(query<HTMLAnchorElement>('open-model-calls')!.getAttribute('href')).toBe('/llm-call-stats?run_id=r1');

    rows[1].click();
    expect(navigate).toHaveBeenCalledWith(['/llm-call-stats'], { queryParams: { run_id: 'r1', call: 'c2' } });
    rows[0].dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter' }));
    expect(navigate).toHaveBeenCalledWith(['/llm-call-stats'], { queryParams: { run_id: 'r1', call: 'c1' } });
  });

  it('shows a call without a status code or token counts as unknown', async () => {
    agentService.getLLMCallStats.and.returnValue(
      of([makeModelCall({ response_code: null, prompt_tokens: null, completion_tokens: null, total_tokens: null })]),
    );
    await render();
    const cells = all('model-call-row')[0].querySelectorAll('td');
    expect(cells[3].textContent!.trim()).toBe('unknown');
    expect(cells[5].textContent!.trim()).toBe('—');
    expect(cells[6].textContent!.trim()).toBe('—');
    expect(stat('Tokens').value).toBe('0');
  });

  it('says when only the first hundred model calls are listed', async () => {
    agentService.getLLMCallStats.and.returnValue(of(Array.from({ length: 100 }, (_, i) => makeModelCall({ id: `c${i}` }))));
    await render();
    expect(stat('Model calls')).toEqual({ value: '100', hint: 'the first 100' });
  });

  it('says so when the run made no model calls', async () => {
    agentService.getLLMCallStats.and.returnValue(of([]));
    await render();
    expect(query('model-calls-empty')!.textContent).toContain('The run made no model calls.');
    expect(query('open-model-calls')).toBeNull();
  });

  it('shows a failed model call list with Retry, beside the steps', async () => {
    agentService.getLLMCallStats.and.returnValue(throwError(() => httpError('Stats table missing')));
    await render();
    expect(query('model-calls-error')!.textContent).toContain('Stats table missing');
    expect(all('task-row').length).toBe(2);

    agentService.getLLMCallStats.and.returnValue(of([makeModelCall()]));
    query<HTMLButtonElement>('model-calls-retry')!.click();
    await settle(fixture);
    expect(all('model-call-row').length).toBe(1);
  });

  it('shows a spinner while the model calls load', async () => {
    agentService.getLLMCallStats.and.returnValue(NEVER);
    await render();
    expect(query('run-model-calls')!.querySelector('ui-loading')).not.toBeNull();
  });

  it('asks for a project when none is selected', async () => {
    user.next(makeUser({ active_project_id: 'None' }));
    await render();
    expect(query('no-project')).not.toBeNull();
    expect(query('run-model-calls')).toBeNull();
    expect(agentService.getSessionDetails).not.toHaveBeenCalled();
  });

  it('loads again for another project', async () => {
    await render();
    user.next(makeUser({ active_project_id: 'p2' }));
    await settle(fixture);
    expect(agentService.getSessionDetails).toHaveBeenCalledWith('p2', 's1');
    user.next(makeUser({ active_project_id: 'p2', name: 'Renamed' }));
    expect(agentService.getSessionDetails).toHaveBeenCalledTimes(2);
  });
});
