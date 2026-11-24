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
import { ActivatedRoute, convertToParamMap, provideRouter } from '@angular/router';
import { NEVER, of, throwError } from 'rxjs';
import { SessionDetailPage } from './session-detail-page';
import { AgentService } from '../../services/agent-service';
import { UserService } from '../../services/user-service';
import { SessionDetails } from '../../models/session';
import { makeMessage, makeRun, makeTask, settle, signedInAs } from '../../../testing';

describe('SessionDetailPage', () => {
  let fixture: ComponentFixture<SessionDetailPage>;
  let el: HTMLElement;
  let agentService: jasmine.SpyObj<AgentService>;

  /** Two runs, listed out of order, the second with an error and no context. */
  const details: SessionDetails = {
    session_id: 's1',
    runs: [
      makeRun({
        id: 'r2-0000000000',
        created_at: '2026-11-24T10:05:00Z',
        duration_seconds: 75,
        tasks_count: 2,
        context: null,
      }),
      makeRun({
        id: 'r1-0000000000',
        created_at: '2026-11-24T10:00:00Z',
        duration_seconds: 0.25,
        context: { run_templates: {} },
      }),
    ],
    messages: [
      makeMessage({ id: 'm2', run_id: 'r1-0000000000', role: 'assistant', content: '{"cards": [1]}', created_at: '2026-11-24T10:00:04Z' }),
      makeMessage({ id: 'm1', run_id: 'r1-0000000000', role: 'user', content: 'How many claims?', created_at: '2026-11-24T10:00:01Z' }),
      makeMessage({ id: 'm3', run_id: 'r1-0000000000', role: 'assistant', content: '   ', created_at: '2026-11-24T10:00:05Z' }),
      makeMessage({ id: 'm4', run_id: null, role: 'system', content: 'Imported', created_at: '2026-11-24T09:00:00Z' }),
    ],
    tasks: [
      makeTask({ id: 't2', run_id: 'r1-0000000000', name: 'answer', seq: 2 }),
      makeTask({ id: 't1', run_id: 'r1-0000000000', name: 'fetch', seq: 1 }),
      makeTask({ id: 't3', run_id: 'r1-0000000000', name: 'sql_tool', seq: 3, parent_task_name: 'fetch' }),
      makeTask({ id: 't4', run_id: 'r2-0000000000', name: null, seq: null, errors: ['boom', 'bang'] }),
      makeTask({ id: 't5', run_id: 'r2-0000000000', name: 'fetch', seq: null, errors: null }),
    ],
  };

  async function render(project: string | null = 'p1'): Promise<void> {
    TestBed.overrideProvider(UserService, {
      useValue: signedInAs('owner', { active_project_id: project as never }),
    });
    fixture = TestBed.createComponent(SessionDetailPage);
    el = fixture.nativeElement;
    fixture.detectChanges();
    await settle(fixture);
  }

  function query<T extends Element = HTMLElement>(testId: string, root: Element = el): T | null {
    return root.querySelector<T>(`[data-testid="${testId}"]`);
  }

  function all(testId: string, root: Element = el): HTMLElement[] {
    return Array.from(root.querySelectorAll<HTMLElement>(`[data-testid="${testId}"]`));
  }

  beforeEach(async () => {
    agentService = jasmine.createSpyObj('AgentService', ['getSessionDetails']);
    await TestBed.configureTestingModule({
      imports: [SessionDetailPage],
      providers: [
        provideRouter([]),
        { provide: AgentService, useValue: agentService },
        { provide: UserService, useValue: signedInAs('owner') },
        { provide: ActivatedRoute, useValue: { paramMap: of(convertToParamMap({ sessionId: 's1-abcdefghij' })) } },
      ],
    }).compileComponents();
  });

  it('opens with a header leading back to the conversations', async () => {
    agentService.getSessionDetails.and.returnValue(of(details));
    await render();
    expect(el.querySelector('ui-page-header h1')!.textContent).toContain('Conversation');
    expect(el.querySelector('ui-page-header h1')!.textContent).toContain('s1-abcde');
    expect(el.querySelector<HTMLAnchorElement>('.ui-back-link')!.getAttribute('href')).toBe('/conversations');
    expect(query<HTMLAnchorElement>('session-model-calls')!.getAttribute('href')).toBe(
      '/llm-call-stats?session_id=s1-abcdefghij'
    );
    expect(agentService.getSessionDetails).toHaveBeenCalledOnceWith('p1', 's1-abcdefghij');
  });

  it('shows a loading state while the conversation is fetched', async () => {
    agentService.getSessionDetails.and.returnValue(NEVER);
    await render();
    expect(query('loading')).not.toBeNull();
    expect(all('run').length).toBe(0);
  });

  it('shows the server error with a Retry that loads again', async () => {
    agentService.getSessionDetails.and.returnValue(
      throwError(() => new HttpErrorResponse({ status: 404, error: { detail: 'Session not found' } }))
    );
    await render();
    expect(query('error')!.textContent).toContain('Session not found');

    agentService.getSessionDetails.and.returnValue(of(details));
    query<HTMLButtonElement>('retry')!.click();
    await settle(fixture);
    expect(query('error')).toBeNull();
    expect(all('run').length).toBe(2);
  });

  it('says so when the session recorded nothing', async () => {
    agentService.getSessionDetails.and.returnValue(of({ session_id: 's1', runs: [], messages: [], tasks: [] }));
    await render();
    expect(query('empty')!.textContent).toContain('No runs or messages');
    expect(all('run').length).toBe(0);
  });

  it('asks for a project when none is active, and loads nothing', async () => {
    await render(null);
    expect(query('no-project')).not.toBeNull();
    expect(agentService.getSessionDetails).not.toHaveBeenCalled();
  });

  it('groups the transcript by run, in the order the runs started', async () => {
    agentService.getSessionDetails.and.returnValue(of(details));
    await render();
    const runs = all('run');
    expect(runs.length).toBe(2);
    expect(runs[0].querySelector('.ui-card-title')!.textContent).toContain('Run 1');
    expect(runs[0].textContent).toContain('r1-00000');
    expect(query('run-duration', runs[0])!.textContent).toContain('250 ms');
    expect(query('run-steps', runs[0])!.textContent).toContain('fetch → answer');
    expect(query('run-errors', runs[0])).toBeNull();

    const messages = all('message', runs[0]);
    expect(messages.map((m) => m.dataset['role'])).toEqual(['user', 'assistant']);
    expect(messages[0].classList).toContain('from-user');
    expect(messages[0].textContent).toContain('User');
    expect(messages[0].textContent).toContain('How many claims?');
    expect(messages[1].textContent).toContain('Agent');
    expect(messages[1].querySelector('pre')!.textContent).toBe('{\n  "cards": [\n    1\n  ]\n}');

    expect(runs[1].querySelector('.ui-card-title')!.textContent).toContain('Run 2');
    expect(query('run-duration', runs[1])!.textContent).toContain('1 min 15 s');
    expect(query('run-errors', runs[1])!.textContent).toContain('2 errors');
    expect(query('run-steps', runs[1])!.textContent).toContain('t4 → fetch');
    expect(query('transcript', runs[1])).toBeNull();
    expect(runs[1].textContent).toContain('This run recorded no messages.');

    const outside = query('unassigned')!;
    expect(outside.textContent).toContain('Imported');
    expect(outside.textContent).toContain('system');
  });

  it('links each run to its tasks and model calls', async () => {
    agentService.getSessionDetails.and.returnValue(of(details));
    await render();
    const first = all('run')[0];
    expect(query<HTMLAnchorElement>('open-tasks', first)!.getAttribute('href')).toBe(
      '/conversations/s1/runs/r1-0000000000/tasks'
    );
    expect(query<HTMLAnchorElement>('run-model-calls', first)!.getAttribute('href')).toBe(
      '/llm-call-stats?session_id=s1&run_id=r1-0000000000'
    );
  });

  it('shows a run payload as a JSON tree in a dialog, and closes it', async () => {
    agentService.getSessionDetails.and.returnValue(of(details));
    await render();
    const dialog = el.querySelector<HTMLDialogElement>('dialog')!;
    expect(dialog.open).toBeFalse();

    query<HTMLButtonElement>('show-input', all('run')[0])!.click();
    await settle(fixture);
    expect(dialog.open).toBeTrue();
    expect(dialog.textContent).toContain('Run 1 input');
    expect(dialog.querySelector('app-json-tree')).not.toBeNull();
    expect(dialog.textContent).toContain('question');

    query<HTMLButtonElement>('close-json')!.click();
    await settle(fixture);
    expect(dialog.open).toBeFalse();
    expect(dialog.querySelector('app-json-tree')).toBeNull();

    query<HTMLButtonElement>('show-output', all('run')[0])!.click();
    await settle(fixture);
    expect(dialog.textContent).toContain('Run 1 output');
    dialog.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await settle(fixture);
    expect(dialog.open).toBeFalse();

    query<HTMLButtonElement>('show-context', all('run')[0])!.click();
    await settle(fixture);
    expect(dialog.textContent).toContain('Run 1 context');
  });

  it('disables the payload a run did not record', async () => {
    agentService.getSessionDetails.and.returnValue(of(details));
    await render();
    const second = all('run')[1];
    expect(query<HTMLButtonElement>('show-context', second)!.disabled).toBeTrue();
    expect(query<HTMLButtonElement>('show-input', second)!.disabled).toBeFalse();
  });

  it('shows text that only looks like JSON as text', async () => {
    agentService.getSessionDetails.and.returnValue(
      of({
        ...details,
        messages: [makeMessage({ run_id: 'r1-0000000000', role: 'assistant', content: '{not json' })],
      })
    );
    await render();
    const message = all('message')[0];
    expect(message.querySelector('pre')).toBeNull();
    expect(message.textContent).toContain('{not json');
  });
});
