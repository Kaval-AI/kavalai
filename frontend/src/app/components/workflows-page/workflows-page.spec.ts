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
import { provideRouter } from '@angular/router';
import { BehaviorSubject, NEVER, of, throwError } from 'rxjs';
import {
  RunItem,
  WorkflowsPage,
  organizeIntoLanes,
  runEnd,
  runStatus,
} from './workflows-page';
import { AgentService } from '../../services/agent-service';
import { UserService } from '../../services/user-service';
import { Agent } from '../../models/agent';
import { SessionDetails } from '../../models/session';
import { UserDetails } from '../../models/user-details';
import { makeRun, makeSession, makeTask, makeUser, settle } from '../../../testing';

const AGENT: Agent = { id: 'a1', name: 'pulse_answer' };

function details(overrides: Partial<SessionDetails> = {}): SessionDetails {
  return {
    session_id: 's1',
    messages: [],
    runs: [makeRun({ id: 'r1', created_at: '2026-11-24T10:00:00Z', duration_seconds: 4 })],
    tasks: [
      makeTask({ id: 't1', run_id: 'r1', name: 'fetch', inputs: { question: 'claims?' } }),
      makeTask({ id: 't2', run_id: 'r1', name: 'answer', errors: ['Model refused'], inputs: null }),
    ],
    ...overrides,
  };
}

function runItem(id: string, createdAt: string, seconds: number | null): RunItem {
  return {
    run: makeRun({ id, created_at: createdAt, duration_seconds: seconds }),
    sessionId: 's1',
    tasks: [],
    status: 'empty',
  };
}

describe('WorkflowsPage', () => {
  let fixture: ComponentFixture<WorkflowsPage>;
  let el: HTMLElement;
  let agentService: jasmine.SpyObj<AgentService>;
  let user: BehaviorSubject<UserDetails | null>;

  function query<T extends Element = HTMLElement>(testId: string): T | null {
    return el.querySelector<T>(`[data-testid="${testId}"]`);
  }

  function all(testId: string): HTMLElement[] {
    return Array.from(el.querySelectorAll<HTMLElement>(`[data-testid="${testId}"]`));
  }

  async function render(): Promise<void> {
    fixture = TestBed.createComponent(WorkflowsPage);
    el = fixture.nativeElement;
    fixture.detectChanges();
    await settle(fixture);
  }

  beforeEach(async () => {
    agentService = jasmine.createSpyObj('AgentService', [
      'getAgentsByProject',
      'getSessions',
      'getSessionDetails',
    ]);
    agentService.getAgentsByProject.and.returnValue(of([AGENT]));
    agentService.getSessions.and.returnValue(of({ sessions: [makeSession()] }));
    agentService.getSessionDetails.and.returnValue(of(details()));
    user = new BehaviorSubject<UserDetails | null>(makeUser());

    await TestBed.configureTestingModule({
      imports: [WorkflowsPage],
      providers: [
        provideRouter([]),
        { provide: AgentService, useValue: agentService },
        { provide: UserService, useValue: { userDetails: user.asObservable() } },
      ],
    }).compileComponents();
  });

  it('is titled Runs', async () => {
    await render();
    expect(el.querySelector('ui-page-header h1')!.textContent).toContain('Runs');
  });

  it('asks for a project when none is active', async () => {
    user.next(makeUser({ active_project_id: null }));
    await render();
    expect(query('runs-no-project')).not.toBeNull();
    expect(query('runs-refresh')).toBeNull();
    expect(agentService.getAgentsByProject).not.toHaveBeenCalled();
  });

  it('waits for a signed-in user', async () => {
    user.next(null);
    await render();
    expect(agentService.getAgentsByProject).not.toHaveBeenCalled();
  });

  it('shows a spinner while the runs load, and a busy Refresh', async () => {
    agentService.getAgentsByProject.and.returnValue(NEVER);
    await render();
    expect(el.querySelector('ui-loading')).not.toBeNull();
    const refresh = query<HTMLButtonElement>('runs-refresh')!;
    expect(refresh.disabled).toBeTrue();
    expect(refresh.querySelector('ui-spinner')).not.toBeNull();
  });

  it("shows the server's error with Retry, which loads again", async () => {
    agentService.getAgentsByProject.and.returnValue(
      throwError(() => new HttpErrorResponse({ status: 503, error: { detail: 'Project database unreachable' } }))
    );
    await render();
    expect(query('runs-error')!.textContent).toContain('Project database unreachable');

    agentService.getAgentsByProject.and.returnValue(of([AGENT]));
    query<HTMLButtonElement>('runs-retry')!.click();
    await settle(fixture);
    expect(query('runs-error')).toBeNull();
    expect(all('agent-runs').length).toBe(1);
  });

  it('says when the project has no agents', async () => {
    agentService.getAgentsByProject.and.returnValue(of([]));
    await render();
    expect(query('runs-empty')).not.toBeNull();
  });

  it('shows a card per agent, its runs as task squares', async () => {
    await render();
    expect(agentService.getSessions).toHaveBeenCalledWith('p1', 'a1', undefined, undefined, undefined, 10, 0);
    expect(agentService.getSessionDetails).toHaveBeenCalledWith('p1', 's1');
    const card = all('agent-runs')[0];
    expect(card.querySelector('.ui-card-title')!.textContent).toContain('pulse_answer');
    expect(card.querySelector('a[href="/conversations?agentId=a1"]')).not.toBeNull();

    const tasks = all('task');
    expect(tasks.map((t) => t.getAttribute('data-status'))).toEqual(['ok', 'error']);
    expect(tasks[1].getAttribute('aria-label')).toBe('answer: failed');
    expect(all('run')[0].classList).toContain('run-failed');
  });

  it('shows an agent without runs, and a run without tasks', async () => {
    agentService.getAgentsByProject.and.returnValue(of([AGENT, { id: 'a2', name: 'quiet' }]));
    agentService.getSessions.and.callFake((_p: string, agentId?: string) =>
      of({ sessions: agentId === 'a2' ? [] : [makeSession()] })
    );
    agentService.getSessionDetails.and.returnValue(of(details({ tasks: [] })));
    await render();
    expect(all('agent-runs').length).toBe(2);
    expect(all('run')[0].textContent).toContain('No tasks');
    expect(query('no-runs')!.textContent).toContain('No runs yet.');
  });

  it('leaves out what fails to load rather than failing the page', async () => {
    agentService.getAgentsByProject.and.returnValue(of([AGENT, { id: 'a2', name: 'broken' }]));
    agentService.getSessions.and.callFake((_p: string, agentId?: string) =>
      agentId === 'a2' ? throwError(() => new Error('down')) : of({ sessions: [makeSession()] })
    );
    agentService.getSessionDetails.and.returnValue(throwError(() => new Error('gone')));
    await render();
    expect(query('runs-error')).toBeNull();
    expect(all('no-runs').length).toBe(2);
  });

  it('opens a task in a dialog, with links to its conversation and run', async () => {
    await render();
    const dialog = el.querySelector<HTMLDialogElement>('ui-dialog dialog')!;
    all('task')[1].click();
    await settle(fixture);

    expect(dialog.open).toBeTrue();
    const detail = query('task-detail')!;
    expect(detail.textContent).toContain('answer');
    expect(detail.querySelector('ui-status-badge')!.textContent).toContain('Error');
    expect(query('task-errors')!.textContent).toContain('Model refused');
    expect(detail.textContent).toContain('No input recorded.');
    expect(query<HTMLAnchorElement>('open-conversation')!.getAttribute('href')).toBe('/conversations/s1#task-t2');
    expect(query<HTMLAnchorElement>('open-run')!.getAttribute('href')).toBe('/conversations/s1/runs/r1/tasks');

    query<HTMLButtonElement>('task-close')!.click();
    await settle(fixture);
    expect(dialog.open).toBeFalse();
    expect(query('task-detail')).toBeNull();
  });

  it("shows a successful task's input, type and tool", async () => {
    agentService.getSessionDetails.and.returnValue(
      of(details({ tasks: [makeTask({ id: 't1', run_id: 'r1', name: null, node_type: 'tool', tool_uri: 'python://lookup', inputs: { q: 1 } })] }))
    );
    await render();
    all('task')[0].click();
    await settle(fixture);
    const detail = query('task-detail')!;
    expect(detail.textContent).toContain('Unnamed task');
    expect(detail.textContent).toContain('python://lookup');
    expect(detail.querySelector('app-json-tree')).not.toBeNull();
    expect(query('task-errors')).toBeNull();
  });

  it('refreshes on request and when the project changes', async () => {
    await render();
    query<HTMLButtonElement>('runs-refresh')!.click();
    await settle(fixture);
    expect(agentService.getAgentsByProject).toHaveBeenCalledTimes(2);

    user.next(makeUser());
    await settle(fixture);
    expect(agentService.getAgentsByProject).toHaveBeenCalledTimes(2);

    user.next(makeUser({ active_project_id: 'p2' }));
    await settle(fixture);
    expect(agentService.getAgentsByProject).toHaveBeenCalledWith('p2');
  });

  describe('lanes', () => {
    it('puts runs that overlap into separate lanes', () => {
      const lanes = organizeIntoLanes([
        runItem('late', '2026-11-24T10:00:30Z', 5),
        runItem('first', '2026-11-24T10:00:00Z', 60),
        runItem('after', '2026-11-24T10:02:00Z', 5),
      ]);
      expect(lanes.map((l) => l.runs.map((r) => r.run.id))).toEqual([['first', 'after'], ['late']]);
    });

    it('assumes a minute for a run without a duration', () => {
      const start = new Date('2026-11-24T10:00:00Z').getTime();
      expect(runEnd(makeRun({ created_at: '2026-11-24T10:00:00Z', duration_seconds: null }))).toBe(start + 60_000);
      expect(runEnd(makeRun({ created_at: '2026-11-24T10:00:00Z', duration_seconds: 2 }))).toBe(start + 2_000);
    });

    it('words a run by its tasks', () => {
      expect(runStatus([])).toBe('empty');
      expect(runStatus([{ task: makeTask(), status: 'ok' }])).toBe('ok');
    });
  });
});
