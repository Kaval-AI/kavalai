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
import { AgentsPage, SummaryStats } from './agents-page';
import { AgentService } from '../../services/agent-service';
import { UserService } from '../../services/user-service';
import { Agent } from '../../models/agent';
import { UserDetails } from '../../models/user-details';
import { DailyStats } from '../../core/daily-stats';
import { makeUser, settle } from '../../../testing';

function makeAgent(overrides: Partial<Agent> = {}): Agent {
  return {
    id: 'a1',
    name: 'pulse_answer',
    description: 'Answers questions about the claims data.',
    workflow: { nodes: [{ name: 'start', type: 'start' }] },
    input_schema: { type: 'object', properties: { question: { type: 'string' } } },
    output_schema: { type: 'object' },
    created_at: '2026-11-20T10:00:00Z',
    updated_at: '2026-11-24T10:00:00Z',
    ...overrides,
  };
}

const SUMMARY: SummaryStats = {
  total_sessions: 12,
  total_runs: 30,
  total_messages: 60,
  total_tasks: 1234,
  total_prompt_tokens: 45000,
  total_completion_tokens: 3000,
  total_embedding_tokens: 0,
};

function dailyStats(sessions: number): DailyStats {
  const day = (count: number, extra = {}) => [{ date: '2026-11-24', count, ...extra }];
  return {
    sessions: day(sessions),
    messages: day(sessions * 2),
    tasks: day(sessions * 3),
    runs: { pulse_answer: day(sessions, { duration_seconds: 8 }) },
    llm: { 'scaleway/glm': day(sessions, { duration_seconds: 4, prompt_tokens: 900, completion_tokens: 100 }) },
    embedding: {},
  };
}

function httpError(detail: string): HttpErrorResponse {
  return new HttpErrorResponse({ status: 500, error: { detail } });
}

describe('AgentsPage', () => {
  let fixture: ComponentFixture<AgentsPage>;
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
    fixture = TestBed.createComponent(AgentsPage);
    el = fixture.nativeElement;
    fixture.detectChanges();
    await settle(fixture);
  }

  beforeEach(async () => {
    agentService = jasmine.createSpyObj('AgentService', [
      'getAgentsByProject',
      'getSummaryStats',
      'getDailyStats',
      'renderWorkflowSvg',
    ]);
    agentService.getAgentsByProject.and.returnValue(of([makeAgent()]));
    agentService.getSummaryStats.and.returnValue(of(SUMMARY));
    agentService.getDailyStats.and.returnValue(of(dailyStats(2)));
    agentService.renderWorkflowSvg.and.returnValue(of('<svg data-graph="1"></svg>'));
    user = new BehaviorSubject<UserDetails | null>(makeUser());

    await TestBed.configureTestingModule({
      imports: [AgentsPage],
      providers: [
        provideRouter([]),
        { provide: AgentService, useValue: agentService },
        { provide: UserService, useValue: { userDetails: user.asObservable() } },
      ],
    }).compileComponents();
  });

  it('opens with its page header', async () => {
    await render();
    expect(el.querySelector('ui-page-header h1')!.textContent).toContain('Agents');
  });

  it('asks for a project when none is active', async () => {
    user.next(makeUser({ active_project_id: 'None' }));
    await render();
    expect(query('agents-no-project')).not.toBeNull();
    expect(agentService.getAgentsByProject).not.toHaveBeenCalled();
  });

  it('waits for a signed-in user', async () => {
    user.next(null);
    await render();
    expect(agentService.getAgentsByProject).not.toHaveBeenCalled();
  });

  it('shows a spinner while the agents load', async () => {
    agentService.getAgentsByProject.and.returnValue(NEVER);
    await render();
    expect(el.querySelector('ui-loading')).not.toBeNull();
  });

  it("shows the server's error with Retry, which loads again", async () => {
    agentService.getAgentsByProject.and.returnValue(throwError(() => httpError('Database unreachable')));
    await render();
    expect(query('agents-error')!.textContent).toContain('Database unreachable');

    agentService.getAgentsByProject.and.returnValue(of([makeAgent()]));
    query<HTMLButtonElement>('agents-retry')!.click();
    await settle(fixture);
    expect(query('agents-error')).toBeNull();
    expect(all('agent-row').length).toBe(1);
  });

  it('says when the project has no agents', async () => {
    agentService.getAgentsByProject.and.returnValue(of([]));
    await render();
    expect(query('agents-empty')).not.toBeNull();
    expect(agentService.getSummaryStats).not.toHaveBeenCalled();
  });

  it('lists the agents and opens the first one', async () => {
    agentService.getAgentsByProject.and.returnValue(
      of([makeAgent(), makeAgent({ id: 'a2', name: 'pulse_index', description: undefined })])
    );
    await render();
    const rows = all('agent-row');
    expect(rows.map((r) => r.querySelector('.agent-name')!.textContent!.trim())).toEqual([
      'pulse_answer',
      'pulse_index',
    ]);
    expect(rows[0].classList).toContain('selected');
    expect(agentService.getAgentsByProject).toHaveBeenCalledWith('p1');
    expect(agentService.getSummaryStats).toHaveBeenCalledWith('p1', 'a1');
    expect(agentService.getDailyStats).toHaveBeenCalledWith('p1', 7, 'a1');

    const detail = query('agent-detail')!;
    expect(detail.querySelector('.ui-card-title')!.textContent).toContain('pulse_answer');
    expect(query('agent-description')!.textContent).toContain('claims data');
    expect(detail.querySelector('[data-graph]')).not.toBeNull();
  });

  it("shows the agent's totals and charts", async () => {
    await render();
    const tiles = Array.from(query('agent-totals')!.querySelectorAll('ui-stat'));
    expect(tiles.map((t) => t.querySelector('.ui-stat-label')!.textContent!.trim())).toEqual([
      'Sessions',
      'Runs',
      'Messages',
      'Tasks',
      'Input tokens',
      'Output tokens',
    ]);
    expect(tiles[3].querySelector('.ui-stat-value')!.textContent!.trim()).toBe('1,234');
    expect(tiles[3].querySelector('.ui-stat-hint')!.textContent!.trim()).toBe('Last 30 days');
    expect(query('activity-chart')!.querySelector('svg.chart')).not.toBeNull();
    expect(query('token-chart')!.querySelector('svg.chart')).not.toBeNull();
    expect(query('call-duration-chart')!.querySelector('canvas')).not.toBeNull();
    expect(query('run-duration-chart')!.querySelector('canvas')).not.toBeNull();
  });

  it('says so instead of drawing empty duration charts', async () => {
    agentService.getDailyStats.and.returnValue(of(dailyStats(0)));
    await render();
    expect(query('call-duration-chart')!.querySelector('canvas')).toBeNull();
    expect(query('run-duration-chart')!.textContent).toContain('No sessions in the last 7 days.');
    expect(query('activity-chart')!.textContent).toContain('No activity in the last 7 days.');
  });

  it('links to the conversations of the selected agent', async () => {
    await render();
    const link = query<HTMLAnchorElement>('agent-conversations')!;
    expect(link.getAttribute('href')).toBe('/conversations?agentId=a1');
  });

  it('opens another agent on click and on Enter', async () => {
    agentService.getAgentsByProject.and.returnValue(of([makeAgent(), makeAgent({ id: 'a2', name: 'pulse_index' })]));
    await render();
    all('agent-row')[1].click();
    await settle(fixture);
    expect(agentService.getSummaryStats).toHaveBeenCalledWith('p1', 'a2');
    expect(all('agent-row')[1].classList).toContain('selected');

    all('agent-row')[0].dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter' }));
    await settle(fixture);
    expect(all('agent-row')[0].classList).toContain('selected');
  });

  it('shows a spinner while the figures load', async () => {
    agentService.getSummaryStats.and.returnValue(NEVER);
    await render();
    expect(query('agent-detail')!.querySelector('ui-loading')).not.toBeNull();
    expect(query('agent-totals')).toBeNull();
  });

  it('shows a failed load of the figures with Retry', async () => {
    agentService.getDailyStats.and.returnValue(throwError(() => new Error('offline')));
    await render();
    expect(query('stats-error')!.textContent).toContain('offline');

    agentService.getDailyStats.and.returnValue(of(dailyStats(1)));
    query<HTMLButtonElement>('stats-retry')!.click();
    await settle(fixture);
    expect(query('stats-error')).toBeNull();
    expect(query('agent-totals')).not.toBeNull();
  });

  it('opens the workflow and the schemas as JSON in a dialog', async () => {
    await render();
    const dialog = el.querySelector<HTMLDialogElement>('ui-dialog dialog')!;
    expect(dialog.open).toBeFalse();

    query<HTMLButtonElement>('open-workflow')!.click();
    await settle(fixture);
    expect(dialog.open).toBeTrue();
    expect(dialog.querySelector('.ui-dialog-title')!.textContent).toContain('Workflow');
    expect(query('document-body')!.querySelector('app-json-tree')).not.toBeNull();

    query<HTMLButtonElement>('document-close')!.click();
    await settle(fixture);
    expect(dialog.open).toBeFalse();

    query<HTMLButtonElement>('open-input')!.click();
    await settle(fixture);
    expect(dialog.querySelector('.ui-dialog-title')!.textContent).toContain('Input schema');
    expect(query('document-body')!.textContent).toContain('properties');
  });

  it('says when a schema is missing', async () => {
    agentService.getAgentsByProject.and.returnValue(of([makeAgent({ output_schema: null })]));
    await render();
    query<HTMLButtonElement>('open-output')!.click();
    await settle(fixture);
    expect(query('document-body')!.textContent).toContain('This agent has no output schema recorded.');
  });

  it('closes the dialog on Escape', async () => {
    await render();
    query<HTMLButtonElement>('open-workflow')!.click();
    await settle(fixture);
    const dialog = el.querySelector<HTMLDialogElement>('ui-dialog dialog')!;
    dialog.close();
    // The native close event is queued as a task of its own.
    await new Promise((resolve) => setTimeout(resolve, 20));
    await settle(fixture);
    expect(query('document-body')).toBeNull();
  });

  it('reloads when the active project changes, and only then', async () => {
    await render();
    user.next(makeUser());
    await settle(fixture);
    expect(agentService.getAgentsByProject).toHaveBeenCalledTimes(1);

    user.next(makeUser({ active_project_id: 'p2' }));
    await settle(fixture);
    expect(agentService.getAgentsByProject).toHaveBeenCalledWith('p2');
  });
});
