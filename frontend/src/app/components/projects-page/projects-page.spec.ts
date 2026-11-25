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
import { HttpErrorResponse } from '@angular/common/http';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';
import { BehaviorSubject, NEVER, Observable, of, throwError } from 'rxjs';
import { DailyStats } from '../../core/daily-stats';
import { Project } from '../../models/project';
import { UserDetails } from '../../models/user-details';
import { AgentService } from '../../services/agent-service';
import { ProjectService } from '../../services/project-service';
import { UserService } from '../../services/user-service';
import { Role, makeProject, makeUser, settle, signedInAs } from '../../../testing';
import { ProjectsPage, activeProjectOf, connectionStrings, databaseLabel } from './projects-page';

const SUMMARY = {
  total_sessions: 12,
  total_messages: 30,
  total_tasks: 41,
  total_runs: 15,
  total_prompt_tokens: 3729,
  total_completion_tokens: 394,
  total_embedding_tokens: 0,
};

function daily(runs: DailyStats['runs'] = { pulse_answer: [{ date: '2026-11-24', count: 2, duration_seconds: 8 }] }): DailyStats {
  return {
    sessions: [
      { date: '2026-11-23', count: 0 },
      { date: '2026-11-24', count: 1 },
    ],
    messages: [
      { date: '2026-11-23', count: 0 },
      { date: '2026-11-24', count: 2 },
    ],
    tasks: [
      { date: '2026-11-23', count: 0 },
      { date: '2026-11-24', count: 5 },
    ],
    runs,
    llm: { 'scaleway/glm-5.3': [{ date: '2026-11-24', count: 1, prompt_tokens: 10, completion_tokens: 5, duration_seconds: 2 }] },
    embedding: {},
  };
}

function httpError(status: number, detail?: string): HttpErrorResponse {
  return new HttpErrorResponse({ status, error: detail ? { detail } : null });
}

describe('ProjectsPage', () => {
  let fixture: ComponentFixture<ProjectsPage>;
  let el: HTMLElement;
  let projects: jasmine.SpyObj<ProjectService>;
  let agents: jasmine.SpyObj<AgentService>;
  let users: jasmine.SpyObj<UserService>;
  let details: BehaviorSubject<UserDetails | null>;

  function query<T extends Element = HTMLElement>(testId: string): T | null {
    return el.querySelector<T>(`[data-testid="${testId}"]`);
  }

  function text(testId: string): string {
    return query(testId)?.textContent?.replace(/\s+/g, ' ').trim() ?? '';
  }

  /** Sign in as `role`, with `list` as the user's projects, and render the page. */
  async function render(
    role: Role = 'owner',
    list: Project[] | Observable<Project[]> = [makeProject({ role: role === 'viewer' ? 'viewer' : 'owner' })],
    user: Partial<UserDetails> = {}
  ): Promise<void> {
    users = signedInAs(role, user);
    // A subject of the spec's own, so a test can switch the active project as the sidebar does.
    details = new BehaviorSubject<UserDetails | null>(makeUser({ is_admin: role === 'admin', ...user }));
    (Object.getOwnPropertyDescriptor(users, 'userDetails')!.get as jasmine.Spy).and.returnValue(details.asObservable());
    users.getIsAdmin.and.callFake(() => !!details.value?.is_admin);
    projects.getAll.and.returnValue(Array.isArray(list) ? of(list) : list);
    TestBed.overrideProvider(UserService, { useValue: users });
    fixture = TestBed.createComponent(ProjectsPage);
    el = fixture.nativeElement;
    fixture.detectChanges();
    await settle(fixture);
  }

  beforeEach(async () => {
    projects = jasmine.createSpyObj<ProjectService>('ProjectService', ['getAll', 'revealPassword']);
    agents = jasmine.createSpyObj<AgentService>('AgentService', ['getSummaryStats', 'getDailyStats']);
    agents.getSummaryStats.and.returnValue(of(SUMMARY));
    agents.getDailyStats.and.returnValue(of(daily()));
    await TestBed.configureTestingModule({
      imports: [ProjectsPage],
      providers: [
        provideRouter([]),
        { provide: ProjectService, useValue: projects },
        { provide: AgentService, useValue: agents },
        { provide: UserService, useValue: signedInAs() },
      ],
    }).compileComponents();
  });


  describe('loading the project', () => {
    it('shows a spinner while the projects load', async () => {
      await render('owner', NEVER);
      expect(query('loading')).not.toBeNull();
    });

    it('shows the server’s reason and loads again on Retry', async () => {
      await render('owner', throwError(() => httpError(503, 'Backoffice database is not connected.')));
      expect(text('load-error')).toContain('Backoffice database is not connected.');

      projects.getAll.and.returnValue(of([makeProject()]));
      query<HTMLButtonElement>('retry')!.click();
      await settle(fixture);
      expect(query('load-error')).toBeNull();
      expect(el.querySelector('h1')!.textContent).toContain('Pulse local');
    });

    it('offers administrators a new project when there is none', async () => {
      await render('admin', []);
      expect(text('no-project')).toContain('No project yet');
      expect(query<HTMLAnchorElement>('new-project')!.getAttribute('href')).toBe('/projects/new');
    });

    it('tells everyone else to ask an administrator', async () => {
      await render('viewer', []);
      expect(text('no-project')).toContain('Ask an administrator');
      expect(query('new-project')).toBeNull();
    });

    it('asks to choose a project when none of the user’s projects is active', async () => {
      await render('owner', [makeProject()], { active_project_id: 'None' });
      expect(text('no-project')).toContain('No active project');
      expect(text('no-project')).toContain('Choose a project in the sidebar');
      expect(agents.getSummaryStats).not.toHaveBeenCalled();
    });

    it('follows the project the sidebar switches to', async () => {
      const other = makeProject({ id: 'p2', name: 'Kaval.AI examples', db_schema: 'agents' });
      await render('owner', [makeProject(), other]);
      expect(el.querySelector('h1')!.textContent).toContain('Pulse local');

      details.next(makeUser({ active_project_id: 'p2' }));
      await settle(fixture);
      expect(el.querySelector('h1')!.textContent).toContain('Kaval.AI examples');
      expect(agents.getSummaryStats).toHaveBeenCalledWith('p2');
    });

    it('waits for the user to be signed in', async () => {
      await render('owner', [makeProject()]);
      projects.getAll.calls.reset();
      details.next(null);
      await settle(fixture);
      expect(projects.getAll).not.toHaveBeenCalled();
    });
  });

  describe('the loaded project', () => {
    it('heads the page with the name, description, role, access and database', async () => {
      await render();
      expect(el.querySelector('h1')!.textContent).toContain('Pulse local');
      expect(text('description')).toBe('Runs recorded by the local Pulse API.');
      expect(text('role-badge')).toBe('owner');
      expect(text('access-badge')).toBe('read-only');
      expect(text('database')).toBe('localhost:6543 / kavalai_dev / pulse');
      expect(query<HTMLAnchorElement>('settings-link')!.getAttribute('href')).toBe('/settings');
    });

    it('marks a read-write project and leaves out a missing description', async () => {
      await render('viewer', [makeProject({ role: 'viewer', read_only: false, description: '' })]);
      expect(text('access-badge')).toBe('read-write');
      expect(text('role-badge')).toBe('viewer');
      expect(query('description')).toBeNull();
    });

    it('shows the figures of the last days', async () => {
      await render();
      expect(agents.getDailyStats).toHaveBeenCalledWith('p1', 7);
      expect(text('stat-sessions')).toContain('12');
      expect(text('stat-runs')).toContain('15');
      expect(text('stat-messages')).toContain('30');
      expect(text('stat-tasks')).toContain('41');
      expect(text('stat-input')).toContain('3,729');
      expect(text('stat-output')).toContain('394');
      expect(text('stat-embedding')).toContain('0');
    });

    it('counts a missing figure as zero', async () => {
      agents.getSummaryStats.and.returnValue(
        of({ total_sessions: 0, total_messages: 0, total_tasks: 0 })
      );
      await render();
      expect(text('stat-runs')).toContain('0');
      expect(text('stat-input')).toContain('0');
    });

    it('draws the activity, tokens and duration charts', async () => {
      await render();
      expect(query('activity-chart')!.querySelector('svg')).not.toBeNull();
      expect(query('token-chart')!.querySelector('svg')).not.toBeNull();
      expect(query('call-duration-chart')!.querySelector('canvas')).not.toBeNull();
      expect(query('run-duration-chart')!.querySelector('canvas')).not.toBeNull();
    });

    it('says so when no agent ran', async () => {
      agents.getDailyStats.and.returnValue(of(daily({})));
      await render();
      expect(text('no-runs')).toBe('No runs in the last 7 days.');
    });

    it('shows a spinner while the database is read', async () => {
      agents.getSummaryStats.and.returnValue(NEVER);
      await render();
      expect(query('stats-loading')).not.toBeNull();
    });

    it('shows why the database could not be read and reads it again on Retry', async () => {
      agents.getSummaryStats.and.returnValue(throwError(() => httpError(503, 'Database is not connected.')));
      await render();
      expect(text('stats-error')).toContain('Database is not connected.');
      expect(query('connection-card')).not.toBeNull();

      agents.getSummaryStats.and.returnValue(of(SUMMARY));
      query<HTMLButtonElement>('stats-retry')!.click();
      await settle(fixture);
      expect(query('stats-error')).toBeNull();
      expect(text('stat-sessions')).toContain('12');
    });
  });

  describe('connection strings', () => {
    function values(): string[] {
      return Array.from(el.querySelectorAll('[data-testid="string-value"]')).map((e) => e.textContent!.trim());
    }

    it('lists the PostgreSQL strings with the password masked, each with a copy button', async () => {
      await render();
      expect(text('string-sqlalchemy')).toContain('SQLAlchemy (PostgreSQL)');
      expect(values()).toEqual([
        'postgresql://kavalai_dev:••••••••@localhost:6543/kavalai_dev',
        'postgresql+asyncpg://kavalai_dev:••••••••@localhost:6543/kavalai_dev',
        'jdbc:postgresql://localhost:6543/kavalai_dev?user=kavalai_dev&password=••••••••',
        'KAVALAI_DB_URI=postgresql://kavalai_dev:••••••••@localhost:6543/kavalai_dev',
      ]);
      expect(query('connection-card')!.querySelectorAll('ui-copy-button').length).toBe(4);
    });

    it('names SQLite for a SQLite project and offers no password', async () => {
      await render('admin', [makeProject({ db_type: 'sqlite', db_name: '/data/agents.db', has_password: false })]);
      expect(text('string-sqlalchemy')).toContain('SQLAlchemy (SQLite)');
      expect(text('string-jdbc')).toContain('JDBC (SQLite)');
      expect(values()).toEqual([
        'sqlite:////data/agents.db',
        'sqlite+aiosqlite:////data/agents.db',
        'jdbc:sqlite:/data/agents.db',
        'KAVALAI_DB_URI=sqlite:////data/agents.db',
      ]);
      expect(text('database')).toBe('/data/agents.db');
      expect(query('toggle-password')).toBeNull();
    });

    it('lets only administrators show the password', async () => {
      await render('owner');
      expect(query('toggle-password')).toBeNull();
      expect(text('connection-card')).toContain('an administrator can show it');
    });

    it('offers nothing to show when no password is stored', async () => {
      await render('admin', [makeProject({ has_password: false })]);
      expect(query('toggle-password')).toBeNull();
    });

    it('asks the server for the password once, then shows and hides it', async () => {
      projects.revealPassword.and.returnValue(of({ db_password: 's3cret/+' }));
      await render('admin');
      const toggle = query<HTMLButtonElement>('toggle-password')!;
      expect(toggle.textContent).toContain('Show password');

      toggle.click();
      await settle(fixture);
      expect(projects.revealPassword).toHaveBeenCalledOnceWith('p1');
      expect(values()[0]).toBe('postgresql://kavalai_dev:s3cret%2F%2B@localhost:6543/kavalai_dev');
      expect(toggle.textContent).toContain('Hide password');

      toggle.click();
      await settle(fixture);
      expect(values()[0]).toContain('••••••••');

      toggle.click();
      await settle(fixture);
      expect(projects.revealPassword).toHaveBeenCalledTimes(1);
      expect(values()[0]).toContain('s3cret');
    });

    it('treats a missing password as empty', async () => {
      projects.revealPassword.and.returnValue(of({ db_password: null }));
      await render('admin');
      query<HTMLButtonElement>('toggle-password')!.click();
      await settle(fixture);
      expect(values()[0]).toBe('postgresql://kavalai_dev:@localhost:6543/kavalai_dev');
    });

    it('holds a spinner while the password is fetched', async () => {
      projects.revealPassword.and.returnValue(NEVER);
      await render('admin');
      query<HTMLButtonElement>('toggle-password')!.click();
      await settle(fixture);
      expect(query<HTMLButtonElement>('toggle-password')!.disabled).toBeTrue();
      expect(query('toggle-password')!.querySelector('ui-spinner')).not.toBeNull();
    });

    it('shows why the password could not be shown', async () => {
      projects.revealPassword.and.returnValue(throwError(() => httpError(403, 'Only administrators can see it.')));
      await render('admin');
      query<HTMLButtonElement>('toggle-password')!.click();
      await settle(fixture);
      expect(text('reveal-error')).toBe('Only administrators can see it.');
      expect(values()[0]).toContain('••••••••');
    });

    it('hides a revealed password when the project changes', async () => {
      projects.revealPassword.and.returnValue(of({ db_password: 'pw' }));
      await render('admin', [makeProject(), makeProject({ id: 'p2', name: 'Other' })]);
      query<HTMLButtonElement>('toggle-password')!.click();
      await settle(fixture);
      details.next(makeUser({ is_admin: true, active_project_id: 'p2' }));
      await settle(fixture);
      expect(values()[0]).toContain('••••••••');
    });
  });

  describe('helpers', () => {
    it('fills in defaults for a sparse PostgreSQL project', () => {
      const strings = connectionStrings(
        makeProject({ db_user: undefined, db_host: undefined, db_port: undefined, db_name: undefined }),
        '••••••••'
      );
      expect(strings[0].value).toBe('postgresql://user:••••••••@localhost:5432/kavalai');
    });

    it('names a default SQLite file', () => {
      expect(connectionStrings(makeProject({ db_type: 'sqlite', db_name: undefined }), '')[0].value).toBe(
        'sqlite:///agents.db'
      );
      expect(databaseLabel(makeProject({ db_type: 'sqlite', db_name: undefined }))).toBe('SQLite file');
    });

    it('reads no active project from an empty or "None" id', () => {
      expect(activeProjectOf(makeUser({ active_project_id: undefined }))).toBe('');
      expect(activeProjectOf(makeUser({ active_project_id: 'None' }))).toBe('');
      expect(activeProjectOf(makeUser({ active_project_id: 'p1' }))).toBe('p1');
    });
  });
});
