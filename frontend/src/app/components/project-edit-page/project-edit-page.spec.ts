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
import { Router, provideRouter } from '@angular/router';
import { BehaviorSubject, NEVER, Observable, Subject, of, throwError } from 'rxjs';
import { Project } from '../../models/project';
import { UserDetails } from '../../models/user-details';
import { ProjectService } from '../../services/project-service';
import { ToastService } from '../../services/toast-service';
import { UserService } from '../../services/user-service';
import { Member, Role, makeMember, makeProject, makeUser, settle, signedInAs } from '../../../testing';
import { ProjectEditPage } from './project-edit-page';

function httpError(status: number, detail?: string): HttpErrorResponse {
  return new HttpErrorResponse({ status, error: detail ? { detail } : null });
}

const ADA = makeMember();
const BOB = makeMember({ id: 'u2', name: 'Bob Byte', email: 'bob@example.com', role: 'viewer' });
const CARL = makeUser({ id: 'u3', name: 'Carl Core', email: 'carl@example.com', picture: '' });

interface Setup {
  url?: string;
  projects?: Project[] | Observable<Project[]>;
  members?: Member[] | Observable<Member[]>;
  user?: Partial<UserDetails>;
}

describe('ProjectEditPage', () => {
  let fixture: ComponentFixture<ProjectEditPage>;
  let el: HTMLElement;
  let projects: jasmine.SpyObj<ProjectService>;
  let users: jasmine.SpyObj<UserService>;
  let toast: jasmine.SpyObj<ToastService>;
  let router: Router;
  let details: BehaviorSubject<UserDetails | null>;

  function query<T extends Element = HTMLElement>(testId: string, root: ParentNode = el): T | null {
    return root.querySelector<T>(`[data-testid="${testId}"]`);
  }

  function text(testId: string): string {
    return query(testId)?.textContent?.replace(/\s+/g, ' ').trim() ?? '';
  }

  function button(testId: string, root: ParentNode = el): HTMLButtonElement {
    return query<HTMLButtonElement>(testId, root)!;
  }

  async function click(testId: string, root: ParentNode = el): Promise<void> {
    button(testId, root).click();
    await settle(fixture);
  }

  async function type(testId: string, value: string): Promise<void> {
    const input = query<HTMLInputElement | HTMLTextAreaElement>(testId)!;
    input.value = value;
    input.dispatchEvent(new Event('input'));
    input.dispatchEvent(new Event('blur'));
    await settle(fixture);
  }

  async function choose(testId: string, value: string, root: ParentNode = el): Promise<void> {
    const select = query<HTMLSelectElement>(testId, root)!;
    select.value = value;
    select.dispatchEvent(new Event('change'));
    await settle(fixture);
  }

  function row(id: string): HTMLElement {
    return query(`member-${id}`)!;
  }

  /** Sign in as `role` and open the page at `url`. */
  async function render(role: Role = 'owner', setup: Setup = {}): Promise<void> {
    const list = setup.projects ?? [makeProject({ role: role === 'viewer' ? 'viewer' : 'owner' })];
    const members = setup.members ?? [ADA, BOB];
    users = signedInAs(role, setup.user);
    details = new BehaviorSubject<UserDetails | null>(makeUser({ is_admin: role === 'admin', ...setup.user }));
    (Object.getOwnPropertyDescriptor(users, 'userDetails')!.get as jasmine.Spy).and.returnValue(details.asObservable());
    users.getIsAdmin.and.callFake(() => !!details.value?.is_admin);
    users.getUserDetailsValue.and.callFake(() => details.value);
    users.getUsers.and.returnValue(of([makeUser(), makeUser({ id: 'u2', name: 'Bob Byte' }), CARL]));
    projects.getAll.and.returnValue(Array.isArray(list) ? of(list) : list);
    projects.getMembers.and.returnValue(Array.isArray(members) ? of(members) : members);
    TestBed.overrideProvider(UserService, { useValue: users });
    router = TestBed.inject(Router);
    spyOnProperty(router, 'url').and.returnValue(setup.url ?? '/settings');
    spyOn(router, 'navigate').and.resolveTo(true);
    fixture = TestBed.createComponent(ProjectEditPage);
    el = fixture.nativeElement;
    fixture.detectChanges();
    await settle(fixture);
  }

  beforeEach(async () => {
    projects = jasmine.createSpyObj<ProjectService>('ProjectService', [
      'getAll',
      'create',
      'update',
      'delete',
      'testConnection',
      'getMembers',
      'addMember',
      'updateMember',
      'removeMember',
    ]);
    toast = jasmine.createSpyObj<ToastService>('ToastService', ['success', 'error']);
    await TestBed.configureTestingModule({
      imports: [ProjectEditPage],
      providers: [
        provideRouter([]),
        { provide: ProjectService, useValue: projects },
        { provide: ToastService, useValue: toast },
        { provide: UserService, useValue: signedInAs() },
      ],
    }).compileComponents();
  });


  describe('loading the settings', () => {
    it('shows a spinner while the project loads', async () => {
      await render('owner', { projects: NEVER });
      expect(query('loading')).not.toBeNull();
    });

    it('waits for the user to be signed in', async () => {
      users = signedInAs('owner');
      details = new BehaviorSubject<UserDetails | null>(null);
      (Object.getOwnPropertyDescriptor(users, 'userDetails')!.get as jasmine.Spy).and.returnValue(details.asObservable());
      TestBed.overrideProvider(UserService, { useValue: users });
      fixture = TestBed.createComponent(ProjectEditPage);
      el = fixture.nativeElement;
      fixture.detectChanges();
      await settle(fixture);
      expect(query('loading')).not.toBeNull();
      expect(projects.getAll).not.toHaveBeenCalled();
    });

    it('shows the server’s reason and loads again on Retry', async () => {
      await render('owner', { projects: throwError(() => httpError(500, 'Backoffice database is down.')) });
      expect(text('load-error')).toContain('Backoffice database is down.');

      projects.getAll.and.returnValue(of([makeProject()]));
      await click('retry');
      expect(query<HTMLInputElement>('name')!.value).toBe('Pulse local');
    });

    it('offers administrators a new project when there is none', async () => {
      await render('admin', { projects: [] });
      expect(text('no-project')).toContain('No project yet');
      expect(query('new-project')!.getAttribute('href')).toBe('/projects/new');
    });

    it('tells everyone else to ask an administrator', async () => {
      await render('viewer', { projects: [] });
      expect(text('no-project')).toContain('Ask an administrator');
      expect(query('new-project')).toBeNull();
    });

    it('asks to choose a project when none is active', async () => {
      await render('owner', { user: { active_project_id: '' } });
      expect(text('no-project')).toContain('Choose a project in the sidebar');
    });

    it('fills the cards from the active project and follows the sidebar', async () => {
      const other = makeProject({ id: 'p2', name: 'Kaval.AI examples', db_schema: 'agents', rag_schema: 'rag' });
      await render('owner', { projects: [makeProject(), other] });
      expect(el.querySelector('h1')!.textContent).toContain('Pulse local');
      expect(query<HTMLInputElement>('db-host')!.value).toBe('localhost');
      expect(query<HTMLInputElement>('db-port')!.value).toBe('6543');
      expect(query<HTMLInputElement>('db-schema')!.value).toBe('pulse');
      expect(query<HTMLInputElement>('db-password')!.placeholder).toBe('Unchanged — type to replace');
      expect(query<HTMLInputElement>('read-only')!.checked).toBeTrue();
      expect(text('role-badge')).toBe('owner');

      details.next(makeUser({ active_project_id: 'p2' }));
      await settle(fixture);
      expect(query<HTMLInputElement>('name')!.value).toBe('Kaval.AI examples');
      expect(query<HTMLInputElement>('rag-schema')!.value).toBe('rag');
      expect(projects.getMembers).toHaveBeenCalledWith('p2');
    });

    it('fills a sparse project with defaults', async () => {
      await render('owner', {
        projects: [
          makeProject({
            description: undefined,
            db_type: undefined,
            db_host: undefined,
            db_port: undefined,
            db_user: undefined,
            db_name: undefined,
            db_schema: undefined,
            read_only: false,
            has_password: false,
          }),
        ],
      });
      expect(query<HTMLSelectElement>('db-type')!.value).toBe('postgresql');
      expect(query<HTMLInputElement>('db-port')!.value).toBe('5432');
      expect(query<HTMLInputElement>('db-password')!.placeholder).toBe('');
      expect(query('password-hint')).toBeNull();
      expect(query<HTMLInputElement>('read-only')!.checked).toBeFalse();
    });

    it('shows a SQLite project as its file', async () => {
      await render('owner', { projects: [makeProject({ db_type: 'sqlite', db_name: '/data/agents.db' })] });
      expect(query<HTMLInputElement>('db-name')!.value).toBe('/data/agents.db');
      expect(query('db-host')).toBeNull();
      expect(query('db-password')).toBeNull();
    });
  });

  describe('permissions', () => {
    it('gives owners every card enabled', async () => {
      await render('owner');
      expect(query('viewer-note')).toBeNull();
      expect(query<HTMLInputElement>('name')!.disabled).toBeFalse();
      expect(query('save-general')).not.toBeNull();
      expect(query('save-database')).not.toBeNull();
      expect(query('danger-card')).not.toBeNull();
      expect(query('member-role', row('u2'))).not.toBeNull();
    });

    it('shows viewers the settings read-only, with a note', async () => {
      await render('viewer');
      expect(text('viewer-note')).toContain('shown read-only');
      expect(query<HTMLInputElement>('name')!.disabled).toBeTrue();
      expect(query<HTMLInputElement>('db-host')!.disabled).toBeTrue();
      expect(query<HTMLInputElement>('read-only')!.disabled).toBeTrue();
      expect(query('save-general')).toBeNull();
      expect(query('save-database')).toBeNull();
      expect(query('danger-card')).toBeNull();
      expect(query('member-role')).toBeNull();
      expect(query('remove-member')).toBeNull();
      expect(row('u2').querySelector('ui-badge')!.textContent!.trim()).toBe('viewer');
      expect(users.getUsers).not.toHaveBeenCalled();
    });

    it('lets an administrator who is a viewer manage members but not change settings', async () => {
      await render('admin', { projects: [makeProject({ role: 'viewer' })] });
      expect(query('viewer-note')).not.toBeNull();
      expect(query('save-general')).toBeNull();
      expect(query('member-role', row('u2'))).not.toBeNull();
      expect(query('add-form')).not.toBeNull();
    });
  });

  describe('General', () => {
    it('refuses an empty name', async () => {
      await render();
      await type('name', '   ');
      await click('save-general');
      expect(projects.update).not.toHaveBeenCalled();
      expect(text('general-card')).toContain('The name is required.');
    });

    it('holds the button busy while saving', async () => {
      projects.update.and.returnValue(NEVER);
      await render();
      await click('save-general');
      expect(button('save-general').disabled).toBeTrue();
      expect(query('save-general')!.querySelector('ui-spinner')).not.toBeNull();
    });

    it('shows the server’s refusal in the card', async () => {
      projects.update.and.returnValue(throwError(() => httpError(403, 'Only project owners can do this.')));
      await render();
      await click('save-general');
      expect(text('general-error')).toBe('Only project owners can do this.');
      expect(toast.success).not.toHaveBeenCalled();
    });

    it('saves the name and description alone', async () => {
      projects.update.and.callFake((_id, body) => of(makeProject({ ...body, role: undefined })));
      await render();
      await type('name', ' Pulse staging ');
      await type('description', 'Staging runs.');
      await click('save-general');
      expect(projects.update).toHaveBeenCalledWith('p1', { name: 'Pulse staging', description: 'Staging runs.' });
      expect(toast.success).toHaveBeenCalledWith('Saved');
      expect(el.querySelector('h1')!.textContent).toContain('Pulse staging');
      expect(text('role-badge')).toBe('owner');
    });
  });

  describe('Database', () => {
    it('refuses an empty database and an impossible port', async () => {
      await render();
      await type('db-name', '');
      await type('db-port', '70000');
      await click('save-database');
      expect(projects.update).not.toHaveBeenCalled();
      expect(text('database-card')).toContain('The database is required.');
      expect(text('database-card')).toContain('The port is not valid.');
    });

    it('asks for the password again when the server changes', async () => {
      await render();
      await type('db-host', 'db.example.com');
      await click('save-database');
      expect(projects.update).not.toHaveBeenCalled();
      expect(text('database-card')).toContain('Type the password again');
      expect(text('password-hint')).toContain('changing any of them asks for it again');
    });

    it('sends the typed password with a changed server', async () => {
      projects.update.and.callFake((_id, body) => of(makeProject({ ...body })));
      await render();
      await type('db-host', 'db.example.com');
      await type('db-password', 'new-secret');
      await click('save-database');
      expect(projects.update).toHaveBeenCalledWith('p1', {
        db_type: 'postgresql',
        db_name: 'kavalai_dev',
        db_host: 'db.example.com',
        db_port: 6543,
        db_user: 'kavalai_dev',
        db_schema: 'pulse',
        rag_schema: undefined,
        db_password: 'new-secret',
      });
      expect(query<HTMLInputElement>('db-password')!.value).toBe('');
      expect(toast.success).toHaveBeenCalledWith('Saved');
    });

    it('keeps the stored password when only the schema changes', async () => {
      projects.update.and.callFake((_id, body) => of(makeProject({ ...body })));
      await render();
      await type('db-schema', 'agents');
      await type('rag-schema', 'rag');
      await click('save-database');
      const body = projects.update.calls.mostRecent().args[1];
      expect(body.db_schema).toBe('agents');
      expect(body.rag_schema).toBe('rag');
      expect('db_password' in body).toBeFalse();
    });

    it('sends only the file for a SQLite project', async () => {
      projects.update.and.callFake((_id, body) => of(makeProject({ ...body })));
      await render();
      await choose('db-type', 'sqlite');
      await type('db-name', '/data/agents.db');
      await click('save-database');
      expect(projects.update).toHaveBeenCalledWith('p1', { db_type: 'sqlite', db_name: '/data/agents.db' });
    });

    it('holds the button busy while saving and shows a refusal', async () => {
      const reply = new Subject<Project>();
      projects.update.and.returnValue(reply);
      await render();
      await click('save-database');
      expect(button('save-database').disabled).toBeTrue();

      reply.error(httpError(400, 'Enter the database password again.'));
      await settle(fixture);
      expect(text('database-error')).toBe('Enter the database password again.');
      expect(button('save-database').disabled).toBeFalse();
    });

    it('tests the typed settings and shows the database’s answer', async () => {
      const reply = new Subject<{ status: string; message?: string }>();
      projects.testConnection.and.returnValue(reply);
      await render();
      await type('db-schema', 'agents');
      await click('test-connection');
      expect(button('test-connection').disabled).toBeTrue();
      expect(query('test-connection')!.querySelector('ui-spinner')).not.toBeNull();
      expect(projects.testConnection.calls.mostRecent().args[0]).toBe('p1');
      expect(projects.testConnection.calls.mostRecent().args[1]).toEqual(jasmine.objectContaining({ db_schema: 'agents' }));

      reply.next({ status: 'success', message: 'Connection successful' });
      reply.complete();
      await settle(fixture);
      expect(text('test-result')).toBe('Connection successful');
      expect(query('test-result')!.getAttribute('tone')).toBe('success');
    });

    it('shows the database’s error when the test fails', async () => {
      projects.testConnection.and.returnValue(
        throwError(() => httpError(400, 'Connection failed: password authentication failed'))
      );
      await render();
      await click('test-connection');
      expect(text('test-result')).toBe('Connection failed: password authentication failed');
      expect(query('test-result')!.getAttribute('tone')).toBe('error');
    });

    it('words a reply without a message', async () => {
      projects.testConnection.and.returnValue(of({ status: 'error' }));
      await render();
      await click('test-connection');
      expect(text('test-result')).toBe('The connection failed.');

      projects.testConnection.and.returnValue(of({ status: 'success' }));
      await click('test-connection');
      expect(text('test-result')).toBe('The connection works.');
    });

    it('lets a viewer test the saved settings', async () => {
      projects.testConnection.and.returnValue(of({ status: 'success', message: 'Connection successful' }));
      await render('viewer');
      await click('test-connection');
      expect(projects.testConnection).toHaveBeenCalledWith('p1', {});
      expect(text('database-card')).toContain('Tests the saved settings.');
    });
  });

  describe('Access', () => {
    it('saves the switch as soon as it is turned', async () => {
      projects.update.and.callFake((_id, body) => of(makeProject({ ...body })));
      await render();
      query<HTMLInputElement>('read-only')!.click();
      await settle(fixture);
      expect(projects.update).toHaveBeenCalledWith('p1', { read_only: false });
      expect(toast.success).toHaveBeenCalledWith('Saved');
    });

    it('turns back and says why when the server refuses', async () => {
      projects.update.and.returnValue(throwError(() => httpError(403, 'Only project owners can do this.')));
      await render();
      query<HTMLInputElement>('read-only')!.click();
      await settle(fixture);
      expect(text('access-error')).toBe('Only project owners can do this.');
      expect(query<HTMLInputElement>('read-only')!.checked).toBeTrue();
    });

    it('holds a spinner while saving', async () => {
      projects.update.and.returnValue(NEVER);
      await render();
      query<HTMLInputElement>('read-only')!.click();
      await settle(fixture);
      expect(query('access-card')!.querySelector('ui-spinner')).not.toBeNull();
    });
  });

  describe('Members', () => {
    it('shows a spinner while the members load', async () => {
      await render('owner', { members: NEVER });
      expect(query('members-loading')).not.toBeNull();
    });

    it('shows why the members could not be loaded and loads them on Retry', async () => {
      await render('owner', { members: throwError(() => httpError(500, 'Members are unavailable.')) });
      expect(text('members-load-error')).toContain('Members are unavailable.');
      projects.getMembers.and.returnValue(of([ADA]));
      await click('members-retry');
      expect(query('member-u1')).not.toBeNull();
    });

    it('lists each member with avatar, name, email and the user marked', async () => {
      await render();
      expect(row('u1').querySelector('ui-avatar')).not.toBeNull();
      expect(row('u1').textContent).toContain('Ada Lovelace');
      expect(row('u1').textContent).toContain('ada@example.com');
      expect(query('you', row('u1'))).not.toBeNull();
      expect(query('you', row('u2'))).toBeNull();
    });

    it('names a member without a name by the address', async () => {
      await render('owner', { members: [ADA, makeMember({ id: 'u9', name: '', email: 'x@example.com', role: 'viewer' })] });
      expect(row('u9').textContent).toContain('x@example.com');
    });

    it('disables the last owner’s controls and says why', async () => {
      await render();
      expect(query<HTMLSelectElement>('member-role', row('u1'))!.disabled).toBeTrue();
      expect(button('remove-member', row('u1')).disabled).toBeTrue();
      expect(text('last-owner')).toContain('Make someone else an owner first');
      expect(button('remove-member', row('u2')).disabled).toBeFalse();
    });

    it('changes a role', async () => {
      projects.updateMember.and.returnValue(of({ status: 'updated' }));
      await render();
      await choose('member-role', 'owner', row('u2'));
      expect(projects.updateMember).toHaveBeenCalledWith('p1', 'u2', 'owner');
      expect(toast.success).toHaveBeenCalledWith('Role changed');
      expect(query<HTMLSelectElement>('member-role', row('u1'))!.disabled).toBeFalse();
      expect(query('last-owner')).toBeNull();
    });

    it('ignores choosing the role a member has', async () => {
      await render();
      await choose('member-role', 'viewer', row('u2'));
      expect(projects.updateMember).not.toHaveBeenCalled();
    });

    it('puts the role back and says why when the change is refused', async () => {
      projects.updateMember.and.returnValue(throwError(() => httpError(400, 'Cannot demote the last owner.')));
      await render();
      await choose('member-role', 'owner', row('u2'));
      expect(text('member-error')).toBe('Cannot demote the last owner.');
      expect(query<HTMLSelectElement>('member-role', row('u2'))!.value).toBe('viewer');
    });

    it('asks in a dialog before the user makes themself a viewer', async () => {
      projects.updateMember.and.returnValue(of({ status: 'updated' }));
      const owners = [ADA, makeMember({ id: 'u2', name: 'Bob Byte', email: 'bob@example.com', role: 'owner' })];
      await render('owner', { members: owners });
      await choose('member-role', 'viewer', row('u1'));
      expect(projects.updateMember).not.toHaveBeenCalled();
      expect(query<HTMLSelectElement>('member-role', row('u1'))!.value).toBe('owner');

      await click('cancel-demote');
      expect(projects.updateMember).not.toHaveBeenCalled();

      await choose('member-role', 'viewer', row('u1'));
      await click('confirm-demote');
      expect(projects.updateMember).toHaveBeenCalledWith('p1', 'u1', 'viewer');
      expect(text('viewer-note')).toContain('shown read-only');
      expect(query('member-role')).toBeNull();
    });

    it('removes a member after a second click', async () => {
      projects.removeMember.and.returnValue(of({ status: 'removed' }));
      await render();
      await click('remove-member', row('u2'));
      expect(projects.removeMember).not.toHaveBeenCalled();
      expect(text('confirm-remove-member')).toBe('Remove Bob Byte');

      await click('cancel-remove');
      expect(query('confirm-remove-member')).toBeNull();

      await click('remove-member', row('u2'));
      await click('confirm-remove-member');
      expect(projects.removeMember).toHaveBeenCalledWith('p1', 'u2');
      expect(query('member-u2')).toBeNull();
      expect(toast.success).toHaveBeenCalledWith('Member removed');
    });

    it('holds the row busy while removing and shows a refusal', async () => {
      const reply = new Subject<unknown>();
      projects.removeMember.and.returnValue(reply);
      await render();
      await click('remove-member', row('u2'));
      await click('confirm-remove-member');
      expect(button('remove-member', row('u2')).disabled).toBeTrue();
      expect(row('u2').querySelector('ui-spinner')).not.toBeNull();

      reply.error(httpError(403, 'Only project owners can do this.'));
      await settle(fixture);
      expect(text('member-error')).toBe('Only project owners can do this.');
      expect(query('member-u2')).not.toBeNull();
    });

    it('takes the user to the overview after they leave the project', async () => {
      projects.removeMember.and.returnValue(of({ status: 'removed' }));
      const owners = [ADA, makeMember({ id: 'u2', name: 'Bob Byte', role: 'owner' })];
      await render('owner', { members: owners });
      await click('remove-member', row('u1'));
      expect(text('confirm-remove-member')).toBe('Leave the project');
      await click('confirm-remove-member');
      expect(users.updateUserDetails).toHaveBeenCalled();
      expect(router.navigate).toHaveBeenCalledWith(['/']);
    });

    it('offers to add only users who are not members yet', async () => {
      projects.addMember.and.returnValue(of({ status: 'added' }));
      await render();
      const options = Array.from(query<HTMLSelectElement>('new-member')!.options).map((o) => o.value);
      expect(options).toEqual(['', 'u3']);
      expect(button('add-member').disabled).toBeTrue();

      await choose('new-member', 'u3');
      await choose('new-member-role', 'owner');
      await click('add-member');
      expect(projects.addMember).toHaveBeenCalledWith('p1', 'u3', 'owner');
      expect(row('u3').textContent).toContain('Carl Core');
      expect(toast.success).toHaveBeenCalledWith('Member added');
      expect(text('no-candidates')).toContain('member already');
    });

    it('shows why a user could not be added', async () => {
      projects.addMember.and.returnValue(throwError(() => httpError(400, 'User is already a member.')));
      await render();
      await choose('new-member', 'u3');
      await click('add-member');
      expect(text('member-error')).toBe('User is already a member.');
      expect(query('member-u3')).toBeNull();
    });

    it('adds nobody when no user is chosen', async () => {
      await render();
      query<HTMLFormElement>('add-form')!.dispatchEvent(new Event('submit'));
      await settle(fixture);
      expect(projects.addMember).not.toHaveBeenCalled();
    });

    it('holds Add busy while adding', async () => {
      projects.addMember.and.returnValue(NEVER);
      await render();
      await choose('new-member', 'u3');
      await click('add-member');
      expect(button('add-member').disabled).toBeTrue();
      expect(query('add-member')!.querySelector('ui-spinner')).not.toBeNull();
    });

    it('tells an owner who may not list users to ask an administrator', async () => {
      await render('owner');
      users.getUsers.and.returnValue(throwError(() => httpError(403, 'Forbidden')));
      fixture.componentInstance.load();
      await settle(fixture);
      expect(text('users-forbidden')).toContain('ask an administrator to add members');
      expect(query('add-form')).toBeNull();
    });

    it('shows any other failure to list the users', async () => {
      await render('admin');
      users.getUsers.and.returnValue(throwError(() => httpError(500, 'Users are unavailable.')));
      fixture.componentInstance.load();
      await settle(fixture);
      expect(text('users-error')).toBe('Users are unavailable.');
    });
  });

  describe('Danger zone', () => {
    async function openDialog(): Promise<void> {
      await click('delete-project');
    }

    it('deletes only once the name is typed', async () => {
      projects.delete.and.returnValue(of(undefined));
      await render('owner', { projects: [makeProject(), makeProject({ id: 'p2', name: 'Other' })] });
      await openDialog();
      expect(button('confirm-delete').disabled).toBeTrue();
      await type('confirm-delete-name', 'Pulse');
      expect(button('confirm-delete').disabled).toBeTrue();
      await type('confirm-delete-name', 'Pulse local');
      expect(button('confirm-delete').disabled).toBeFalse();

      await click('confirm-delete');
      expect(projects.delete).toHaveBeenCalledWith('p1');
      expect(toast.success).toHaveBeenCalledWith('Project deleted');
      expect(users.setActiveProject).toHaveBeenCalledWith('p2');
      expect(router.navigate).toHaveBeenCalledWith(['/']);
    });

    it('leaves no project active when it was the last', async () => {
      projects.delete.and.returnValue(of(undefined));
      await render();
      await openDialog();
      await type('confirm-delete-name', 'Pulse local');
      projects.getAll.and.returnValue(of([]));
      await click('confirm-delete');
      expect(users.setActiveProject).not.toHaveBeenCalled();
      expect(users.updateUserDetails).toHaveBeenCalled();
      expect(router.navigate).toHaveBeenCalledWith(['/']);
    });

    it('still goes to the overview when the projects cannot be listed afterwards', async () => {
      projects.delete.and.returnValue(of(undefined));
      await render();
      await openDialog();
      await type('confirm-delete-name', 'Pulse local');
      projects.getAll.and.returnValue(throwError(() => httpError(500)));
      await click('confirm-delete');
      expect(users.updateUserDetails).toHaveBeenCalled();
      expect(router.navigate).toHaveBeenCalledWith(['/']);
    });

    it('holds the dialog busy and shows a refusal', async () => {
      const reply = new Subject<void>();
      projects.delete.and.returnValue(reply);
      await render();
      await openDialog();
      await type('confirm-delete-name', 'Pulse local');
      await click('confirm-delete');
      expect(button('confirm-delete').disabled).toBeTrue();
      expect(button('cancel-delete').disabled).toBeTrue();
      fixture.componentInstance.closeDelete();
      expect(fixture.componentInstance.deleteOpen()).toBeTrue();

      reply.error(httpError(403, 'Only project owners can do this.'));
      await settle(fixture);
      expect(text('delete-error')).toBe('Only project owners can do this.');
      expect(router.navigate).not.toHaveBeenCalled();
    });

    it('deletes nothing while the typed name differs', async () => {
      await render();
      await openDialog();
      await type('confirm-delete-name', 'Pulse');
      fixture.componentInstance.deleteProject();
      expect(projects.delete).not.toHaveBeenCalled();
    });

    it('closes on Cancel and starts empty when opened again', async () => {
      await render();
      await openDialog();
      await type('confirm-delete-name', 'Pulse local');
      await click('cancel-delete');
      expect(fixture.componentInstance.deleteOpen()).toBeFalse();
      await openDialog();
      expect(query<HTMLInputElement>('confirm-delete-name')!.value).toBe('');
    });
  });

  describe('New project', () => {
    it('is refused to anyone but an administrator', async () => {
      await render('owner', { url: '/projects/new' });
      expect(text('create-forbidden')).toContain('Only administrators create projects');
      expect(query('create')).toBeNull();
    });

    it('shows General, Database and Access with one Create button', async () => {
      await render('admin', { url: '/projects/new?from=menu' });
      expect(el.querySelector('h1')!.textContent).toContain('New project');
      expect(query('general-card')).not.toBeNull();
      expect(query('database-card')).not.toBeNull();
      expect(query('access-card')).not.toBeNull();
      expect(query('members-card')).toBeNull();
      expect(query('danger-card')).toBeNull();
      expect(query('save-general')).toBeNull();
      expect(query('save-database')).toBeNull();
      expect(query('create')).not.toBeNull();
      expect(projects.getAll).not.toHaveBeenCalled();
    });

    it('refuses a project without a name or database', async () => {
      await render('admin', { url: '/projects/new' });
      await click('create');
      expect(projects.create).not.toHaveBeenCalled();
      expect(text('general-card')).toContain('The name is required.');
      expect(text('database-card')).toContain('The database is required.');
    });

    it('tests the typed settings before Create', async () => {
      projects.testConnection.and.returnValue(of({ status: 'success', message: 'Connection successful' }));
      await render('admin', { url: '/projects/new' });
      await click('test-connection');
      expect(projects.testConnection).not.toHaveBeenCalled();

      await type('db-name', 'agents');
      await type('db-password', 'pw');
      await click('test-connection');
      expect(projects.testConnection).toHaveBeenCalledWith(
        'new',
        jasmine.objectContaining({ db_host: 'localhost', db_port: 5432, db_name: 'agents', db_password: 'pw' })
      );
      expect(text('test-result')).toBe('Connection successful');
    });

    it('holds Create busy and shows the server’s refusal', async () => {
      const reply = new Subject<Project>();
      projects.create.and.returnValue(reply);
      await render('admin', { url: '/projects/new' });
      await type('name', 'Pulse staging');
      await type('db-name', 'agents');
      await click('create');
      expect(button('create').disabled).toBeTrue();

      reply.error(httpError(403, 'Only administrators can create new projects.'));
      await settle(fixture);
      expect(text('create-error')).toBe('Only administrators can create new projects.');
      expect(router.navigate).not.toHaveBeenCalled();
    });

    it('creates the project, makes it active and opens the overview', async () => {
      projects.create.and.returnValue(of(makeProject({ id: 'p9' })));
      await render('admin', { url: '/projects/new' });
      await type('name', 'Local file');
      await choose('db-type', 'sqlite');
      await type('db-name', '/data/agents.db');
      query<HTMLInputElement>('read-only')!.click();
      await settle(fixture);
      expect(projects.update).not.toHaveBeenCalled();

      await click('create');
      expect(projects.create).toHaveBeenCalledWith({
        name: 'Local file',
        description: '',
        db_type: 'sqlite',
        db_name: '/data/agents.db',
        read_only: false,
      });
      expect(users.setActiveProject).toHaveBeenCalledWith('p9');
      expect(toast.success).toHaveBeenCalledWith('Project created');
      expect(router.navigate).toHaveBeenCalledWith(['/']);
    });
  });
});
