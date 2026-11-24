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
import { Router, provideRouter } from '@angular/router';
import { Component } from '@angular/core';
import { Subject, of, throwError } from 'rxjs';
import { Shell } from './shell';
import { ProjectService } from '../services/project-service';
import { UserService } from '../services/user-service';
import { Role, makeProject, settle, signedInAs } from '../../testing';

@Component({ template: '<p data-testid="page">A page</p>' })
class Page {}

describe('Shell', () => {
  let fixture: ComponentFixture<Shell>;
  let el: HTMLElement;
  let userService: jasmine.SpyObj<UserService>;
  let projectService: jasmine.SpyObj<ProjectService>;
  let changes: Subject<void>;

  const pulse = makeProject();
  const sqlite = makeProject({
    id: 'p2',
    name: 'Examples',
    db_type: 'sqlite',
    db_name: '/data/agents.db',
    role: 'viewer',
    read_only: false,
  });

  async function render(
    role: Role = 'owner',
    projects = [pulse, sqlite],
    activeProjectId = 'p1'
  ): Promise<void> {
    userService = signedInAs(role, { active_project_id: activeProjectId });
    changes = new Subject<void>();
    projectService = jasmine.createSpyObj('ProjectService', ['getAll'], {
      changes: changes.asObservable(),
    });
    projectService.getAll.and.returnValue(of(projects));
    TestBed.configureTestingModule({
      imports: [Shell],
      providers: [
        provideRouter([
          { path: '', component: Page },
          { path: 'conversations', component: Page },
          { path: 'profile', component: Page },
          { path: 'projects/new', component: Page },
        ]),
        { provide: UserService, useValue: userService },
        { provide: ProjectService, useValue: projectService },
      ],
    });
    fixture = TestBed.createComponent(Shell);
    el = fixture.nativeElement;
    fixture.detectChanges();
    await settle(fixture);
  }

  function query<T extends HTMLElement = HTMLElement>(testId: string): T | null {
    return el.querySelector<T>(`[data-testid="${testId}"]`);
  }

  function navLabels(group: string): string[] {
    return Array.from(el.querySelectorAll(`[data-testid="${group}"] ui-nav-item`)).map(
      (item) => item.textContent!.trim()
    );
  }

  it('renders the page beside the sidebar', async () => {
    await render();
    await TestBed.inject(Router).navigateByUrl('/');
    await settle(fixture);
    expect(query('page')).not.toBeNull();
  });

  it("lists the project's pages, and Users for administrators only", async () => {
    await render('owner');
    expect(navLabels('project-nav')).toEqual([
      'Overview',
      'Conversations',
      'Runs',
      'Model calls',
      'Agents',
      'RAG',
    ]);
    expect(navLabels('settings-nav')).toEqual(['Settings']);
    expect(query('admin-nav')).toBeNull();

    TestBed.resetTestingModule();
    await render('admin');
    expect(navLabels('admin-nav')).toEqual(['Users']);
  });

  it('has no settings link without an active project', async () => {
    await render('owner', []);
    expect(query('settings-nav')).toBeNull();
  });

  it('offers a new project to administrators only', async () => {
    await render('owner');
    query('project-switcher')!.click();
    await settle(fixture);
    expect(query('new-project')).toBeNull();

    TestBed.resetTestingModule();
    await render('admin');
    const router = TestBed.inject(Router);
    spyOn(router, 'navigate').and.resolveTo(true);
    query('project-switcher')!.click();
    await settle(fixture);
    query('new-project')!.click();
    await settle(fixture);
    expect(router.navigate).toHaveBeenCalledWith(['/projects/new']);
    expect(query('project-menu')).toBeNull();
  });

  it('names the active project and where its runs are', async () => {
    await render();
    const switcher = query('project-switcher')!;
    expect(switcher.textContent).toContain('Pulse local');
    expect(switcher.textContent).toContain('localhost:6543 / pulse');
    expect(el.querySelector('.switcher-access')).toBeNull();
  });

  it('marks a project the backoffice may write to', async () => {
    await render('owner', [pulse, sqlite], 'p2');
    expect(query('project-switcher')!.textContent).toContain('agents.db');
    expect(el.querySelector('.switcher-access')!.textContent!.trim()).toBe('read-write');
  });

  it('switches the active project from its menu, reloading the list on open', async () => {
    await render();
    query('project-switcher')!.click();
    await settle(fixture);
    expect(projectService.getAll).toHaveBeenCalledTimes(2);

    const options = el.querySelectorAll('[data-testid="project-menu"] button');
    expect(options.length).toBe(2);
    expect(query('project-option-p1')!.getAttribute('aria-current')).toBe('true');
    expect(query('project-option-p2')!.textContent).toContain('agents.db');

    query('project-option-p2')!.click();
    await settle(fixture);
    expect(userService.setActiveProject).toHaveBeenCalledOnceWith('p2');
    expect(query('project-menu')).toBeNull();
  });

  it('reloads the projects when one is created, renamed or deleted', async () => {
    await render();
    projectService.getAll.and.returnValue(of([makeProject({ name: 'Pulse renamed' })]));
    changes.next();
    await settle(fixture);
    expect(query('project-switcher')!.textContent).toContain('Pulse renamed');
  });

  it('closes the project menu when focus leaves the switcher', async () => {
    await render();
    query('project-switcher')!.click();
    await settle(fixture);
    const switcher = el.querySelector('.switcher')!;
    switcher.dispatchEvent(new FocusEvent('focusout', { relatedTarget: null }));
    await settle(fixture);
    expect(query('project-menu')).toBeNull();
  });

  it('does not save a choice of the project already active', async () => {
    await render();
    query('project-switcher')!.click();
    await settle(fixture);
    query('project-option-p1')!.click();
    expect(userService.setActiveProject).not.toHaveBeenCalled();
  });

  it('says what to do without a project, by role', async () => {
    await render('viewer', []);
    expect(query('project-switcher')!.textContent).toContain('No project yet');
    query('project-switcher')!.click();
    await settle(fixture);
    expect(el.querySelector('.menu-empty')!.textContent).toContain('Ask an administrator');

    TestBed.resetTestingModule();
    await render('admin', []);
    query('project-switcher')!.click();
    await settle(fixture);
    expect(el.querySelector('.menu-empty')!.textContent).not.toContain('Ask an administrator');
    expect(query('new-project')).not.toBeNull();
  });

  it('says when the projects cannot be loaded', async () => {
    await render();
    projectService.getAll.and.returnValue(throwError(() => new Error('down')));
    query('project-switcher')!.click();
    await settle(fixture);
    expect(el.querySelector('[data-testid="project-menu"] [role="alert"]')).not.toBeNull();
  });

  it('opens the profile and signs out from the user menu', async () => {
    await render('admin');
    const router = TestBed.inject(Router);
    spyOn(router, 'navigate').and.resolveTo(true);
    query('user-button')!.click();
    await settle(fixture);
    expect(query('user-menu')!.textContent).toContain('ada@example.com');
    expect(query('user-menu')!.textContent).toContain('administrator');

    query('profile')!.click();
    await settle(fixture);
    expect(router.navigate).toHaveBeenCalledWith(['/profile']);
    expect(query('user-menu')).toBeNull();

    query('user-button')!.click();
    await settle(fixture);
    query('sign-out')!.click();
    expect(userService.logout).toHaveBeenCalled();
  });

  it('shows initials for a user without a picture', async () => {
    await render();
    expect(el.querySelector('.user-button img')).toBeNull();
    expect(el.querySelector('.user-button ui-avatar')!.textContent!.trim()).toBe('AL');
  });

  it('closes a sheet when focus leaves it, but not when it moves inside', async () => {
    await render();
    const shell = fixture.componentInstance;
    shell.menuOpen.set(true);
    const container = el.querySelector('.user-menu') as HTMLElement;
    const inside = el.querySelector('.user-button') as HTMLElement;
    shell.closeOnFocusOut(
      { currentTarget: container, relatedTarget: inside } as unknown as FocusEvent,
      () => shell.closeMenu()
    );
    expect(shell.menuOpen()).toBeTrue();
    shell.closeOnFocusOut(
      { currentTarget: container, relatedTarget: null } as unknown as FocusEvent,
      () => shell.closeMenu()
    );
    expect(shell.menuOpen()).toBeFalse();
  });

  it('opens the sidebar from the burger and closes it on navigation', async () => {
    await render();
    query('burger')!.click();
    await settle(fixture);
    expect(el.querySelector('.shell')!.classList).toContain('sidebar-open');
    await TestBed.inject(Router).navigateByUrl('/conversations');
    await settle(fixture);
    expect(el.querySelector('.shell')!.classList).not.toContain('sidebar-open');
  });
});
