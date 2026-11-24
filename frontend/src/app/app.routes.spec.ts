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
import { routes, pageTitle } from './app.routes';
import { UserEditPage } from './components/user-edit-page/user-edit-page';
import { UsersPage } from './components/users-page/users-page';

describe('routes', () => {
  it('gives every page a tab title and sends unknown paths to the overview', () => {
    const pages = routes.filter((route) => route.component);
    expect(pages.map((route) => route.path)).toEqual([
      '',
      'conversations',
      'conversations/:sessionId',
      'conversations/:sessionId/runs/:runId/tasks',
      'workflows',
      'llm-call-stats',
      'agents',
      'rag',
      'settings',
      'projects/new',
      'users',
      'profile',
    ]);
    expect(pages.every((route) => typeof route.title === 'string')).toBeTrue();
    expect(routes[routes.length - 1]).toEqual({ path: '**', redirectTo: '' });
  });

  it('sends the old project editor addresses to their new pages', () => {
    expect(routes.find((r) => r.path === 'project-edit/new')!.redirectTo).toBe('projects/new');
    expect(routes.find((r) => r.path === 'project-edit/:id')!.redirectTo).toBe('settings');
  });

  it('sends the old user editor to the users list and gives the profile its own page', () => {
    expect(routes.find((r) => r.path === 'user-edit/:id')!.redirectTo).toBe('users');
    expect(routes.find((r) => r.path === 'profile')!.component).toBe(UserEditPage);
    expect(routes.find((r) => r.path === 'users')!.component).toBe(UsersPage);
  });

  it('names the product after the page', () => {
    expect(pageTitle('Runs')).toBe('Runs · Kaval.AI backoffice');
  });
});
