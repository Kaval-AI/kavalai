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
import { Routes } from '@angular/router';
import { ConversationsPage } from './components/conversations-page/conversations-page';
import { WorkflowsPage } from './components/workflows-page/workflows-page';
import { LlmCallStatsPage } from './components/llm-call-stats-page/llm-call-stats-page';
import { RagPage } from './components/rag-page/rag-page';
import { SessionDetailPage } from './components/session-detail-page/session-detail-page';
import { RunTasksPage } from './components/run-tasks-page/run-tasks-page';

import { ProjectsPage } from './components/projects-page/projects-page';
import { AgentsPage } from './components/agents-page/agents-page';
import { ProjectEditPage } from './components/project-edit-page/project-edit-page';
import { UserEditPage } from './components/user-edit-page/user-edit-page';
import { UsersPage } from './components/users-page/users-page';

/** The browser tab's title for a page: the page, then the product. */
export function pageTitle(page: string): string {
  return `${page} · Kaval.AI backoffice`;
}

export const routes: Routes = [
  { path: '', component: ProjectsPage, title: pageTitle('Overview') },
  { path: 'conversations', component: ConversationsPage, title: pageTitle('Conversations') },
  { path: 'conversations/:sessionId', component: SessionDetailPage, title: pageTitle('Conversation') },
  {
    path: 'conversations/:sessionId/runs/:runId/tasks',
    component: RunTasksPage,
    title: pageTitle('Run'),
  },
  { path: 'workflows', component: WorkflowsPage, title: pageTitle('Runs') },
  { path: 'llm-call-stats', component: LlmCallStatsPage, title: pageTitle('Model calls') },
  { path: 'agents', component: AgentsPage, title: pageTitle('Agents') },
  { path: 'rag', component: RagPage, title: pageTitle('RAG') },
  { path: 'settings', component: ProjectEditPage, title: pageTitle('Settings') },
  { path: 'projects/new', component: ProjectEditPage, title: pageTitle('New project') },
  { path: 'users', component: UsersPage, title: pageTitle('Users') },
  { path: 'profile', component: UserEditPage, title: pageTitle('Profile') },
  { path: 'user-edit/:id', redirectTo: 'users' },
  { path: 'project-edit/new', redirectTo: 'projects/new' },
  { path: 'project-edit/:id', redirectTo: 'settings' },
  { path: '**', redirectTo: '' },
];
