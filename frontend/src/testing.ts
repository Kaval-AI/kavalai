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

/**
 * Shared spec helpers: `settle` to let a fixture finish its work, builders
 * for the records the API returns (each taking overrides), and a signed-in
 * `UserService` double per role. Specs import from here rather than writing
 * their fixtures inline; excluded from the application build.
 */
import { ComponentFixture } from '@angular/core/testing';
import { BehaviorSubject, of } from 'rxjs';
import { ChatMessage } from './app/models/chat-message';
import { LLMCallStat } from './app/models/llm-call-stat';
import { Project } from './app/models/project';
import { Run } from './app/models/run';
import { SessionSummary } from './app/models/session';
import { Task } from './app/models/task';
import { UserDetails } from './app/models/user-details';
import { UserService } from './app/services/user-service';

const CREATED = '2026-11-24T10:00:00Z';

/** Let pending promises and HTTP callbacks settle, then re-render. */
export async function settle(fixture: ComponentFixture<unknown>): Promise<void> {
  await new Promise((resolve) => setTimeout(resolve, 0));
  await fixture.whenStable();
  fixture.detectChanges();
}

export function makeUser(overrides: Partial<UserDetails> = {}): UserDetails {
  return {
    id: 'u1',
    email: 'ada@example.com',
    name: 'Ada Lovelace',
    is_admin: false,
    picture: '',
    active_project_id: 'p1',
    has_password: true,
    ...overrides,
  };
}

export function makeProject(overrides: Partial<Project> = {}): Project {
  return {
    id: 'p1',
    name: 'Pulse local',
    description: 'Runs recorded by the local Pulse API.',
    db_type: 'postgresql',
    db_host: 'localhost',
    db_port: 6543,
    db_user: 'kavalai_dev',
    db_name: 'kavalai_dev',
    db_schema: 'pulse',
    rag_schema: undefined,
    read_only: true,
    has_password: true,
    role: 'owner',
    created_at: CREATED,
    updated_at: CREATED,
    ...overrides,
  };
}

/** A project member as `GET /projects/{id}/members` lists them. */
export interface Member {
  id: string;
  name: string;
  email: string;
  picture: string | null;
  role: 'owner' | 'viewer';
}

export function makeMember(overrides: Partial<Member> = {}): Member {
  return {
    id: 'u1',
    name: 'Ada Lovelace',
    email: 'ada@example.com',
    picture: null,
    role: 'owner',
    ...overrides,
  };
}

export function makeSession(overrides: Partial<SessionSummary> = {}): SessionSummary {
  return {
    session_id: 's1',
    agent_id: 'a1',
    agent_name: 'pulse_answer',
    external_id: null,
    runs_count: 1,
    tasks_count: 3,
    messages_count: 2,
    errors_count: 0,
    first_message: 'How many new claims did we get last month?',
    last_message: '{"cards": []}',
    created_at: CREATED,
    updated_at: CREATED,
    ...overrides,
  };
}

export function makeRun(overrides: Partial<Run> = {}): Run {
  return {
    id: 'r1',
    session_id: 's1',
    input_data: { question: 'How many new claims did we get last month?' },
    output_data: { cards: [] },
    context: null,
    tasks_count: 3,
    duration_seconds: 4.2,
    created_at: CREATED,
    updated_at: CREATED,
    ...overrides,
  };
}

export function makeTask(overrides: Partial<Task> = {}): Task {
  return {
    id: 't1',
    agent_id: 'a1',
    session_id: 's1',
    run_id: 'r1',
    inputs: null,
    output: null,
    name: 'fetch',
    node_type: 'agent',
    prompt: null,
    errors: null,
    duration_seconds: 1.5,
    seq: 1,
    parent_task_name: null,
    tool_uri: null,
    created_at: CREATED,
    updated_at: CREATED,
    ...overrides,
  };
}

export function makeMessage(overrides: Partial<ChatMessage> = {}): ChatMessage {
  return {
    id: 'm1',
    agent_id: 'a1',
    session_id: 's1',
    run_id: 'r1',
    role: 'user',
    content: 'How many new claims did we get last month?',
    created_at: CREATED,
    updated_at: CREATED,
    ...overrides,
  };
}

export function makeModelCall(overrides: Partial<LLMCallStat> = {}): LLMCallStat {
  return {
    id: 'c1',
    call_type: 'llm',
    model: 'scaleway/glm-5.3',
    agent_id: 'a1',
    session_id: 's1',
    run_id: 'r1',
    response_code: 200,
    prompt_tokens: 861,
    completion_tokens: 189,
    total_tokens: 1050,
    cached_prompt_tokens: 0,
    reasoning_tokens: 31,
    duration_seconds: 2.489,
    request_data: { messages: [{ role: 'user', content: 'Hello' }] },
    response_data: { content: 'Hi' },
    created_at: CREATED,
    updated_at: CREATED,
    ...overrides,
  };
}

export type Role = 'admin' | 'owner' | 'viewer';

/**
 * A `UserService` double for a signed-in user. `admin` is an administrator;
 * `owner` and `viewer` are not, and differ only in the role their projects
 * carry (`makeProject({ role })`). `userDetails` emits the user, and the
 * spies answer from it.
 */
export function signedInAs(
  role: Role = 'owner',
  overrides: Partial<UserDetails> = {}
): jasmine.SpyObj<UserService> {
  const user = makeUser({ is_admin: role === 'admin', ...overrides });
  const details = new BehaviorSubject<UserDetails | null>(user);
  const spy = jasmine.createSpyObj<UserService>(
    'UserService',
    [
      'getIsAdmin',
      'getIsLoggedIn',
      'getUserDetailsValue',
      'getActiveProjectId',
      'setActiveProject',
      'updateUserDetails',
      'getUsers',
      'logout',
      'getAuthMethods',
      'loginWithPassword',
      'login',
      'createUser',
      'updateUser',
      'deleteUser',
      'getMe',
      'updateProfile',
      'changePassword',
    ],
    { userDetails: details.asObservable() }
  );
  spy.getIsAdmin.and.callFake(() => !!details.value?.is_admin);
  spy.getIsLoggedIn.and.callFake(() => details.value !== null);
  spy.getUserDetailsValue.and.callFake(() => details.value);
  spy.getActiveProjectId.and.callFake(() => details.value?.active_project_id ?? null);
  spy.getUsers.and.returnValue(of([user]));
  spy.getAuthMethods.and.returnValue(of({ password: true, google: false }));
  spy.getMe.and.callFake(() => of(details.value ?? user));
  return spy;
}
