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

import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, Subject } from 'rxjs';
import { tap } from 'rxjs/operators';
import { Project } from '../models/project';

@Injectable({
  providedIn: 'root',
})
export class ProjectService {
  private http = inject(HttpClient);
  private changed = new Subject<void>();

  /**
   * Emits after a project was created, updated or deleted here, so a list
   * shown elsewhere — the sidebar's project switcher — can reload.
   */
  readonly changes = this.changed.asObservable();

  create(project: Partial<Project>): Observable<Project> {
    return this.http
      .post<Project>('/api/projects/create', project)
      .pipe(tap(() => this.changed.next()));
  }

  getById(id: string | number): Observable<Project> {
    return this.http.get<Project>(`/api/projects/get/${id}`);
  }

  getAll(): Observable<Project[]> {
    return this.http.get<Project[]>("/api/projects/all");
  }

  update(id: string, project: Partial<Project>): Observable<Project> {
    return this.http
      .put<Project>(`/api/projects/update/${id}`, project)
      .pipe(tap(() => this.changed.next()));
  }

  delete(id: string | number): Observable<void> {
    return this.http
      .delete<void>(`/api/projects/delete/${id}`)
      .pipe(tap(() => this.changed.next()));
  }

  testConnection(projectId: string, data: any = {}): Observable<{status: string, message?: string}> {
    return this.http.post<{status: string, message?: string}>(`/api/projects/test-connection/${projectId}`, data);
  }

  /** The stored database password; administrators only, and logged by the server. */
  revealPassword(projectId: string): Observable<{ db_password: string | null }> {
    return this.http.post<{ db_password: string | null }>(
      `/api/projects/${projectId}/reveal-password`,
      {}
    );
  }

  getMembers(projectId: string): Observable<any[]> {
    return this.http.get<any[]>(`/api/projects/${projectId}/members`);
  }

  addMember(projectId: string, userId: string, role: string): Observable<any> {
    return this.http.post<any>(`/api/projects/${projectId}/members/add`, { user_id: userId, role });
  }

  updateMember(projectId: string, userId: string, role: string): Observable<any> {
    return this.http.put<any>(`/api/projects/${projectId}/members/update`, { user_id: userId, role });
  }

  removeMember(projectId: string, userId: string): Observable<any> {
    return this.http.delete<any>(`/api/projects/${projectId}/members/remove/${userId}`);
  }
}
