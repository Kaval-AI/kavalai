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

import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { BehaviorSubject, Observable } from 'rxjs';
import { tap } from 'rxjs/operators';
import { ProfileFields, UserDetails, UserFields } from '../models/user-details';

export interface AuthMethods {
  password: boolean
  google: boolean
}

@Injectable({
  providedIn: 'root',
})
export class UserService {
  private loggedIn: boolean = false;
  private userDetails$ = new BehaviorSubject<UserDetails | null>(null);

  public userDetails = this.userDetails$.asObservable();

  constructor(private http: HttpClient) {}

  login() {
    this.redirect('/api/login');
  }

  getAuthMethods(): Observable<AuthMethods> {
    return this.http.get<AuthMethods>('/api/auth/methods');
  }

  loginWithPassword(email: string, password: string): Observable<UserDetails> {
    return this.http.post<UserDetails>('/api/auth/login', { email, password }).pipe(
      tap((userDetails) => {
        this.loggedIn = true;
        this.userDetails$.next(userDetails);
      })
    );
  }

  private redirect(url: string) {
    window.location.href = url;
  }

  logout(): void {
    this.http.post('/api/logout', {}).subscribe({
      next: () => {
        this.loggedIn = false;
        this.userDetails$.next(null);
      },
      error: () => {
        this.loggedIn = false;
        this.userDetails$.next(null);
      },
    });
  }

  updateUserDetails(): void {
    this.http.get<UserDetails>('/api/user/get_details').subscribe({
      next: (user_details) => {
        this.loggedIn = true;
        this.userDetails$.next(user_details as UserDetails);
      },
      error: () => {
        this.loggedIn = false;
        this.userDetails$.next(null);
      },
    });
  }

  getIsLoggedIn(): boolean {
    return this.loggedIn;
  }

  getIsAdmin(): boolean {
    const user = this.userDetails$.value;
    return this.loggedIn && user != null && user.is_admin;
  }

  getUserDetailsValue(): UserDetails | null {
    return this.userDetails$.value;
  }

  getActiveProjectId(): string | null {
    const user = this.userDetails$.value;
    if (user && user.active_project_id && user.active_project_id !== 'None') {
      return user.active_project_id;
    }
    return null;
  }

  setActiveProject(projectId: string): void { // Changed from Observable<any> to void
    this.http.post<any>(`/api/user/set_active_project/${projectId}`, {}).subscribe({
      next: () => {
        const currentDetails = this.userDetails$.value;
        if (currentDetails) {
          this.userDetails$.next({
            ...currentDetails,
            active_project_id: projectId
          });
        }
      },
      error: (err) => {
        console.error('Failed to set active project', err);
      }
    });
  }

  getUsers(): Observable<UserDetails[]> {
    return this.http.get<UserDetails[]>('/api/users/all');
  }

  createUser(userData: UserFields): Observable<UserDetails> {
    return this.http.post<UserDetails>('/api/users/create', userData);
  }

  updateUser(userId: string, userData: Partial<UserFields>): Observable<UserDetails> {
    return this.http.put<UserDetails>(`/api/users/update/${userId}`, userData);
  }

  deleteUser(userId: string): Observable<any> {
    return this.http.delete(`/api/users/delete/${userId}`);
  }

  /** The signed-in user's own record, with `has_password`. */
  getMe(): Observable<UserDetails> {
    return this.http.get<UserDetails>('/api/user/me');
  }

  /** Saves the signed-in user's own name and picture. */
  updateProfile(fields: ProfileFields): Observable<UserDetails> {
    return this.http.patch<UserDetails>('/api/user/me', fields);
  }

  /**
   * Sets or changes the signed-in user's own password. `currentPassword` is
   * required when the user has one, and left out for a first password.
   */
  changePassword(newPassword: string, currentPassword?: string): Observable<UserDetails> {
    const body: { new_password: string; current_password?: string } = { new_password: newPassword };
    if (currentPassword !== undefined) body.current_password = currentPassword;
    return this.http.post<UserDetails>('/api/user/me/password', body);
  }
}
