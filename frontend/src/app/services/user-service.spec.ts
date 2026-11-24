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

import { TestBed } from '@angular/core/testing';
import { HttpClientTestingModule, HttpTestingController } from '@angular/common/http/testing';
import { UserService } from './user-service';
import { UserDetails } from '../models/user-details';

describe('UserService', () => {
  let service: UserService;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      imports: [HttpClientTestingModule],
      providers: [UserService]
    });
    service = TestBed.inject(UserService);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    httpMock.verify();
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });

  it('should update user details', () => {
    const mockUser: UserDetails = { id: 'user1', email: 'test@test.com', name: 'Test', picture: '', is_admin: true, active_project_id: 'proj1' };
    service.updateUserDetails();

    const req = httpMock.expectOne('/api/user/get_details');
    expect(req.request.method).toBe('GET');
    req.flush(mockUser);

    expect(service.getIsLoggedIn()).toBeTrue();
    expect(service.getIsAdmin()).toBeTrue();
    expect(service.getActiveProjectId()).toBe('proj1');
  });

  it('should handle logout', () => {
    service.logout();
    const req = httpMock.expectOne('/api/logout');
    expect(req.request.method).toBe('GET');
    req.flush({});

    expect(service.getIsLoggedIn()).toBeFalse();
  });

  it('should set active project', () => {
    const mockUser: UserDetails = { id: 'user1', email: 'test@test.com', name: 'Test', picture: '', is_admin: false, active_project_id: 'old' };

    // Setup initial state
    service.updateUserDetails();
    httpMock.expectOne('/api/user/get_details').flush(mockUser);

    service.setActiveProject('new');
    const req = httpMock.expectOne('/api/user/set_active_project/new');
    expect(req.request.method).toBe('POST');
    req.flush({});

    expect(service.getActiveProjectId()).toBe('new');
  });

  it('should handle login redirection', () => {
    const spy = spyOn(service as any, 'redirect');
    service.login();
    expect(spy).toHaveBeenCalledWith('/api/login');
  });

  it('should read the sign-in methods', () => {
    let methods: any;
    service.getAuthMethods().subscribe((m) => (methods = m));
    const req = httpMock.expectOne('/api/auth/methods');
    expect(req.request.method).toBe('GET');
    req.flush({ password: true, google: false });

    expect(methods).toEqual({ password: true, google: false });
  });

  it('should log in with a password', () => {
    const mockUser: UserDetails = { id: 'user1', email: 'test@test.com', name: 'Test', picture: '', is_admin: true, active_project_id: 'proj1' };
    service.loginWithPassword('test@test.com', 'correct horse').subscribe();

    const req = httpMock.expectOne('/api/auth/login');
    expect(req.request.method).toBe('POST');
    expect(req.request.body).toEqual({ email: 'test@test.com', password: 'correct horse' });
    req.flush(mockUser);

    expect(service.getIsLoggedIn()).toBeTrue();
    expect(service.getIsAdmin()).toBeTrue();
    expect(service.getUserDetailsValue()).toEqual(mockUser);
  });

  it('should stay logged out when the password is refused', () => {
    let failed = false;
    service.loginWithPassword('test@test.com', 'wrong').subscribe({ error: () => (failed = true) });
    httpMock.expectOne('/api/auth/login').flush({}, { status: 401, statusText: 'Unauthorized' });

    expect(failed).toBeTrue();
    expect(service.getIsLoggedIn()).toBeFalse();
  });

  it('should handle updateUserDetails error', () => {
    service.updateUserDetails();
    const req = httpMock.expectOne('/api/user/get_details');
    req.error(new ErrorEvent('Network error'));

    expect(service.getIsLoggedIn()).toBeFalse();
    expect(service.getActiveProjectId()).toBeNull();
  });

  it('should handle logout error', () => {
    service.logout();
    const req = httpMock.expectOne('/api/logout');
    req.error(new ErrorEvent('Network error'));

    expect(service.getIsLoggedIn()).toBeFalse();
  });

  it('should handle setActiveProject error', () => {
    const consoleSpy = spyOn(console, 'error');
    service.setActiveProject('new');
    const req = httpMock.expectOne('/api/user/set_active_project/new');
    req.error(new ErrorEvent('API error'));

    expect(consoleSpy).toHaveBeenCalled();
  });

  it('lists, creates, updates and deletes users', () => {
    const fields = { email: 'a@example.com', name: 'A', picture: null, is_admin: false, password: 'correct horse' };
    service.getUsers().subscribe();
    let req = httpMock.expectOne('/api/users/all');
    expect(req.request.method).toBe('GET');
    req.flush([]);

    service.createUser(fields).subscribe();
    req = httpMock.expectOne('/api/users/create');
    expect(req.request.method).toBe('POST');
    expect(req.request.body).toEqual(fields);
    req.flush({});

    service.updateUser('u2', { name: 'B' }).subscribe();
    req = httpMock.expectOne('/api/users/update/u2');
    expect(req.request.method).toBe('PUT');
    expect(req.request.body).toEqual({ name: 'B' });
    req.flush({});

    service.deleteUser('u2').subscribe();
    req = httpMock.expectOne('/api/users/delete/u2');
    expect(req.request.method).toBe('DELETE');
    req.flush({});
  });

  it('reads and saves the signed-in user\'s own profile', () => {
    service.getMe().subscribe();
    let req = httpMock.expectOne('/api/user/me');
    expect(req.request.method).toBe('GET');
    req.flush({});

    service.updateProfile({ name: 'Ada', picture: null }).subscribe();
    req = httpMock.expectOne('/api/user/me');
    expect(req.request.method).toBe('PATCH');
    expect(req.request.body).toEqual({ name: 'Ada', picture: null });
    req.flush({});
  });

  it('changes a password with the current one, and sets a first one without', () => {
    service.changePassword('battery staple', 'correct horse').subscribe();
    let req = httpMock.expectOne('/api/user/me/password');
    expect(req.request.method).toBe('POST');
    expect(req.request.body).toEqual({ new_password: 'battery staple', current_password: 'correct horse' });
    req.flush({});

    service.changePassword('battery staple').subscribe();
    req = httpMock.expectOne('/api/user/me/password');
    expect(req.request.body).toEqual({ new_password: 'battery staple' });
    req.flush({});
  });

  it('should return false for isAdmin when user is null', () => {
    expect(service.getIsAdmin()).toBeFalse();
  });

  it('should return false for isAdmin when not logged in', () => {
    // This is a bit tricky as loggedIn is private.
    // But initially it's false.
    expect(service.getIsAdmin()).toBeFalse();
  });
});
