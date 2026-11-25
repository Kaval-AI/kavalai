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
import { ProjectService } from './project-service';
import { Project } from '../models/project';

describe('ProjectService', () => {
  let service: ProjectService;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      imports: [HttpClientTestingModule],
      providers: [ProjectService]
    });
    service = TestBed.inject(ProjectService);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    httpMock.verify();
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });

  it('should create a project', () => {
    const mockProject = { name: 'New Project' } as Project;
    service.create(mockProject).subscribe(project => {
      expect(project).toEqual(mockProject);
    });
    const req = httpMock.expectOne('/api/projects/create');
    expect(req.request.method).toBe('POST');
    req.flush(mockProject);
  });

  it('should get project by id', () => {
    const mockProject = { id: '1', name: 'Project 1' } as Project;
    service.getById('1').subscribe(project => {
      expect(project).toEqual(mockProject);
    });
    const req = httpMock.expectOne('/api/projects/get/1');
    expect(req.request.method).toBe('GET');
    req.flush(mockProject);
  });

  it('should get all projects', () => {
    const mockProjects = [{ id: '1', name: 'Project 1' }] as Project[];
    service.getAll().subscribe(projects => {
      expect(projects).toEqual(mockProjects);
    });
    const req = httpMock.expectOne('/api/projects/all');
    expect(req.request.method).toBe('GET');
    req.flush(mockProjects);
  });

  it('should update a project', () => {
    const mockProject = { id: '1', name: 'Updated' } as Project;
    service.update('1', mockProject).subscribe(project => {
      expect(project).toEqual(mockProject);
    });
    const req = httpMock.expectOne('/api/projects/update/1');
    expect(req.request.method).toBe('PUT');
    req.flush(mockProject);
  });

  it('should delete a project', () => {
    service.delete('1').subscribe();
    const req = httpMock.expectOne('/api/projects/delete/1');
    expect(req.request.method).toBe('DELETE');
    req.flush(null);
  });

  it('announces a created, updated or deleted project, and nothing else', () => {
    let count = 0;
    service.changes.subscribe(() => count++);

    service.getAll().subscribe();
    httpMock.expectOne('/api/projects/all').flush([]);
    service.create({ name: 'P' }).subscribe();
    httpMock.expectOne('/api/projects/create').flush({ id: '1', name: 'P' });
    service.update('1', { name: 'Q' }).subscribe();
    httpMock.expectOne('/api/projects/update/1').flush({ id: '1', name: 'Q' });
    service.delete('1').subscribe();
    httpMock.expectOne('/api/projects/delete/1').flush(null);
    expect(count).toBe(3);

    service.update('1', { name: 'R' }).subscribe({ error: () => undefined });
    httpMock
      .expectOne('/api/projects/update/1')
      .flush({ detail: 'No' }, { status: 403, statusText: 'Forbidden' });
    expect(count).toBe(3);
  });

  it('lists, adds, changes and removes the members of a project', () => {
    service.getMembers('p1').subscribe();
    httpMock.expectOne('/api/projects/p1/members').flush([]);

    service.addMember('p1', 'u2', 'viewer').subscribe();
    const add = httpMock.expectOne('/api/projects/p1/members/add');
    expect(add.request.method).toBe('POST');
    expect(add.request.body).toEqual({ user_id: 'u2', role: 'viewer' });
    add.flush({});

    service.updateMember('p1', 'u2', 'owner').subscribe();
    const update = httpMock.expectOne('/api/projects/p1/members/update');
    expect(update.request.method).toBe('PUT');
    expect(update.request.body).toEqual({ user_id: 'u2', role: 'owner' });
    update.flush({});

    service.removeMember('p1', 'u2').subscribe();
    const remove = httpMock.expectOne('/api/projects/p1/members/remove/u2');
    expect(remove.request.method).toBe('DELETE');
    remove.flush({});
  });

  it('should test connection', () => {
    const mockRes = { status: 'success' };
    service.testConnection('1').subscribe(res => {
      expect(res).toEqual(mockRes);
    });
    const req = httpMock.expectOne('/api/projects/test-connection/1');
    expect(req.request.method).toBe('POST');
    req.flush(mockRes);
  });

  it('should ask for the password with a POST', () => {
    service.revealPassword('1').subscribe(res => {
      expect(res).toEqual({ db_password: 'pw' });
    });
    const req = httpMock.expectOne('/api/projects/1/reveal-password');
    expect(req.request.method).toBe('POST');
    req.flush({ db_password: 'pw' });
  });

  it('should handle getById with number id', () => {
    const mockProject = { id: '1', name: 'Project 1' } as Project;
    service.getById(1).subscribe(project => {
      expect(project).toEqual(mockProject);
    });
    const req = httpMock.expectOne('/api/projects/get/1');
    expect(req.request.method).toBe('GET');
    req.flush(mockProject);
  });

  it('should handle delete with number id', () => {
    service.delete(1).subscribe();
    const req = httpMock.expectOne('/api/projects/delete/1');
    expect(req.request.method).toBe('DELETE');
    req.flush(null);
  });
});
