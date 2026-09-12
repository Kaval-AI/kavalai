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
import { ReactiveFormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';
import { of, throwError } from 'rxjs';
import { ActivatedRoute, Router } from '@angular/router';
import { ProjectEditPage } from './project-edit-page';
import { ProjectService } from '../../services/project-service';
import { UserService } from '../../services/user-service';
import { Project } from '../../models/project';

describe('ProjectEditPage', () => {
  let component: ProjectEditPage;
  let fixture: ComponentFixture<ProjectEditPage>;
  let projectServiceSpy: jasmine.SpyObj<ProjectService>;
  let userServiceSpy: jasmine.SpyObj<UserService>;
  let routerSpy: jasmine.SpyObj<Router>;
  let activatedRouteMock: any;

  beforeEach(async () => {
    projectServiceSpy = jasmine.createSpyObj('ProjectService', ['getAll', 'create', 'update', 'testConnection', 'getMembers']);
    userServiceSpy = jasmine.createSpyObj('UserService', ['setActiveProject', 'getIsAdmin', 'getUserDetailsValue']);
    routerSpy = jasmine.createSpyObj('Router', ['navigate']);
    activatedRouteMock = {
      snapshot: {
        paramMap: {
          get: jasmine.createSpy('get').and.returnValue('new')
        }
      }
    };

    projectServiceSpy.getAll.and.returnValue(of([]));
    userServiceSpy.getIsAdmin.and.returnValue(false);
    userServiceSpy.getUserDetailsValue.and.returnValue(null);
    projectServiceSpy.getMembers.and.returnValue(of([]));

    await TestBed.configureTestingModule({
      imports: [ProjectEditPage, ReactiveFormsModule, CommonModule],
      providers: [
        { provide: ProjectService, useValue: projectServiceSpy },
        { provide: UserService, useValue: userServiceSpy },
        { provide: Router, useValue: routerSpy },
        { provide: ActivatedRoute, useValue: activatedRouteMock }
      ]
    }).compileComponents();

    fixture = TestBed.createComponent(ProjectEditPage);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should load project when id is provided', () => {
    const mockProject = { id: '1', name: 'Test Project' } as Project;
    activatedRouteMock.snapshot.paramMap.get.and.returnValue('1');
    projectServiceSpy.getAll.and.returnValue(of([mockProject]));

    component.ngOnInit();

    expect(projectServiceSpy.getAll).toHaveBeenCalled();
    expect(component.projectForm.get('name')?.value).toBe('Test Project');
  });

  it('should save new project and navigate home', () => {
    const newProject = { id: '2', name: 'New Project' } as Project;
    component.projectId = 'new';
    component.projectForm.patchValue({ name: 'New Project' });
    projectServiceSpy.create.and.returnValue(of(newProject));

    component.saveProject();

    expect(projectServiceSpy.create).toHaveBeenCalled();
    expect(userServiceSpy.setActiveProject).toHaveBeenCalledWith('2');
    expect(routerSpy.navigate).toHaveBeenCalledWith(['/']);
  });

  it('should update existing project and navigate home', () => {
    component.projectId = '1';
    component.projectForm.patchValue({ name: 'Updated Name' });
    projectServiceSpy.update.and.returnValue(of({ id: '1', name: 'Updated Name' } as Project));

    component.saveProject();

    expect(projectServiceSpy.update).toHaveBeenCalledWith('1', jasmine.any(Object));
    expect(routerSpy.navigate).toHaveBeenCalledWith(['/']);
  });

  it('should navigate home on cancel', () => {
    component.cancel();
    expect(routerSpy.navigate).toHaveBeenCalledWith(['/']);
  });

  it('should default to PostgreSQL and switch the form to a file for SQLite', () => {
    expect(component.projectForm.value.db_type).toBe('postgresql');
    expect(component.isSqlite).toBeFalse();

    component.projectForm.patchValue({ db_type: 'sqlite', db_name: '/data/agents.db' });

    expect(component.isSqlite).toBeTrue();
    expect(component.projectForm.value.db_name).toBe('/data/agents.db');
  });

  it('should test connection for existing project', () => {
    component.projectId = '1';
    const mockRes = { status: 'success', message: 'Connected' };
    projectServiceSpy.testConnection.and.returnValue(of(mockRes));

    component.testConnection();

    expect(projectServiceSpy.testConnection).toHaveBeenCalledWith('1', {});
    expect(component.connectionStatus).toEqual(mockRes);
  });

  it('should test connection for new project', () => {
    component.projectId = 'new';
    component.projectForm.patchValue({ name: 'New Project', db_host: 'localhost' });
    const mockRes = { status: 'success', message: 'Connected' };
    projectServiceSpy.testConnection.and.returnValue(of(mockRes));

    component.testConnection();

    expect(projectServiceSpy.testConnection).toHaveBeenCalledWith('new', component.projectForm.value);
    expect(component.connectionStatus).toEqual(mockRes);
  });

  it('should handle test connection error', () => {
    component.projectId = '1';
    const errorResponse = { error: { message: 'Failed' } };
    projectServiceSpy.testConnection.and.returnValue(throwError(() => errorResponse));

    component.testConnection();

    expect(component.connectionStatus?.status).toBe('error');
    expect(component.connectionStatus?.message).toBe('Failed');
  });

  it('should NOT disable save button when form is pristine but valid', () => {
    const mockProject = { id: '1', name: 'Test Project' } as Project;
    activatedRouteMock.snapshot.paramMap.get.and.returnValue('1');
    projectServiceSpy.getAll.and.returnValue(of([mockProject]));

    component.ngOnInit();
    fixture.detectChanges();

    const saveButton = fixture.nativeElement.querySelector('button[type="submit"]');
    expect(saveButton.disabled).toBeFalse();

    component.projectForm.patchValue({ name: '' }); // make it invalid
    fixture.detectChanges();

    expect(saveButton.disabled).toBeTrue();
  });
});
