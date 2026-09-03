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
import { Component, OnInit, inject } from '@angular/core';
import { FormBuilder, FormsModule, ReactiveFormsModule, Validators } from '@angular/forms';
import { CommonModule } from '@angular/common';
import { ActivatedRoute, Router } from '@angular/router';
import { ProjectService } from '../../services/project-service';
import { Project } from '../../models/project';
import { UserService } from '../../services/user-service';
import { UserDetails } from '../../models/user-details';

@Component({
  selector: 'app-project-edit-page',
  standalone: true,
  imports: [FormsModule, ReactiveFormsModule, CommonModule],
  templateUrl: './project-edit-page.html',
  styleUrl: './project-edit-page.css',
})
export class ProjectEditPage implements OnInit {
  private projectService = inject(ProjectService);
  private userService = inject(UserService);
  private fb = inject(FormBuilder);
  private route = inject(ActivatedRoute);
  private router = inject(Router);

  projectId: string | null = null;
  project: Project | null = null;
  connectionStatus: { status: string; message?: string } | null = null;
  isTestingConnection = false;

  members: any[] = [];
  allUsers: UserDetails[] = [];
  selectedUserId: string = '';
  selectedRole: string = 'viewer';
  canManageMembers = false;
  activeRoleDropdown: string | null = null;

  get isSqlite() {
    return this.projectForm.controls.db_type.value === 'sqlite';
  }

  get filteredUsers() {
    const memberIds = this.members.map(m => m.id);
    return this.allUsers.filter(u => !memberIds.includes(u.id));
  }

  projectForm = this.fb.group({
    name: ['', [Validators.required]],
    description: ['', []],
    db_type: ['postgresql', []],
    db_host: ['', []],
    db_port: [5432, [Validators.min(1), Validators.max(65535)]],
    db_user: ['', []],
    db_password: ['', []],
    db_name: ['', []],
    db_schema: ['public', []]
  });

  ngOnInit() {
    this.projectId = this.route.snapshot.paramMap.get('id');
    if (this.projectId && this.projectId !== 'new') {
      this.loadProject(this.projectId);
      this.loadMembers(this.projectId);
      this.checkPermissions();
    }
    if (this.userService.getIsAdmin()) {
      this.loadAllUsers();
    }
  }

  checkPermissions() {
    const user = this.userService.getUserDetailsValue();
    if (!user) return;
    if (user.is_admin) {
      this.canManageMembers = true;
      return;
    }
    // If not admin, check if owner of this project
    this.projectService.getAll().subscribe(projects => {
      const proj = projects.find(p => p.id === this.projectId);
      if (proj && proj.role === 'owner') {
        this.canManageMembers = true;
        this.loadAllUsers(); // Owners also need to see users to add them
      }
    });
  }

  loadAllUsers() {
    this.userService.getUsers().subscribe(users => {
      this.allUsers = users;
    });
  }

  loadMembers(id: string) {
    this.projectService.getMembers(id).subscribe(members => {
      this.members = members;
    });
  }

  addMember() {
    if (!this.projectId || !this.selectedUserId) return;
    this.projectService.addMember(this.projectId, this.selectedUserId, this.selectedRole).subscribe(() => {
      this.loadMembers(this.projectId!);
      this.selectedUserId = '';
    });
  }

  toggleRoleDropdown(userId: string) {
    if (this.activeRoleDropdown === userId) {
      this.activeRoleDropdown = null;
    } else {
      this.activeRoleDropdown = userId;
    }
  }

  changeRole(member: any, newRole: string) {
    this.activeRoleDropdown = null;
    if (member.role === newRole) return;

    const userId = member.id;
    const currentUser = this.userService.getUserDetailsValue();

    if (member.role === 'owner' && newRole !== 'owner' && currentUser && currentUser.id === userId) {
      if (!confirm('You are about to demote yourself from owner. You may lose administrative access to this project. Are you sure?')) {
        return;
      }
    }

    if (!this.projectId) return;
    this.projectService.updateMember(this.projectId, userId, newRole).subscribe({
      next: () => {
        this.loadMembers(this.projectId!);
      },
      error: (err) => {
        alert(err.error?.detail || 'Failed to update member role');
        this.loadMembers(this.projectId!);
      }
    });
  }

  removeMember(userId: string) {
    if (!this.projectId) return;
    const currentUser = this.userService.getUserDetailsValue();
    let message = 'Are you sure you want to remove this member?';

    if (currentUser && currentUser.id === userId) {
      message = 'Are you sure you want to remove YOURSELF from this project? You will lose access to it.';
    }

    if (confirm(message)) {
      this.projectService.removeMember(this.projectId, userId).subscribe({
        next: () => {
          if (currentUser && currentUser.id === userId) {
            this.router.navigate(['/']);
          } else {
            this.loadMembers(this.projectId!);
          }
        },
        error: (err) => {
          alert(err.error?.detail || 'Failed to remove member');
        }
      });
    }
  }

  loadProject(id: string) {
    this.projectService.getAll().subscribe((projects) => {
      const p = projects.find(proj => proj.id === id);
      if (p) {
        this.project = p;
        this.projectForm.patchValue(p);
        this.projectForm.markAsPristine();
      }
    });
  }

  saveProject() {
    if (this.projectForm.invalid) return;

    const formValue = this.projectForm.value as Partial<Project>;

    if (this.projectId && this.projectId !== 'new') {
      this.projectService.update(this.projectId, formValue).subscribe(() => {
        this.router.navigate(['/']);
      });
    } else {
      this.projectService.create(formValue).subscribe((newProj: Project) => {
        this.userService.setActiveProject(newProj.id);
        this.router.navigate(['/']);
      });
    }
  }

  cancel() {
    this.router.navigate(['/']);
  }

  testConnection() {
    if (!this.projectId) return;

    this.isTestingConnection = true;
    this.connectionStatus = null;

    const data = this.projectId === 'new' ? this.projectForm.value : {};

    this.projectService.testConnection(this.projectId, data).subscribe({
      next: (res) => {
        this.connectionStatus = res;
        this.isTestingConnection = false;
      },
      error: (err) => {
        this.connectionStatus = {
          status: 'error',
          message: err.error?.message || err.message || 'Unknown error'
        };
        this.isTestingConnection = false;
      }
    });
  }
}
