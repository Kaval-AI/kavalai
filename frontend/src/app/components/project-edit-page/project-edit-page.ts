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
import { HttpErrorResponse } from '@angular/common/http';
import { Component, DestroyRef, OnInit, WritableSignal, computed, inject, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import {
  AbstractControl,
  FormBuilder,
  FormsModule,
  ReactiveFormsModule,
  ValidationErrors,
  Validators,
} from '@angular/forms';
import { Router, RouterLink } from '@angular/router';
import { Observable } from 'rxjs';
import { distinctUntilChanged, map } from 'rxjs/operators';
import { errorMessage } from '../../core/http-error';
import { Project } from '../../models/project';
import { UserDetails } from '../../models/user-details';
import { ProjectService } from '../../services/project-service';
import { ToastService } from '../../services/toast-service';
import { UserService } from '../../services/user-service';
import {
  UiAlert,
  UiAvatar,
  UiBadge,
  UiButton,
  UiCard,
  UiDialog,
  UiEmptyState,
  UiField,
  UiLoading,
  UiPageHeader,
  UiSpinner,
  UiTable,
} from '../../ui';
import { activeProjectOf } from '../projects-page/projects-page';

export type ProjectRole = 'owner' | 'viewer';

/** A project member as `GET /projects/{id}/members` lists them. */
export interface ProjectMember {
  id: string;
  name: string;
  email: string;
  picture: string | null;
  role: ProjectRole;
}

/** The fields naming the server and account a stored password belongs to (`CONNECTION_IDENTITY`). */
const CONNECTION_IDENTITY = ['db_type', 'db_host', 'db_port', 'db_user', 'db_name'] as const;

/** A control's value without its surrounding blanks is not empty. */
function notBlank(control: AbstractControl): ValidationErrors | null {
  return String(control.value ?? '').trim() ? null : { required: true };
}

/** The result of Test connection, worded as the server worded it. */
export interface ConnectionResult {
  ok: boolean;
  message: string;
}

/**
 * The settings of the active project (`/settings`), or a new project
 * (`/projects/new`), one card per concern, each saving itself: General,
 * Database (with Test connection), Access, Members and the danger zone. A new
 * project has General, Database and Access and one Create button.
 *
 * Permissions are shown, not only enforced: owners change the settings and
 * delete the project; administrators and owners manage members; viewers see
 * the settings read-only; only administrators create projects. The server
 * checks every one of them.
 */
@Component({
  selector: 'app-project-edit-page',
  imports: [
    FormsModule,
    ReactiveFormsModule,
    RouterLink,
    UiAlert,
    UiAvatar,
    UiBadge,
    UiButton,
    UiCard,
    UiDialog,
    UiEmptyState,
    UiField,
    UiLoading,
    UiPageHeader,
    UiSpinner,
    UiTable,
  ],
  templateUrl: './project-edit-page.html',
  styleUrl: './project-edit-page.css',
})
export class ProjectEditPage implements OnInit {
  private readonly projectService = inject(ProjectService);
  private readonly userService = inject(UserService);
  private readonly toast = inject(ToastService);
  private readonly router = inject(Router);
  private readonly fb = inject(FormBuilder);
  private readonly destroyRef = inject(DestroyRef);

  /** `/projects/new` creates a project; `/settings` edits the active one. */
  readonly creating = this.router.url.split('?')[0].startsWith('/projects/new');

  readonly isAdmin = signal(false);
  readonly userId = signal<string | null>(null);

  readonly loading = signal(true);
  readonly loadError = signal('');
  readonly project = signal<Project | null>(null);
  readonly hasProjects = signal(false);
  private activeId = '';

  readonly canEdit = computed(() => this.creating || this.project()?.role === 'owner');
  readonly canManageMembers = computed(() => this.isAdmin() || this.project()?.role === 'owner');

  readonly general = this.fb.nonNullable.group({
    name: ['', [notBlank]],
    description: [''],
  });

  readonly db = this.fb.nonNullable.group({
    db_type: ['postgresql' as 'postgresql' | 'sqlite'],
    db_host: ['localhost'],
    db_port: [5432, [Validators.min(1), Validators.max(65535)]],
    db_user: [''],
    db_password: [''],
    db_name: ['', [notBlank]],
    db_schema: ['public'],
    rag_schema: [''],
  });

  readonly access = this.fb.nonNullable.group({ read_only: [true] });

  readonly isSqlite = signal(false);

  /** Which card is saving: `general`, `database`, `access` or `create`. */
  readonly saving = signal<string | null>(null);
  readonly generalError = signal('');
  readonly dbError = signal('');
  readonly accessError = signal('');
  readonly createError = signal('');

  readonly testing = signal(false);
  readonly testResult = signal<ConnectionResult | null>(null);

  readonly members = signal<ProjectMember[]>([]);
  readonly membersLoading = signal(false);
  readonly membersError = signal('');
  /** What a refused member change said. */
  readonly memberError = signal('');
  /** The member being changed, or `add`. */
  readonly memberBusy = signal<string | null>(null);
  /** The member whose Remove waits for a second click. */
  readonly confirmingRemove = signal<string | null>(null);
  /** The user's own demotion to viewer, waiting for the dialog. */
  readonly demotingSelf = signal<ProjectMember | null>(null);

  readonly users = signal<UserDetails[]>([]);
  /** Why the users to add cannot be listed, if they cannot. */
  readonly usersError = signal('');
  readonly usersForbidden = signal(false);
  newMemberId = '';
  newMemberRole: ProjectRole = 'viewer';

  readonly ownerCount = computed(() => this.members().filter((m) => m.role === 'owner').length);
  readonly candidates = computed(() => {
    const ids = new Set(this.members().map((m) => m.id));
    return this.users().filter((u) => !ids.has(u.id));
  });

  readonly deleteOpen = signal(false);
  readonly deleteName = signal('');
  readonly deleting = signal(false);
  readonly deleteError = signal('');

  constructor() {
    this.db.controls.db_type.valueChanges
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe((type) => this.isSqlite.set(type === 'sqlite'));
  }

  ngOnInit(): void {
    this.userService.userDetails
      .pipe(
        map((user) => (user ? activeProjectOf(user) : null)),
        distinctUntilChanged(),
        takeUntilDestroyed(this.destroyRef)
      )
      .subscribe((activeId) => {
        if (activeId === null) return;
        const user = this.userService.getUserDetailsValue();
        this.isAdmin.set(this.userService.getIsAdmin());
        this.userId.set(user?.id ?? null);
        this.activeId = activeId;
        if (this.creating) this.loading.set(false);
        else this.load();
      });
  }

  /** The user's projects, of which the active one is edited. */
  load(): void {
    this.loading.set(true);
    this.loadError.set('');
    this.projectService.getAll().subscribe({
      next: (projects) => {
        this.hasProjects.set(projects.length > 0);
        this.loading.set(false);
        this.show(projects.find((p) => p.id === this.activeId) ?? null);
      },
      error: (err) => {
        this.loading.set(false);
        this.loadError.set(errorMessage(err, 'The project could not be loaded.'));
      },
    });
  }

  /** Fill the cards from `project`; viewers get them disabled. */
  private show(project: Project | null): void {
    this.project.set(project);
    this.testResult.set(null);
    this.generalError.set('');
    this.dbError.set('');
    this.accessError.set('');
    this.confirmingRemove.set(null);
    this.memberError.set('');
    if (!project) return;
    this.general.reset({ name: project.name, description: project.description ?? '' });
    this.db.reset({
      db_type: project.db_type ?? 'postgresql',
      db_host: project.db_host ?? '',
      db_port: project.db_port ?? 5432,
      db_user: project.db_user ?? '',
      db_password: '',
      db_name: project.db_name ?? '',
      db_schema: project.db_schema ?? '',
      rag_schema: project.rag_schema ?? '',
    });
    this.access.reset({ read_only: project.read_only !== false });
    for (const form of [this.general, this.db, this.access]) {
      if (this.canEdit()) form.enable();
      else form.disable();
    }
    this.loadMembers();
    if (this.canManageMembers()) this.loadUsers();
  }

  /** Whether the typed settings name another server or account than the stored ones. */
  private identityChanged(): boolean {
    const project = this.project()!;
    const typed = this.db.getRawValue();
    return CONNECTION_IDENTITY.some((key) => String(typed[key] ?? '') !== String(project[key] ?? ''));
  }

  /**
   * The password is needed again when one is stored and the typed settings
   * point elsewhere: the server keeps a stored password only for the same
   * server and account, so it cannot be sent to another host.
   */
  passwordMissing(): boolean {
    const project = this.project();
    return (
      !!project?.has_password &&
      !this.isSqlite() &&
      !this.db.controls.db_password.value &&
      this.identityChanged()
    );
  }

  fieldError(control: AbstractControl, label: string): string {
    if (!control.touched || control.valid) return '';
    if (control.errors?.['required']) return `${label} is required.`;
    return `${label} is not valid.`;
  }

  /** The database settings as the API takes them; an empty password keeps the stored one. */
  private dbBody(): Partial<Project> {
    const value = this.db.getRawValue();
    const body: Partial<Project> = {
      db_type: value.db_type,
      db_name: value.db_name.trim(),
    };
    if (value.db_type === 'sqlite') return body;
    return {
      ...body,
      db_host: value.db_host.trim(),
      db_port: Number(value.db_port),
      db_user: value.db_user.trim(),
      db_schema: value.db_schema.trim(),
      rag_schema: value.rag_schema.trim() || undefined,
      ...(value.db_password ? { db_password: value.db_password } : {}),
    };
  }

  saveGeneral(): void {
    this.general.markAllAsTouched();
    if (this.general.invalid) return;
    const { name, description } = this.general.getRawValue();
    this.save('general', this.generalError, { name: name.trim(), description: description.trim() }, () =>
      this.general.markAsPristine()
    );
  }

  saveDatabase(): void {
    this.db.markAllAsTouched();
    if (this.db.invalid || this.passwordMissing()) return;
    this.save('database', this.dbError, this.dbBody(), () => {
      this.db.controls.db_password.setValue('');
      this.db.markAsPristine();
    });
  }

  /** The switch saves as soon as it is turned, and turns back when the server refuses. */
  saveAccess(): void {
    const readOnly = this.access.controls.read_only.value;
    this.save('access', this.accessError, { read_only: readOnly }, undefined, () =>
      this.access.controls.read_only.setValue(!readOnly)
    );
  }

  private save(
    card: string,
    error: WritableSignal<string>,
    body: Partial<Project>,
    done?: () => void,
    failed?: () => void
  ): void {
    const project = this.project()!;
    this.saving.set(card);
    error.set('');
    this.projectService.update(project.id, body).subscribe({
      next: (updated) => {
        this.saving.set(null);
        this.project.set({ ...project, ...updated, role: project.role });
        done?.();
        this.toast.success('Saved');
      },
      error: (err) => {
        this.saving.set(null);
        error.set(errorMessage(err, 'The change could not be saved.'));
        failed?.();
      },
    });
  }

  /**
   * Check the typed settings before they are saved. An owner sends the form
   * (an empty password means the stored one, which the server reuses only for
   * the same server and account); a viewer, who may not change them, tests
   * the saved settings.
   */
  testConnection(): void {
    let request: Observable<{ status: string; message?: string }>;
    if (this.creating) {
      this.db.markAllAsTouched();
      if (this.db.invalid) return;
      request = this.projectService.testConnection('new', this.dbBody());
    } else {
      const project = this.project()!;
      request = this.projectService.testConnection(project.id, this.canEdit() ? this.dbBody() : {});
    }
    this.testing.set(true);
    this.testResult.set(null);
    request.subscribe({
      next: (res) => {
        this.testing.set(false);
        const ok = res.status === 'success';
        this.testResult.set({
          ok,
          message: res.message || (ok ? 'The connection works.' : 'The connection failed.'),
        });
      },
      error: (err) => {
        this.testing.set(false);
        this.testResult.set({ ok: false, message: errorMessage(err, 'The connection failed.') });
      },
    });
  }

  /** Create the project from the three cards, make it the active one and open its overview. */
  create(): void {
    this.general.markAllAsTouched();
    this.db.markAllAsTouched();
    if (this.general.invalid || this.db.invalid) return;
    const { name, description } = this.general.getRawValue();
    this.saving.set('create');
    this.createError.set('');
    this.projectService
      .create({
        name: name.trim(),
        description: description.trim(),
        ...this.dbBody(),
        read_only: this.access.controls.read_only.value,
      })
      .subscribe({
        next: (created) => {
          this.saving.set(null);
          this.userService.setActiveProject(created.id);
          this.toast.success('Project created');
          this.router.navigate(['/']);
        },
        error: (err) => {
          this.saving.set(null);
          this.createError.set(errorMessage(err, 'The project could not be created.'));
        },
      });
  }

  loadMembers(): void {
    const project = this.project()!;
    this.membersLoading.set(true);
    this.membersError.set('');
    this.projectService.getMembers(project.id).subscribe({
      next: (members) => {
        this.members.set(members as ProjectMember[]);
        this.membersLoading.set(false);
      },
      error: (err) => {
        this.membersLoading.set(false);
        this.membersError.set(errorMessage(err, 'The members could not be loaded.'));
      },
    });
  }

  /** Every user, to choose whom to add; the API lists them to administrators only. */
  private loadUsers(): void {
    this.usersError.set('');
    this.usersForbidden.set(false);
    this.userService.getUsers().subscribe({
      next: (users) => this.users.set(users),
      error: (err) => {
        this.users.set([]);
        if (err instanceof HttpErrorResponse && err.status === 403) this.usersForbidden.set(true);
        else this.usersError.set(errorMessage(err, 'The users could not be loaded.'));
      },
    });
  }

  isLastOwner(member: ProjectMember): boolean {
    return member.role === 'owner' && this.ownerCount() <= 1;
  }

  /**
   * Change a member's role from its select. Making oneself a viewer takes the
   * owner's controls away, so it waits for the dialog; a refusal puts the
   * select back.
   */
  setRole(member: ProjectMember, select: HTMLSelectElement): void {
    const role = select.value as ProjectRole;
    if (role === member.role) return;
    if (member.id === this.userId() && role === 'viewer') {
      select.value = member.role;
      this.demotingSelf.set(member);
      return;
    }
    this.changeRole(member, role, () => (select.value = member.role));
  }

  confirmDemoteSelf(): void {
    const member = this.demotingSelf();
    this.demotingSelf.set(null);
    if (member) this.changeRole(member, 'viewer');
  }

  private changeRole(member: ProjectMember, role: ProjectRole, failed?: () => void): void {
    const project = this.project()!;
    this.runMember(
      member.id,
      this.projectService.updateMember(project.id, member.id, role),
      () => {
        this.members.update((list) => list.map((m) => (m.id === member.id ? { ...m, role } : m)));
        this.toast.success('Role changed');
        // The user's own role decides which cards they may change.
        if (member.id === this.userId()) this.show({ ...project, role });
      },
      failed
    );
  }

  /** The first click asks, the second removes; removing oneself leaves the project. */
  remove(member: ProjectMember): void {
    if (this.confirmingRemove() !== member.id) {
      this.memberError.set('');
      this.confirmingRemove.set(member.id);
      return;
    }
    this.confirmingRemove.set(null);
    const project = this.project()!;
    this.runMember(member.id, this.projectService.removeMember(project.id, member.id), () => {
      this.members.update((list) => list.filter((m) => m.id !== member.id));
      this.toast.success('Member removed');
      if (member.id === this.userId()) {
        this.userService.updateUserDetails();
        this.router.navigate(['/']);
      }
    });
  }

  cancelRemove(): void {
    this.confirmingRemove.set(null);
  }

  addMember(): void {
    const project = this.project()!;
    if (!this.newMemberId) return;
    const user = this.users().find((u) => u.id === this.newMemberId);
    const role = this.newMemberRole;
    this.runMember('add', this.projectService.addMember(project.id, this.newMemberId, role), () => {
      if (user) {
        this.members.update((list) => [
          ...list,
          { id: user.id, name: user.name, email: user.email, picture: user.picture || null, role },
        ]);
      }
      this.newMemberId = '';
      this.newMemberRole = 'viewer';
      this.toast.success('Member added');
    });
  }

  private runMember(key: string, request: Observable<unknown>, done: () => void, failed?: () => void): void {
    this.memberBusy.set(key);
    this.memberError.set('');
    request.subscribe({
      next: () => {
        this.memberBusy.set(null);
        done();
      },
      error: (err) => {
        this.memberBusy.set(null);
        this.memberError.set(errorMessage(err, 'The change was refused.'));
        failed?.();
      },
    });
  }

  openDelete(): void {
    this.deleteName.set('');
    this.deleteError.set('');
    this.deleteOpen.set(true);
  }

  closeDelete(): void {
    if (!this.deleting()) this.deleteOpen.set(false);
  }

  /**
   * Delete the project once its name is typed, then make another of the
   * user's projects active (the server clears the active project when none is
   * left) and go to the overview.
   */
  deleteProject(): void {
    const project = this.project()!;
    if (this.deleteName().trim() !== project.name) return;
    this.deleting.set(true);
    this.deleteError.set('');
    this.projectService.delete(project.id).subscribe({
      next: () => {
        this.deleting.set(false);
        this.deleteOpen.set(false);
        this.toast.success('Project deleted');
        this.projectService.getAll().subscribe({
          next: (projects) => this.afterDelete(projects.find((p) => p.id !== project.id) ?? null),
          error: () => this.afterDelete(null),
        });
      },
      error: (err) => {
        this.deleting.set(false);
        this.deleteError.set(errorMessage(err, 'The project could not be deleted.'));
      },
    });
  }

  private afterDelete(next: Project | null): void {
    if (next) this.userService.setActiveProject(next.id);
    else this.userService.updateUserDetails();
    this.router.navigate(['/']);
  }

  displayName(member: ProjectMember): string {
    return member.name || member.email;
  }
}
