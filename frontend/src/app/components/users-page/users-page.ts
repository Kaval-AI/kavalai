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
import { Component, OnInit, computed, inject, signal } from '@angular/core';
import { toSignal } from '@angular/core/rxjs-interop';
import { NonNullableFormBuilder, ReactiveFormsModule, Validators } from '@angular/forms';
import { firstValueFrom } from 'rxjs';
import { errorMessage } from '../../core/http-error';
import { MIN_PASSWORD_LENGTH, UserDetails, UserFields } from '../../models/user-details';
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

/**
 * The backoffice's users, for administrators: one dialog creates a user or
 * edits one, a second confirms a deletion. An administrator cannot delete
 * their own account, so that control is disabled with the reason.
 */
@Component({
  selector: 'app-users-page',
  imports: [
    ReactiveFormsModule,
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
  templateUrl: './users-page.html',
  styleUrl: './users-page.css',
})
export class UsersPage implements OnInit {
  private readonly userService = inject(UserService);
  private readonly toast = inject(ToastService);
  private readonly fb = inject(NonNullableFormBuilder);

  readonly minPasswordLength = MIN_PASSWORD_LENGTH;
  readonly users = signal<UserDetails[]>([]);
  readonly loading = signal(true);
  readonly loadError = signal('');

  /** The form's subject: `'new'` for a new user, the user being edited, or `null` when closed. */
  readonly editing = signal<UserDetails | 'new' | null>(null);
  readonly isNew = computed(() => this.editing() === 'new');
  readonly saving = signal(false);
  readonly saveError = signal('');

  /** The user whose deletion the dialog asks to confirm. */
  readonly removing = signal<UserDetails | null>(null);
  readonly deleting = signal(false);
  readonly deleteError = signal('');

  private readonly me = toSignal(this.userService.userDetails, { initialValue: null });
  readonly myId = computed(() => this.me()?.id ?? null);

  readonly form = this.fb.group({
    email: ['', [Validators.required, Validators.email]],
    name: ['', [Validators.required]],
    picture: [''],
    is_admin: [false],
    password: ['', [Validators.minLength(MIN_PASSWORD_LENGTH)]],
  });

  ngOnInit(): void {
    void this.load();
  }

  async load(): Promise<void> {
    this.loading.set(true);
    this.loadError.set('');
    try {
      this.users.set(await firstValueFrom(this.userService.getUsers()));
    } catch (err) {
      this.loadError.set(errorMessage(err, 'Could not load the users.'));
    } finally {
      this.loading.set(false);
    }
  }

  openNew(): void {
    this.form.reset();
    this.saveError.set('');
    this.editing.set('new');
  }

  openEdit(user: UserDetails): void {
    this.form.reset({
      email: user.email,
      name: user.name,
      picture: user.picture ?? '',
      is_admin: user.is_admin,
      password: '',
    });
    this.saveError.set('');
    this.editing.set(user);
  }

  closeForm(): void {
    this.editing.set(null);
  }

  /** The message under a field once it has been touched, or `''`. */
  fieldError(name: 'email' | 'name' | 'password'): string {
    const control = this.form.controls[name];
    if (!control.touched || control.valid) return '';
    if (control.hasError('required')) return name === 'email' ? 'The email is required.' : 'The name is required.';
    if (control.hasError('email')) return 'This is not an email address.';
    return `At least ${MIN_PASSWORD_LENGTH} characters.`;
  }

  async save(): Promise<void> {
    const subject = this.editing();
    if (!subject) return;
    if (this.form.invalid) {
      this.form.markAllAsTouched();
      return;
    }
    const value = this.form.getRawValue();
    const fields: UserFields = {
      email: value.email.trim(),
      name: value.name.trim(),
      picture: value.picture.trim() || null,
      is_admin: value.is_admin,
    };
    if (value.password) fields.password = value.password;

    this.saving.set(true);
    this.saveError.set('');
    try {
      if (subject === 'new') {
        await firstValueFrom(this.userService.createUser(fields));
        this.toast.success('User added');
      } else {
        await firstValueFrom(this.userService.updateUser(subject.id, fields));
        this.toast.success('Saved');
        if (subject.id === this.myId()) this.userService.updateUserDetails();
      }
      this.editing.set(null);
      await this.load();
    } catch (err) {
      this.saveError.set(errorMessage(err, 'Could not save the user.'));
    } finally {
      this.saving.set(false);
    }
  }

  askDelete(user: UserDetails): void {
    this.deleteError.set('');
    this.removing.set(user);
  }

  async confirmDelete(): Promise<void> {
    const user = this.removing();
    if (!user) return;
    this.deleting.set(true);
    this.deleteError.set('');
    try {
      await firstValueFrom(this.userService.deleteUser(user.id));
      this.removing.set(null);
      this.users.update((users) => users.filter((u) => u.id !== user.id));
      this.toast.success('User deleted');
    } catch (err) {
      this.deleteError.set(errorMessage(err, 'Could not delete the user.'));
    } finally {
      this.deleting.set(false);
    }
  }
}
