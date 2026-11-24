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
import { Component, OnInit, inject, signal } from '@angular/core';
import {
  AbstractControl,
  NonNullableFormBuilder,
  ReactiveFormsModule,
  ValidationErrors,
  Validators,
} from '@angular/forms';
import { firstValueFrom } from 'rxjs';
import { errorMessage } from '../../core/http-error';
import { MIN_PASSWORD_LENGTH, UserDetails } from '../../models/user-details';
import { ToastService } from '../../services/toast-service';
import { UserService } from '../../services/user-service';
import {
  UiAlert,
  UiAvatar,
  UiBadge,
  UiButton,
  UiCard,
  UiField,
  UiInfo,
  UiLoading,
  UiPageHeader,
  UiSpinner,
} from '../../ui';

/** The password form's group check: the repeated password must be the new one. */
function passwordsMatch(group: AbstractControl): ValidationErrors | null {
  const { new_password, repeat_password } = group.value as { new_password: string; repeat_password: string };
  return new_password === repeat_password ? null : { mismatch: true };
}

/**
 * The signed-in user's own profile, open to every user: the name and the
 * picture in one card, the password in another. A user who signs in through
 * Google only has no current password to give and sets a first one.
 */
@Component({
  selector: 'app-user-edit-page',
  imports: [
    ReactiveFormsModule,
    UiAlert,
    UiAvatar,
    UiBadge,
    UiButton,
    UiCard,
    UiField,
    UiInfo,
    UiLoading,
    UiPageHeader,
    UiSpinner,
  ],
  templateUrl: './user-edit-page.html',
  styleUrl: './user-edit-page.css',
})
export class UserEditPage implements OnInit {
  private readonly userService = inject(UserService);
  private readonly toast = inject(ToastService);
  private readonly fb = inject(NonNullableFormBuilder);

  readonly minPasswordLength = MIN_PASSWORD_LENGTH;
  readonly me = signal<UserDetails | null>(null);
  readonly loading = signal(true);
  readonly loadError = signal('');

  readonly savingProfile = signal(false);
  readonly profileError = signal('');
  readonly savingPassword = signal(false);
  readonly passwordError = signal('');

  readonly profileForm = this.fb.group({
    name: ['', [Validators.required]],
    picture: [''],
  });

  readonly passwordForm = this.fb.group(
    {
      current_password: [''],
      new_password: ['', [Validators.required, Validators.minLength(MIN_PASSWORD_LENGTH)]],
      repeat_password: [''],
    },
    { validators: passwordsMatch }
  );

  ngOnInit(): void {
    void this.load();
  }

  async load(): Promise<void> {
    this.loading.set(true);
    this.loadError.set('');
    try {
      this.show(await firstValueFrom(this.userService.getMe()));
    } catch (err) {
      this.loadError.set(errorMessage(err, 'Could not load your profile.'));
    } finally {
      this.loading.set(false);
    }
  }

  private show(user: UserDetails): void {
    this.me.set(user);
    this.profileForm.reset({ name: user.name, picture: user.picture ?? '' });
    const current = this.passwordForm.controls.current_password;
    current.setValidators(user.has_password ? [Validators.required] : []);
    this.passwordForm.reset();
  }

  /** The message under a field once it has been touched, or `''`. */
  nameError(): string {
    const control = this.profileForm.controls.name;
    return control.touched && control.invalid ? 'The name is required.' : '';
  }

  passwordFieldError(name: 'current_password' | 'new_password' | 'repeat_password'): string {
    const control = this.passwordForm.controls[name];
    if (!control.touched) return '';
    if (name === 'repeat_password') {
      return this.passwordForm.hasError('mismatch') ? 'The two passwords differ.' : '';
    }
    if (control.hasError('required')) {
      return name === 'current_password' ? 'Your current password is required.' : 'A new password is required.';
    }
    return control.hasError('minlength') ? `At least ${MIN_PASSWORD_LENGTH} characters.` : '';
  }

  async saveProfile(): Promise<void> {
    if (this.profileForm.invalid) {
      this.profileForm.markAllAsTouched();
      return;
    }
    const value = this.profileForm.getRawValue();
    this.savingProfile.set(true);
    this.profileError.set('');
    try {
      const saved = await firstValueFrom(
        this.userService.updateProfile({ name: value.name.trim(), picture: value.picture.trim() || null })
      );
      this.me.set(saved);
      this.profileForm.reset({ name: saved.name, picture: saved.picture ?? '' });
      this.userService.updateUserDetails();
      this.toast.success('Saved');
    } catch (err) {
      this.profileError.set(errorMessage(err, 'Could not save your profile.'));
    } finally {
      this.savingProfile.set(false);
    }
  }

  async savePassword(me: UserDetails): Promise<void> {
    if (this.passwordForm.invalid) {
      this.passwordForm.markAllAsTouched();
      return;
    }
    const value = this.passwordForm.getRawValue();
    this.savingPassword.set(true);
    this.passwordError.set('');
    try {
      const saved = await firstValueFrom(
        this.userService.changePassword(
          value.new_password,
          me.has_password ? value.current_password : undefined
        )
      );
      this.toast.success(me.has_password ? 'Password changed' : 'Password set');
      this.me.set(saved);
      this.passwordForm.controls.current_password.setValidators([Validators.required]);
      this.passwordForm.reset();
    } catch (err) {
      this.passwordError.set(errorMessage(err, 'Could not change your password.'));
    } finally {
      this.savingPassword.set(false);
    }
  }
}
