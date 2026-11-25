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
import { FormsModule } from '@angular/forms';
import { HttpErrorResponse } from '@angular/common/http';
import { UserService } from '../../services/user-service';
import { errorMessage } from '../../core/http-error';
import { UiAlert, UiButton, UiField, UiSpinner } from '../../ui';

/**
 * The sign-in page: a password form, and Google above it when the server has
 * a Google client (`GET /auth/methods`). There is no sign-up: accounts are
 * created by an administrator, on the Users page or with
 * `python -m kavalai.backoffice.users`.
 */
@Component({
  selector: 'app-login-page',
  imports: [FormsModule, UiAlert, UiButton, UiField, UiSpinner],
  templateUrl: './login-page.html',
  styleUrl: './login-page.css',
})
export class LoginPage implements OnInit {
  private userService = inject(UserService);

  email = '';
  password = '';
  readonly errorMessage = signal('');
  readonly isSubmitting = signal(false);
  readonly googleEnabled = signal(false);

  ngOnInit(): void {
    this.userService.getAuthMethods().subscribe({
      next: (methods) => this.googleEnabled.set(methods.google),
      error: () => this.googleEnabled.set(false),
    });
  }

  loginWithPassword(): void {
    if (!this.email || !this.password || this.isSubmitting()) return;

    this.isSubmitting.set(true);
    this.errorMessage.set('');
    this.userService.loginWithPassword(this.email, this.password).subscribe({
      next: () => this.isSubmitting.set(false),
      error: (err: unknown) => {
        this.isSubmitting.set(false);
        this.password = '';
        this.errorMessage.set(
          err instanceof HttpErrorResponse && err.status === 401
            ? 'Invalid email or password.'
            : errorMessage(err, 'Sign-in failed. Please try again.')
        );
      },
    });
  }

  login(): void {
    this.userService.login();
  }
}
