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
import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { UserService } from '../../services/user-service';

@Component({
  selector: 'app-login-page',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
  ],
  templateUrl: './login-page.html',
  styleUrls: ['./login-page.css'],
})
export class LoginPage implements OnInit {
  email = '';
  password = '';
  errorMessage = '';
  isSubmitting = false;
  googleEnabled = false;

  constructor(
    private userService: UserService
  ) {}

  ngOnInit(): void {
    this.userService.getAuthMethods().subscribe({
      next: (methods) => {
        this.googleEnabled = methods.google;
      },
      error: () => {
        this.googleEnabled = false;
      },
    });
  }

  loginWithPassword(): void {
    if (!this.email || !this.password || this.isSubmitting) return;

    this.isSubmitting = true;
    this.errorMessage = '';
    this.userService.loginWithPassword(this.email, this.password).subscribe({
      next: () => {
        this.isSubmitting = false;
      },
      error: (err) => {
        this.isSubmitting = false;
        this.password = '';
        this.errorMessage = err.status === 401
          ? 'Invalid email or password.'
          : 'Sign-in failed. Please try again.';
      },
    });
  }

  login(): void {
    this.userService.login();
  }
}
