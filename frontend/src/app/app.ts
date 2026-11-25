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
import { Shell } from './layout/shell';
import { LoginPage } from './components/login-page/login-page';
import { Toast } from './components/toast/toast';
import { UserService } from './services/user-service';

@Component({
  selector: 'app-root',
  imports: [LoginPage, Shell, Toast],
  templateUrl: './app.html',
  styleUrl: './app.css',
})
export class App implements OnInit {
  constructor(private userService: UserService) {}

  ngOnInit(): void {
    this.userService.updateUserDetails();
  }

  get isLoggedIn(): boolean {
    return this.userService.getIsLoggedIn();
  }
}
