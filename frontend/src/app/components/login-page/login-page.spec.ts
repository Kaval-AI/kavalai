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
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { UserService } from '../../services/user-service';

import { LoginPage } from './login-page';

describe('LoginPage', () => {
  let component: LoginPage;
  let fixture: ComponentFixture<LoginPage>;
  let userService: UserService;
  let httpMock: HttpTestingController;

  function answerAuthMethods(google: boolean): void {
    httpMock.expectOne('/api/auth/methods').flush({ password: true, google });
    fixture.detectChanges();
  }

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [LoginPage],
      providers: [
        provideHttpClient(),
        provideHttpClientTesting(),
        UserService,
      ]
    }).compileComponents();

    fixture = TestBed.createComponent(LoginPage);
    component = fixture.componentInstance;
    userService = TestBed.inject(UserService);
    httpMock = TestBed.inject(HttpTestingController);
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should call userService.login when login is called', () => {
    const loginSpy = spyOn(userService, 'login');
    component.login();
    expect(loginSpy).toHaveBeenCalled();
  });

  it('should have header links', () => {
    const compiled = fixture.nativeElement as HTMLElement;
    const links = compiled.querySelectorAll('div.navbar a');
    expect(links.length).toBe(3);
    expect(links[0].textContent).toContain('Docs');
    expect(links[0].getAttribute('href')).toBe('https://docs.kaval.ai');
    expect(links[1].textContent).toContain('GitHub');
    expect(links[1].getAttribute('href')).toBe('https://github.com/kavalai/kavalai');
    expect(links[2].textContent).toContain('PyPI');
    expect(links[2].getAttribute('href')).toBe('https://pypi.org/project/kavalai/');
  });

  it('should offer Google sign-in when the server has it configured', () => {
    answerAuthMethods(true);
    const compiled = fixture.nativeElement as HTMLElement;
    const loginButton = compiled.querySelector('button.login-google-button');
    expect(loginButton).toBeTruthy();
    expect(loginButton?.textContent).toContain('Sign in with Google');
  });

  it('should hide Google sign-in when the server has none', () => {
    const compiled = fixture.nativeElement as HTMLElement;
    expect(compiled.querySelector('button.login-google-button')).toBeNull();
    answerAuthMethods(false);
    expect(compiled.querySelector('button.login-google-button')).toBeNull();
    expect(compiled.querySelector('form.login-form')).toBeTruthy();
  });

  it('should hide Google sign-in when the methods cannot be read', () => {
    httpMock.expectOne('/api/auth/methods').error(new ErrorEvent('Network error'));
    fixture.detectChanges();
    const compiled = fixture.nativeElement as HTMLElement;
    expect(compiled.querySelector('button.login-google-button')).toBeNull();
  });

  it('should call login when the Google button is clicked', () => {
    answerAuthMethods(true);
    spyOn(component, 'login');
    const compiled = fixture.nativeElement as HTMLElement;
    const loginButton = compiled.querySelector('button.login-google-button') as HTMLButtonElement;
    loginButton.click();
    expect(component.login).toHaveBeenCalled();
  });

  it('should have an email and a password field', () => {
    const compiled = fixture.nativeElement as HTMLElement;
    expect(compiled.querySelector('input[type="email"]')).toBeTruthy();
    expect(compiled.querySelector('input[type="password"]')).toBeTruthy();
    const submit = compiled.querySelector('button.login-submit-button') as HTMLButtonElement;
    expect(submit.textContent).toContain('Sign in');
    expect(submit.disabled).toBeTrue();
  });

  it('should sign in with the entered email and password', () => {
    component.email = 'user@test.com';
    component.password = 'correct horse';
    component.loginWithPassword();

    const req = httpMock.expectOne('/api/auth/login');
    expect(req.request.method).toBe('POST');
    expect(req.request.body).toEqual({ email: 'user@test.com', password: 'correct horse' });
    expect(component.isSubmitting).toBeTrue();
    req.flush({ id: 'u1', email: 'user@test.com', name: 'User', picture: '', is_admin: false });

    expect(component.isSubmitting).toBeFalse();
    expect(component.errorMessage).toBe('');
    expect(userService.getIsLoggedIn()).toBeTrue();
  });

  it('should not submit without an email and a password', () => {
    component.email = 'user@test.com';
    component.loginWithPassword();
    httpMock.expectNone('/api/auth/login');
  });

  it('should show an error and clear the password when sign-in is refused', () => {
    component.email = 'user@test.com';
    component.password = 'wrong';
    component.loginWithPassword();
    httpMock.expectOne('/api/auth/login').flush(
      { detail: 'Invalid email or password.' },
      { status: 401, statusText: 'Unauthorized' }
    );
    fixture.detectChanges();

    expect(component.password).toBe('');
    expect(component.isSubmitting).toBeFalse();
    const error = (fixture.nativeElement as HTMLElement).querySelector('.login-error');
    expect(error?.textContent).toContain('Invalid email or password.');
    expect(userService.getIsLoggedIn()).toBeFalse();
  });

  it('should report a failure that is not a refusal differently', () => {
    component.email = 'user@test.com';
    component.password = 'correct horse';
    component.loginWithPassword();
    httpMock.expectOne('/api/auth/login').flush(
      {}, { status: 503, statusText: 'Service Unavailable' }
    );

    expect(component.errorMessage).toBe('Sign-in failed. Please try again.');
  });

  it('should have a background container with the correct class', () => {
    const compiled = fixture.nativeElement as HTMLElement;
    const backgroundContainer = compiled.querySelector('.login-background-container');
    expect(backgroundContainer).toBeTruthy();
  });
});
