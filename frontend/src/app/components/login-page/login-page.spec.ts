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
import { HttpErrorResponse } from '@angular/common/http';
import { of, throwError } from 'rxjs';
import { LoginPage } from './login-page';
import { UserService } from '../../services/user-service';
import { makeUser, settle } from '../../../testing';

describe('LoginPage', () => {
  let fixture: ComponentFixture<LoginPage>;
  let el: HTMLElement;
  let userService: jasmine.SpyObj<UserService>;

  async function render(google = false): Promise<void> {
    userService.getAuthMethods.and.returnValue(of({ password: true, google }));
    fixture = TestBed.createComponent(LoginPage);
    el = fixture.nativeElement;
    fixture.detectChanges();
    await settle(fixture);
  }

  function query<T extends Element = HTMLElement>(testId: string): T | null {
    return el.querySelector<T>(`[data-testid="${testId}"]`);
  }

  async function signIn(email: string, password: string): Promise<void> {
    const emailInput = el.querySelector<HTMLInputElement>('#email')!;
    const passwordInput = el.querySelector<HTMLInputElement>('#password')!;
    emailInput.value = email;
    emailInput.dispatchEvent(new Event('input'));
    passwordInput.value = password;
    passwordInput.dispatchEvent(new Event('input'));
    await settle(fixture);
    query<HTMLButtonElement>('sign-in')!.click();
    await settle(fixture);
  }

  beforeEach(async () => {
    userService = jasmine.createSpyObj('UserService', [
      'getAuthMethods',
      'loginWithPassword',
      'login',
    ]);
    await TestBed.configureTestingModule({
      imports: [LoginPage],
      providers: [{ provide: UserService, useValue: userService }],
    }).compileComponents();
  });


  it('links to the documentation, the repository and the package', async () => {
    await render();
    const links = Array.from(el.querySelectorAll<HTMLAnchorElement>('.login-links a'));
    expect(links.map((a) => a.textContent!.trim())).toEqual(['Documentation', 'GitHub', 'PyPI']);
    expect(links.map((a) => a.href)).toEqual([
      'https://docs.kaval.ai/',
      'https://github.com/Kaval-AI/kavalai',
      'https://pypi.org/project/kavalai/',
    ]);
    expect(links.every((a) => a.target === '_blank' && a.rel === 'noopener')).toBeTrue();
  });

  it('offers Google only when the server has a Google client', async () => {
    await render(false);
    expect(query('google')).toBeNull();
    expect(el.querySelector('.divider')).toBeNull();

    fixture.destroy();
    await render(true);
    expect(query('google')).not.toBeNull();
    expect(el.querySelector('.divider')!.textContent).toContain('or with email');
  });

  it('hides Google when the sign-in methods cannot be read', async () => {
    userService.getAuthMethods.and.returnValue(throwError(() => new Error('down')));
    fixture = TestBed.createComponent(LoginPage);
    el = fixture.nativeElement;
    fixture.detectChanges();
    await settle(fixture);
    expect(query('google')).toBeNull();
  });

  it('starts the Google sign-in from its button', async () => {
    await render(true);
    query<HTMLButtonElement>('google')!.click();
    expect(userService.login).toHaveBeenCalled();
  });

  it('keeps the sign-in button disabled until both fields are filled', async () => {
    await render();
    expect(query<HTMLButtonElement>('sign-in')!.disabled).toBeTrue();
    fixture.componentInstance.email = 'ada@example.com';
    fixture.detectChanges();
    expect(query<HTMLButtonElement>('sign-in')!.disabled).toBeTrue();
    fixture.componentInstance.loginWithPassword();
    expect(userService.loginWithPassword).not.toHaveBeenCalled();
  });

  it('signs in with the typed address and password', async () => {
    userService.loginWithPassword.and.returnValue(of(makeUser()));
    await render();
    await signIn('ada@example.com', 'secret-password');
    expect(userService.loginWithPassword).toHaveBeenCalledOnceWith(
      'ada@example.com',
      'secret-password'
    );
    expect(query('login-error')).toBeNull();
  });

  it('shows a refusal and clears the password', async () => {
    userService.loginWithPassword.and.returnValue(
      throwError(() => new HttpErrorResponse({ status: 401, error: { detail: 'x' } }))
    );
    await render();
    await signIn('ada@example.com', 'wrong-password');
    expect(query('login-error')!.textContent!.trim()).toBe('Invalid email or password.');
    expect(fixture.componentInstance.password).toBe('');
    expect(query<HTMLButtonElement>('sign-in')!.disabled).toBeTrue();
  });

  it("shows the server's reason for any other failure", async () => {
    userService.loginWithPassword.and.returnValue(
      throwError(() => new HttpErrorResponse({ status: 503, error: { detail: 'Database not connected.' } }))
    );
    await render();
    await signIn('ada@example.com', 'secret-password');
    expect(query('login-error')!.textContent!.trim()).toBe('Database not connected.');
  });

  it('says how an account is made, since there is no sign-up', async () => {
    await render();
    expect(el.querySelector('.login-note')!.textContent).toContain('administrator');
  });
});
