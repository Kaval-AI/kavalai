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
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { Observable, Subject, of, throwError } from 'rxjs';
import { UserDetails } from '../../models/user-details';
import { ToastService } from '../../services/toast-service';
import { UserService } from '../../services/user-service';
import { makeUser, settle, signedInAs } from '../../../testing';
import { UserEditPage } from './user-edit-page';

function refused(detail: string, status = 400): HttpErrorResponse {
  return new HttpErrorResponse({ status, error: { detail } });
}

/**
 * The profile calls of `UserService`, which `signedInAs` does not stub:
 * added to its double here.
 */
interface ProfileCalls {
  getMe: jasmine.Spy<() => Observable<UserDetails>>;
  updateProfile: jasmine.Spy<UserService['updateProfile']>;
  changePassword: jasmine.Spy<UserService['changePassword']>;
}

describe('UserEditPage (profile)', () => {
  let fixture: ComponentFixture<UserEditPage>;
  let users: jasmine.SpyObj<UserService> & ProfileCalls;
  let toast: ToastService;

  async function render(me: UserDetails = makeUser()): Promise<void> {
    users.getMe.and.returnValue(of(me));
    fixture = TestBed.createComponent(UserEditPage);
    fixture.detectChanges();
    await settle(fixture);
  }

  function find<T extends HTMLElement = HTMLElement>(testid: string): T | null {
    return fixture.nativeElement.querySelector(`[data-testid="${testid}"]`);
  }

  function type(testid: string, value: string): void {
    const input = find<HTMLInputElement>(testid)!;
    input.value = value;
    input.dispatchEvent(new Event('input'));
    input.dispatchEvent(new Event('blur'));
  }

  async function click(testid: string): Promise<void> {
    find(testid)!.click();
    await settle(fixture);
  }

  beforeEach(async () => {
    users = Object.assign(signedInAs('viewer'), {
      getMe: jasmine.createSpy('getMe'),
      updateProfile: jasmine.createSpy('updateProfile'),
      changePassword: jasmine.createSpy('changePassword'),
    }) as jasmine.SpyObj<UserService> & ProfileCalls;
    await TestBed.configureTestingModule({
      imports: [UserEditPage],
      providers: [{ provide: UserService, useValue: users }],
    }).compileComponents();
    toast = TestBed.inject(ToastService);
    spyOn(toast, 'success');
  });

  it('shows a spinner while the profile loads', () => {
    users.getMe.and.returnValue(new Subject<UserDetails>());
    fixture = TestBed.createComponent(UserEditPage);
    fixture.detectChanges();

    expect(fixture.nativeElement.querySelector('ui-loading')).not.toBeNull();
    expect(find('profile-card')).toBeNull();
  });

  it('shows a failed load with a Retry that loads again', async () => {
    users.getMe.and.returnValue(throwError(() => refused('Unauthorized.', 401)));
    fixture = TestBed.createComponent(UserEditPage);
    fixture.detectChanges();
    await settle(fixture);

    expect(find('load-error')!.textContent).toContain('Unauthorized.');

    users.getMe.and.returnValue(of(makeUser()));
    await click('retry');

    expect(find('load-error')).toBeNull();
    expect(find('profile-card')).not.toBeNull();
  });

  it('shows the user, their access and their current details', async () => {
    await render(makeUser({ is_admin: true, picture: 'https://example.com/ada.png' }));

    const header = fixture.nativeElement.querySelector('ui-page-header');
    expect(header.textContent).toContain('Profile');
    expect(header.textContent).toContain('ada@example.com');
    const card = find('profile-card')!;
    expect(card.textContent).toContain('admin');
    expect(card.textContent).toContain('password');
    expect(find<HTMLInputElement>('profile-name')!.value).toBe('Ada Lovelace');
    expect(find<HTMLInputElement>('profile-picture')!.value).toBe('https://example.com/ada.png');
    expect(card.querySelector('ui-avatar img')!.getAttribute('src')).toBe('https://example.com/ada.png');
  });

  describe('the details card', () => {
    it('saves the name and picture, then refreshes the signed-in user', async () => {
      await render();
      users.updateProfile.and.returnValue(of(makeUser({ name: 'Ada King', picture: null })));

      type('profile-name', ' Ada King ');
      type('profile-picture', '  ');
      await click('save-profile');

      expect(users.updateProfile).toHaveBeenCalledWith({ name: 'Ada King', picture: null });
      expect(users.updateUserDetails).toHaveBeenCalled();
      expect(toast.success).toHaveBeenCalledWith('Saved');
      expect(find<HTMLInputElement>('profile-name')!.value).toBe('Ada King');
      expect(find('profile-card')!.querySelector('ui-avatar span')!.textContent).toContain('AK');
    });

    it('refuses an empty name without calling the server', async () => {
      await render();

      type('profile-name', '');
      await click('save-profile');

      expect(find('profile-card')!.textContent).toContain('The name is required.');
      expect(users.updateProfile).not.toHaveBeenCalled();
    });

    it('holds a spinner in the busy Save button', async () => {
      await render();
      users.updateProfile.and.returnValue(new Subject<UserDetails>());

      await click('save-profile');

      const save = find<HTMLButtonElement>('save-profile')!;
      expect(save.disabled).toBeTrue();
      expect(save.querySelector('ui-spinner')).not.toBeNull();
    });

    it('shows the server refusal in the card', async () => {
      await render();
      users.updateProfile.and.returnValue(throwError(() => refused('The name is required.')));

      await click('save-profile');

      expect(find('profile-error')!.textContent).toContain('The name is required.');
      expect(users.updateUserDetails).not.toHaveBeenCalled();
      expect(toast.success).not.toHaveBeenCalled();
    });
  });

  describe('the password card', () => {
    it('changes the password with the current one', async () => {
      await render();
      users.changePassword.and.returnValue(of(makeUser()));

      type('current-password', 'correct horse');
      type('new-password', 'battery staple');
      type('repeat-password', 'battery staple');
      await click('save-password');

      expect(users.changePassword).toHaveBeenCalledWith('battery staple', 'correct horse');
      expect(toast.success).toHaveBeenCalledWith('Password changed');
      expect(find<HTMLInputElement>('new-password')!.value).toBe('');
      expect(find<HTMLInputElement>('current-password')!.value).toBe('');
    });

    it('refuses a missing current password, a short one and a mismatch', async () => {
      await render();

      await click('save-password');
      const card = find('password-card')!;
      expect(card.textContent).toContain('Your current password is required.');
      expect(card.textContent).toContain('A new password is required.');

      type('current-password', 'correct horse');
      type('new-password', 'short');
      type('repeat-password', 'shorter');
      await click('save-password');

      expect(card.textContent).toContain('At least 8 characters.');
      expect(card.textContent).toContain('The two passwords differ.');
      expect(users.changePassword).not.toHaveBeenCalled();
    });

    it('holds a spinner in the busy button', async () => {
      await render();
      users.changePassword.and.returnValue(new Subject<UserDetails>());

      type('current-password', 'correct horse');
      type('new-password', 'battery staple');
      type('repeat-password', 'battery staple');
      await click('save-password');

      const save = find<HTMLButtonElement>('save-password')!;
      expect(save.disabled).toBeTrue();
      expect(save.querySelector('ui-spinner')).not.toBeNull();
    });

    it('shows a wrong current password in the card', async () => {
      await render();
      users.changePassword.and.returnValue(throwError(() => refused('The current password is not correct.')));

      type('current-password', 'wrong horse');
      type('new-password', 'battery staple');
      type('repeat-password', 'battery staple');
      await click('save-password');

      expect(find('password-error')!.textContent).toContain('The current password is not correct.');
      expect(toast.success).not.toHaveBeenCalled();
    });

    it('lets a Google-only user set a first password without a current one', async () => {
      await render(makeUser({ has_password: false }));
      users.changePassword.and.returnValue(of(makeUser({ has_password: true })));

      expect(find('current-password')).toBeNull();
      expect(find('no-password-note')!.textContent).toContain('sign in with Google');
      expect(find('profile-card')!.textContent).toContain('Google only');
      expect(find('save-password')!.textContent).toContain('Set password');

      type('new-password', 'battery staple');
      type('repeat-password', 'battery staple');
      await click('save-password');

      expect(users.changePassword).toHaveBeenCalledWith('battery staple', undefined);
      expect(toast.success).toHaveBeenCalledWith('Password set');
      expect(find('current-password')).not.toBeNull();
      expect(find('save-password')!.textContent).toContain('Change password');
    });
  });
});
