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
import { Subject, of, throwError } from 'rxjs';
import { UserDetails } from '../../models/user-details';
import { ToastService } from '../../services/toast-service';
import { UserService } from '../../services/user-service';
import { makeUser, settle, signedInAs } from '../../../testing';
import { UsersPage } from './users-page';

function refused(detail: string, status = 400): HttpErrorResponse {
  return new HttpErrorResponse({ status, error: { detail } });
}

describe('UsersPage', () => {
  let fixture: ComponentFixture<UsersPage>;
  let users: jasmine.SpyObj<UserService>;
  let toast: ToastService;

  const me = makeUser({ id: 'u1', name: 'Ada Lovelace', is_admin: true, has_password: true });
  const grace = makeUser({
    id: 'u2',
    email: 'grace@example.com',
    name: 'Grace Hopper',
    picture: 'https://example.com/grace.png',
    has_password: false,
  });

  async function render(list: UserDetails[] = [me, grace]): Promise<HTMLElement> {
    users.getUsers.and.returnValue(of(list));
    fixture = TestBed.createComponent(UsersPage);
    fixture.detectChanges();
    await settle(fixture);
    return fixture.nativeElement;
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

  function dialogs(): HTMLDialogElement[] {
    return Array.from(fixture.nativeElement.querySelectorAll('dialog'));
  }

  async function submit(): Promise<void> {
    find('save-user')!.click();
    await settle(fixture);
  }

  beforeEach(async () => {
    users = signedInAs('admin', { id: 'u1', name: 'Ada Lovelace' });
    await TestBed.configureTestingModule({
      imports: [UsersPage],
      providers: [{ provide: UserService, useValue: users }],
    }).compileComponents();
    toast = TestBed.inject(ToastService);
    spyOn(toast, 'success');
  });

  it('shows a spinner while the users load', () => {
    users.getUsers.and.returnValue(new Subject<UserDetails[]>());
    fixture = TestBed.createComponent(UsersPage);
    fixture.detectChanges();

    expect(fixture.nativeElement.querySelector('ui-loading')).not.toBeNull();
    expect(find('users-table')).toBeNull();
  });

  it('shows a failed load with a Retry that loads again', async () => {
    users.getUsers.and.returnValue(throwError(() => refused('Only administrators can do this.', 403)));
    fixture = TestBed.createComponent(UsersPage);
    fixture.detectChanges();
    await settle(fixture);

    expect(find('load-error')!.textContent).toContain('Only administrators can do this.');

    users.getUsers.and.returnValue(of([me]));
    find('retry')!.click();
    await settle(fixture);

    expect(find('load-error')).toBeNull();
    expect(find('users-table')).not.toBeNull();
  });

  it('names the next step when there are no users', async () => {
    await render([]);

    expect(find('users-empty')!.textContent).toContain('Add a user');
  });

  it('lists each user with their avatar, address and access', async () => {
    await render();

    const ada = find('user-u1')!;
    expect(ada.textContent).toContain('Ada Lovelace');
    expect(ada.textContent).toContain('(you)');
    expect(ada.textContent).toContain('admin');
    expect(ada.textContent).toContain('password');
    expect(ada.querySelector('ui-avatar span')!.textContent).toContain('AL');

    const row = find('user-u2')!;
    expect(row.textContent).toContain('grace@example.com');
    expect(row.textContent).toContain('Google only');
    expect(row.textContent).not.toContain('admin');
    expect(row.textContent).not.toContain('(you)');
    expect(row.querySelector('ui-avatar img')!.getAttribute('src')).toBe('https://example.com/grace.png');
  });

  it('does not let administrators delete themselves, and says why', async () => {
    await render();

    const own = find<HTMLButtonElement>('delete-u1')!;
    expect(own.disabled).toBeTrue();
    expect(own.title).toBe('You cannot delete your own account.');
    expect(find('self-delete-note')!.textContent).toContain('another administrator can');
    expect(find<HTMLButtonElement>('delete-u2')!.disabled).toBeFalse();
  });

  describe('the user form', () => {
    it('adds a user, with a password and administrator access', async () => {
      await render();
      users.createUser.and.returnValue(of(makeUser({ id: 'u3' })));

      find('add-user')!.click();
      fixture.detectChanges();
      expect(dialogs()[0].open).toBeTrue();
      expect(dialogs()[0].textContent).toContain('Add user');
      expect(dialogs()[0].textContent).toContain('signs in through Google only');

      type('user-email', ' alan@example.com ');
      type('user-name', 'Alan Turing');
      type('user-password', 'correct horse');
      find<HTMLInputElement>('user-admin')!.click();
      await submit();

      expect(users.createUser).toHaveBeenCalledWith({
        email: 'alan@example.com',
        name: 'Alan Turing',
        picture: null,
        is_admin: true,
        password: 'correct horse',
      });
      expect(toast.success).toHaveBeenCalledWith('User added');
      expect(dialogs()[0].open).toBeFalse();
      expect(users.getUsers).toHaveBeenCalledTimes(2);
    });

    it('refuses an incomplete form without calling the server', async () => {
      await render();
      find('add-user')!.click();
      fixture.detectChanges();

      await submit();
      expect(dialogs()[0].textContent).toContain('The email is required.');
      expect(dialogs()[0].textContent).toContain('The name is required.');

      type('user-email', 'not an address');
      type('user-name', 'Alan');
      type('user-password', 'short');
      await submit();

      expect(dialogs()[0].textContent).toContain('This is not an email address.');
      expect(dialogs()[0].textContent).toContain('At least 8 characters.');
      expect(users.createUser).not.toHaveBeenCalled();
      expect(dialogs()[0].open).toBeTrue();
    });

    it('holds a spinner in the busy Save button', async () => {
      await render();
      users.createUser.and.returnValue(new Subject<UserDetails>());
      find('add-user')!.click();
      fixture.detectChanges();
      type('user-email', 'alan@example.com');
      type('user-name', 'Alan Turing');
      await submit();

      const save = find<HTMLButtonElement>('save-user')!;
      expect(save.disabled).toBeTrue();
      expect(save.querySelector('ui-spinner')).not.toBeNull();
    });

    it('shows the server refusal in the dialog', async () => {
      await render();
      users.createUser.and.returnValue(throwError(() => refused('The password must be at least 8 characters long.')));
      find('add-user')!.click();
      fixture.detectChanges();
      type('user-email', 'alan@example.com');
      type('user-name', 'Alan Turing');
      await submit();

      expect(find('save-error')!.textContent).toContain('The password must be at least 8 characters long.');
      expect(dialogs()[0].open).toBeTrue();
      expect(toast.success).not.toHaveBeenCalled();
    });

    it('edits a user and keeps the password when the field is left empty', async () => {
      await render();
      users.updateUser.and.returnValue(of(grace));

      find('edit-u2')!.click();
      fixture.detectChanges();
      expect(dialogs()[0].textContent).toContain('Edit user');
      expect(dialogs()[0].textContent).toContain('Leave empty to keep the current password.');
      expect(find<HTMLInputElement>('user-email')!.value).toBe('grace@example.com');
      expect(find<HTMLInputElement>('user-picture')!.value).toBe('https://example.com/grace.png');
      expect(find<HTMLInputElement>('user-password')!.value).toBe('');

      type('user-picture', '');
      await submit();

      expect(users.updateUser).toHaveBeenCalledWith('u2', {
        email: 'grace@example.com',
        name: 'Grace Hopper',
        picture: null,
        is_admin: false,
      });
      expect(toast.success).toHaveBeenCalledWith('Saved');
      expect(users.updateUserDetails).not.toHaveBeenCalled();
      expect(dialogs()[0].open).toBeFalse();
    });

    it('sends a new password, and refreshes the shell after editing oneself', async () => {
      await render();
      users.updateUser.and.returnValue(of(me));

      find('edit-u1')!.click();
      fixture.detectChanges();
      type('user-password', 'battery staple');
      await submit();

      expect(users.updateUser).toHaveBeenCalledWith('u1', jasmine.objectContaining({ password: 'battery staple' }));
      expect(users.updateUserDetails).toHaveBeenCalled();
    });

    it('closes on Cancel, and ignores a submit while closed', async () => {
      await render();
      find('edit-u2')!.click();
      fixture.detectChanges();
      find('cancel-user')!.click();
      fixture.detectChanges();
      expect(dialogs()[0].open).toBeFalse();

      find('user-form')!.dispatchEvent(new Event('submit'));
      await settle(fixture);
      expect(users.updateUser).not.toHaveBeenCalled();
      expect(users.createUser).not.toHaveBeenCalled();
    });
  });

  describe('deleting a user', () => {
    it('asks first, then deletes and drops the row', async () => {
      await render();
      users.deleteUser.and.returnValue(of({ status: 'deleted' }));

      find('delete-u2')!.click();
      fixture.detectChanges();
      expect(dialogs()[1].open).toBeTrue();
      expect(find('delete-question')!.textContent).toContain('grace@example.com');
      expect(users.deleteUser).not.toHaveBeenCalled();

      find('confirm-delete')!.click();
      await settle(fixture);

      expect(users.deleteUser).toHaveBeenCalledWith('u2');
      expect(find('user-u2')).toBeNull();
      expect(dialogs()[1].open).toBeFalse();
      expect(toast.success).toHaveBeenCalledWith('User deleted');
    });

    it('shows a refused deletion in the dialog, busy while it runs', async () => {
      await render();
      const answer = new Subject<unknown>();
      users.deleteUser.and.returnValue(answer);

      find('delete-u2')!.click();
      fixture.detectChanges();
      find('confirm-delete')!.click();
      fixture.detectChanges();
      expect(find<HTMLButtonElement>('confirm-delete')!.disabled).toBeTrue();
      expect(find('confirm-delete')!.querySelector('ui-spinner')).not.toBeNull();

      answer.error(refused('User not found', 404));
      await settle(fixture);

      expect(find('delete-error')!.textContent).toContain('User not found');
      expect(dialogs()[1].open).toBeTrue();
      expect(find('user-u2')).not.toBeNull();
    });

    it('closes on Cancel, and does nothing when confirmed while closed', async () => {
      await render();
      find('delete-u2')!.click();
      fixture.detectChanges();
      find('cancel-delete')!.click();
      fixture.detectChanges();
      expect(dialogs()[1].open).toBeFalse();

      find('confirm-delete')!.click();
      await settle(fixture);
      expect(users.deleteUser).not.toHaveBeenCalled();
    });
  });
});
