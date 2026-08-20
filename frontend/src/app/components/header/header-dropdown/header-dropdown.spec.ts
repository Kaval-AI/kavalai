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

import { ComponentFixture, TestBed, fakeAsync, tick } from '@angular/core/testing';
import { provideRouter, Router } from '@angular/router';
import { of } from 'rxjs';
import { Location } from '@angular/common';

import { HeaderDropdown } from './header-dropdown';
import { UserService } from '../../../services/user-service';

describe('HeaderDropdown', () => {
  let component: HeaderDropdown;
  let fixture: ComponentFixture<HeaderDropdown>;
  let userServiceSpy: jasmine.SpyObj<UserService>;
  let router: Router;
  let location: Location;

  beforeEach(async () => {
    userServiceSpy = jasmine.createSpyObj('UserService', ['getIsAdmin'], { userDetails: of({ id: 'u1' }) });

    await TestBed.configureTestingModule({
      imports: [HeaderDropdown],
      providers: [
        provideRouter([
          { path: '', component: HeaderDropdown },
          { path: 'agents', component: HeaderDropdown }
        ]),
        { provide: UserService, useValue: userServiceSpy },
      ]
    }).compileComponents();

    fixture = TestBed.createComponent(HeaderDropdown);
    component = fixture.componentInstance;
    router = TestBed.inject(Router);
    location = TestBed.inject(Location);
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should navigate to /agents when Agents link is clicked', fakeAsync(() => {
    component.isMenuOpen = true;
    fixture.detectChanges();

    const agentsLink = fixture.nativeElement.querySelector('a[routerLink="/agents"]');
    expect(agentsLink).toBeTruthy();

    const dropdownButton = fixture.nativeElement.querySelector('.btn-circle');
    dropdownButton.focus();
    fixture.detectChanges();

    // Trigger blur on dropdown button
    dropdownButton.dispatchEvent(new Event('blur'));
    tick(200); // Wait for the timeout
    fixture.detectChanges();

    // The menu should now be hidden
    const dropdownContent = fixture.nativeElement.querySelector('.dropdown-content');
    expect(dropdownContent.classList).toContain('hidden');

    agentsLink.click();
    tick();

    expect(location.path()).toBe('/agents');
  }));

  it('should have 4 menu sections with correct titles', () => {
    userServiceSpy.getIsAdmin.and.returnValue(true);
    fixture.detectChanges();

    const menuTitles = fixture.nativeElement.querySelectorAll('.menu-title');
    expect(menuTitles.length).toBe(4);
    expect(menuTitles[0].textContent).toBe('Admin');
    expect(menuTitles[1].textContent).toBe('Agents');
    expect(menuTitles[2].textContent).toBe('RAG');
    expect(menuTitles[3].textContent).toBe('Other');
  });

  it('should have correct links in each section when admin', () => {
    userServiceSpy.getIsAdmin.and.returnValue(true);
    fixture.detectChanges();

    const dropdownContent = fixture.nativeElement.querySelector('.dropdown-content');
    const menus = dropdownContent.querySelectorAll('ul.menu');

    // Admin section (1st column)
    const adminMenu = menus[0].querySelectorAll('li');
    expect(adminMenu[0].querySelector('.menu-title').textContent).toBe('Admin');
    expect(adminMenu[1].querySelector('a').getAttribute('routerLink')).toBe('/');
    expect(adminMenu[2].querySelector('a').getAttribute('routerLink')).toBe('/users');

    // Agents section (2nd column)
    const agentsMenu = menus[1].querySelectorAll('li');
    expect(agentsMenu[0].querySelector('.menu-title').textContent).toBe('Agents');
    expect(agentsMenu[1].querySelector('a').getAttribute('routerLink')).toBe('/agents');
    expect(agentsMenu[2].querySelector('a').getAttribute('routerLink')).toBe('/workflows');
    expect(agentsMenu[3].querySelector('a').getAttribute('routerLink')).toBe('/conversations');
    expect(agentsMenu[4].querySelector('a').getAttribute('routerLink')).toBe('/llm-call-stats');

    // RAG section (3rd column)
    const ragMenu = menus[2].querySelectorAll('li');
    expect(ragMenu[0].querySelector('.menu-title').textContent).toBe('RAG');
    expect(ragMenu[1].querySelector('a').getAttribute('routerLink')).toBe('/rag');

    // Other section (4th column)
    const otherMenu = menus[3].querySelectorAll('li');
    expect(otherMenu[0].querySelector('.menu-title').textContent).toBe('Other');
    const docsLink = otherMenu[1].querySelector('a');
    expect(docsLink.getAttribute('href')).toBe('https://docs.kaval.ai/');
    expect(docsLink.getAttribute('target')).toBe('_blank');
    expect(docsLink.textContent).toBe('Documentation');
  });

  it('should NOT show Users link when NOT admin', () => {
    userServiceSpy.getIsAdmin.and.returnValue(false);
    fixture.detectChanges();

    const adminMenuLinks = fixture.nativeElement.querySelectorAll('ul.menu:first-child li a');
    const usersLink = Array.from(adminMenuLinks).find((a: any) => a.textContent.includes('Users'));
    expect(usersLink).toBeFalsy();
  });

  it('should call closeMenu when dropdown content is mousedown', fakeAsync(() => {
    spyOn(component, 'closeMenu').and.callThrough();
    const dropdownContent = fixture.nativeElement.querySelector('.dropdown-content');
    dropdownContent.dispatchEvent(new MouseEvent('mousedown'));
    expect(component.closeMenu).toHaveBeenCalled();
    tick(200);
  }));

  it('should set isMenuOpen to false when closeMenu is called', fakeAsync(() => {
    component.isMenuOpen = true;
    component.closeMenu();
    tick(200);
    expect(component.isMenuOpen).toBeFalse();
  }));

  it('should set isMenuOpen to true when openMenu is called', () => {
    component.isMenuOpen = false;
    component.openMenu();
    expect(component.isMenuOpen).toBeTrue();
  });

  it('should toggle hidden and z-50 classes based on isMenuOpen', fakeAsync(() => {
    const dropdownContent = fixture.nativeElement.querySelector('.dropdown-content');

    component.isMenuOpen = true;
    fixture.detectChanges();
    expect(dropdownContent.classList).not.toContain('hidden');
    expect(dropdownContent.classList).toContain('z-50');

    component.closeMenu();
    tick(200);
    fixture.detectChanges();
    expect(dropdownContent.classList).toContain('hidden');
  }));

  it('should have the correct classes for the dropdown content', () => {
    const dropdownContent = fixture.nativeElement.querySelector('.dropdown-content');
    expect(dropdownContent.classList).toContain('dropdown-content');
    expect(dropdownContent.classList).toContain('rounded-box');
    expect(dropdownContent.classList).toContain('shadow');
  });
});
