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
import { NavigationEnd, Router, RouterLink, RouterOutlet } from '@angular/router';
import { toSignal } from '@angular/core/rxjs-interop';
import { filter } from 'rxjs/operators';
import { Project } from '../models/project';
import { ProjectService } from '../services/project-service';
import { UserService } from '../services/user-service';
import { UiAvatar, UiBadge, UiButton, UiMenu, UiMenuItem, UiNavItem } from '../ui';

/** A sidebar destination. */
export interface NavItem {
  link: string;
  label: string;
  icon: string;
  exact?: boolean;
}

/** The pages of the active project, in the sidebar's order. */
export const PROJECT_NAV: NavItem[] = [
  { link: '/', label: 'Overview', icon: 'fa-gauge-high', exact: true },
  { link: '/conversations', label: 'Conversations', icon: 'fa-comments' },
  { link: '/workflows', label: 'Runs', icon: 'fa-diagram-project' },
  { link: '/llm-call-stats', label: 'Model calls', icon: 'fa-microchip' },
  { link: '/agents', label: 'Agents', icon: 'fa-robot' },
  { link: '/rag', label: 'RAG', icon: 'fa-layer-group' },
];

/** The settings of the active project. */
export const SETTINGS_NAV: NavItem[] = [
  { link: '/settings', label: 'Settings', icon: 'fa-gear' },
];

/** What administrators also manage. */
export const ADMIN_NAV: NavItem[] = [
  { link: '/users', label: 'Users', icon: 'fa-users' },
];

export const RESOURCE_LINKS = [
  { href: 'https://docs.kaval.ai', label: 'Documentation', icon: 'fa-solid fa-book' },
  { href: 'https://github.com/Kaval-AI/kavalai', label: 'GitHub', icon: 'fa-brands fa-github' },
  { href: 'https://pypi.org/project/kavalai/', label: 'PyPI', icon: 'fa-brands fa-python' },
];

/**
 * The signed-in frame of every page: the charcoal top bar with the user menu,
 * the sidebar with the project switcher and the destinations, and the page.
 *
 * The switcher is the one place the active project changes. It saves the
 * choice (`UserService.setActiveProject`), whose new user details the pages
 * already follow. Below 900px the sidebar is hidden behind the burger.
 */
@Component({
  selector: 'app-shell',
  imports: [RouterLink, RouterOutlet, UiAvatar, UiBadge, UiButton, UiMenu, UiMenuItem, UiNavItem],
  templateUrl: './shell.html',
  styleUrl: './shell.css',
})
export class Shell implements OnInit {
  private userService = inject(UserService);
  private projectService = inject(ProjectService);
  private router = inject(Router);

  readonly projectNav = PROJECT_NAV;
  readonly settingsNav = SETTINGS_NAV;
  readonly adminNav = ADMIN_NAV;
  readonly resourceLinks = RESOURCE_LINKS;

  readonly user = toSignal(this.userService.userDetails, { initialValue: null });
  readonly projects = signal<Project[]>([]);
  readonly projectsError = signal(false);
  readonly sidebarOpen = signal(false);
  readonly menuOpen = signal(false);
  readonly switcherOpen = signal(false);

  readonly isAdmin = computed(() => !!this.user()?.is_admin);
  readonly activeProject = computed(() => {
    const id = this.user()?.active_project_id;
    return this.projects().find((p) => p.id === id) ?? null;
  });

  constructor() {
    this.router.events
      .pipe(filter((event) => event instanceof NavigationEnd))
      .subscribe(() => this.closeSidebar());
  }

  ngOnInit(): void {
    this.loadProjects();
    this.projectService.changes.subscribe(() => this.loadProjects());
  }

  loadProjects(): void {
    this.projectService.getAll().subscribe({
      next: (projects) => {
        this.projects.set(projects);
        this.projectsError.set(false);
      },
      error: () => this.projectsError.set(true),
    });
  }

  /** Where a project's runs are: `host:port / schema`, or the SQLite file. */
  database(project: Project): string {
    if (project.db_type === 'sqlite') {
      return project.db_name?.split('/').pop() || 'SQLite';
    }
    const server = [project.db_host, project.db_port].filter(Boolean).join(':');
    return [server, project.db_schema].filter(Boolean).join(' / ');
  }

  toggleSwitcher(): void {
    if (!this.switcherOpen()) this.loadProjects();
    this.switcherOpen.update((open) => !open);
  }

  chooseProject(project: Project): void {
    this.switcherOpen.set(false);
    if (project.id !== this.user()?.active_project_id) {
      this.userService.setActiveProject(project.id);
    }
  }

  toggleMenu(): void {
    this.menuOpen.update((open) => !open);
  }

  /** Close a sheet when focus leaves the element that holds it. */
  closeOnFocusOut(event: FocusEvent, close: () => void): void {
    const container = event.currentTarget as HTMLElement;
    if (!container.contains(event.relatedTarget as Node | null)) close();
  }

  closeMenu(): void {
    this.menuOpen.set(false);
  }

  closeSwitcher(): void {
    this.switcherOpen.set(false);
  }

  openProfile(): void {
    this.menuOpen.set(false);
    this.router.navigate(['/profile']);
  }

  newProject(): void {
    this.switcherOpen.set(false);
    this.router.navigate(['/projects/new']);
  }

  signOut(): void {
    this.menuOpen.set(false);
    this.userService.logout();
  }

  toggleSidebar(): void {
    this.sidebarOpen.update((open) => !open);
  }

  closeSidebar(): void {
    this.sidebarOpen.set(false);
  }
}
