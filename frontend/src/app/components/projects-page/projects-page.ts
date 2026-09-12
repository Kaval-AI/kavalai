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
import { Component, OnInit, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Router } from '@angular/router';
import { ProjectService } from '../../services/project-service';
import { AgentService } from '../../services/agent-service';
import { Project } from '../../models/project';
import { UserService } from '../../services/user-service';
import { ToastService } from '../../services/toast-service';
import { UserDetails } from '../../models/user-details';
import { BaseChartDirective } from 'ng2-charts';
import { ChartConfiguration, ChartOptions, Chart, registerables } from 'chart.js';
import { NavigationService } from '../../services/navigation-service';

Chart.register(...registerables);

@Component({
  selector: 'app-projects-page',
  standalone: true,
  imports: [CommonModule, FormsModule, BaseChartDirective],
  templateUrl: './projects-page.html',
  styleUrl: './projects-page.css',
})
export class ProjectsPage implements OnInit {
  private projectService = inject(ProjectService);
  private agentService = inject(AgentService);
  private userService = inject(UserService);
  private router = inject(Router);
  private navigationService = inject(NavigationService);
  private toastService = inject(ToastService);

  projects: Project[] = [];
  selectedProject: Project | null = null;
  summaryStats: any = null;
  dailyStats: any = null;
  dbConnectionError: string | null = null;

  // Chart state
  public activityChartData: ChartConfiguration<'line'>['data'] = {
    datasets: [],
    labels: []
  };
  public tokensChartData: ChartConfiguration<'line'>['data'] = {
    datasets: [],
    labels: []
  };
  public durationsChartData: ChartConfiguration<'line'>['data'] = {
    datasets: [],
    labels: []
  };
  public runtimesChartData: ChartConfiguration<'line'>['data'] = {
    datasets: [],
    labels: []
  };
  public chartOptions: ChartOptions<'line'> = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      y: {
        beginAtZero: true,
        ticks: {
          stepSize: 1
        }
      }
    },
    plugins: {
      legend: {
        display: true,
        position: 'top',
        labels: {
          boxWidth: 10,
          usePointStyle: true,
          pointStyle: 'circle'
        }
      },
      tooltip: {
        mode: 'index',
        intersect: false
      }
    }
  };

  // Editing state
  isEditing = false;
  editableProject: Partial<Project> = {};

  // Deletion state
  showDeleteModal = false;
  deleteConfirmationName = '';
  isDeleting = false;

  // User management state
  projectMembers: any[] = [];
  allUsers: UserDetails[] = [];
  showAddUserModal = false;
  showRemoveMemberModal = false;
  memberToRemove: any = null;
  newMemberUserId = '';
  newMemberRole: 'owner' | 'viewer' = 'viewer';

  // Section collapse states
  accessDetailsCollapsed = true;
  connectionStringsCollapsed = true; // Default to collapsed for connection strings as they are often long
  statsCollapsed = false;

  // Password visibility
  showPassword = false;
  showConnectionPassword = false;

  ngOnInit() {
    this.navigationService.setTitle('Project info');
    this.userService.userDetails.subscribe(details => {
      if (details && details.active_project_id) {
        const newActiveProjectId = details.active_project_id !== 'None' ? details.active_project_id : null;

        // If active project changed and we already have projects loaded, update selection
        if (this.projects.length > 0 && newActiveProjectId) {
          const projectToSelect = this.projects.find(p => p.id === newActiveProjectId);
          if (projectToSelect && projectToSelect.id !== this.selectedProject?.id) {
            this.selectProject(projectToSelect);
          }
        } else {
          // Initial load or no projects yet
          this.loadProjects();
        }
      } else if (details) {
        this.loadProjects();
      }
    });
  }

  loadProjects() {
    this.projectService.getAll().subscribe({
      next: (data: Project[]) => {
        this.projects = data;
        this.dbConnectionError = null;
        // If we already have a selected project, keep it selected (and updated)
        // otherwise, grab the active one from user service, or the first one.
        const activeProjectId = this.userService.getActiveProjectId();
        const toSelect = this.projects.find(p => p.id === this.selectedProject?.id) ||
                         (activeProjectId ? this.projects.find(p => p.id === activeProjectId) : null) ||
                         (this.projects.length > 0 ? this.projects[0] : null);

        this.selectProject(toSelect);
      },
      error: (err) => {
        if (err.status === 503) {
          this.dbConnectionError = err.error?.detail || 'Backoffice database is not connected.';
          this.projects = [];
        } else {
          console.error('Failed to load projects:', err);
        }
      }
    });
  }

  getIsAdmin() {
    return this.userService.getIsAdmin();
  }

  selectProject(project: Project | null) {
    this.selectedProject = project;
    this.summaryStats = null;
    this.dbConnectionError = null;
    this.isEditing = false;
    if (project) {
      this.loadSummaryStats(project.id);
      this.loadProjectMembers(project.id);
    }
  }

  loadSummaryStats(projectId: string) {
    this.agentService.getSummaryStats(projectId).subscribe({
      next: (stats) => {
        this.summaryStats = stats;
        this.dbConnectionError = null;
      },
      error: (err) => {
        if (err.status === 503) {
          this.dbConnectionError = err.error?.detail || 'Database is not connected.';
        } else {
          console.error('Failed to load summary stats:', err);
        }
      }
    });

    this.agentService.getDailyStats(projectId, 7).subscribe({
      next: (stats) => {
        this.dailyStats = stats;
        this.prepareChartData();
      },
      error: (err) => {
        console.error('Failed to load daily stats', err);
        this.dailyStats = null;
      },
    });
  }

  setChartStat(stat: any): void {
    this.prepareChartData();
  }

  getEmbeddingBatchSize(): number {
    if (!this.dailyStats || !this.dailyStats.embedding) return 0;
    let total = 0;
    for (const model of Object.values(this.dailyStats.embedding) as any[]) {
      for (const day of model) {
        total += day.batch_size || 0;
      }
    }
    return total;
  }

  private prepareChartData(): void {
    if (!this.dailyStats) return;

    const labels = this.dailyStats.sessions.map((d: any) => {
      const date = new Date(d.date);
      const day = String(date.getDate()).padStart(2, '0');
      const month = String(date.getMonth() + 1).padStart(2, '0');
      return `${day}-${month}`;
    });

    const colors = {
      sessions: '#acc12f',
      messages: '#1b998b',
      tasks: '#b185a7',
      runs: '#6d9999',
      input: '#e67f0d',
      output: '#82204a',
      embedding: '#809537',
      llm_duration: '#4a90e2',
      embedding_duration: '#50e3c2'
    };

    const palette = [
      '#acc12f', '#1b998b', '#b185a7', '#6d9999', '#e67f0d', '#82204a', '#809537',
      '#4a90e2', '#50e3c2', '#f5a623', '#bd10e0', '#9013fe'
    ];

    // Activity Chart: sessions, messages, tasks, runs (total)
    const activityDatasets = [
      {
        data: this.dailyStats.sessions.map((d: any) => d.count),
        label: 'Sessions',
        borderColor: colors.sessions,
        backgroundColor: colors.sessions,
        fill: false,
        tension: 0.1,
        borderWidth: 2,
        pointRadius: 3,
        pointHoverRadius: 5
      },
      {
        data: this.dailyStats.messages.map((d: any) => d.count),
        label: 'Messages',
        borderColor: colors.messages,
        backgroundColor: colors.messages,
        fill: false,
        tension: 0.1,
        borderWidth: 2,
        pointRadius: 3,
        pointHoverRadius: 5
      },
      {
        data: this.dailyStats.tasks.map((d: any) => d.count),
        label: 'Tasks',
        borderColor: colors.tasks,
        backgroundColor: colors.tasks,
        fill: false,
        tension: 0.1,
        borderWidth: 2,
        pointRadius: 3,
        pointHoverRadius: 5
      },
      {
        data: labels.map((_: any, i: number) => {
          let total = 0;
          for (const agent of Object.values(this.dailyStats.runs) as any[]) {
            total += agent[i]?.count || 0;
          }
          return total;
        }),
        label: 'Workflow Runs',
        borderColor: colors.runs,
        backgroundColor: colors.runs,
        fill: false,
        tension: 0.1,
        borderWidth: 2,
        pointRadius: 3,
        pointHoverRadius: 5
      }
    ];

    this.activityChartData = {
      labels: labels,
      datasets: activityDatasets
    };

    // Tokens Chart: input, output, embedding
    // We sum up tokens from all models for each day
    const tokenDatasets = [
      {
        data: labels.map((_: any, i: number) => {
          let total = 0;
          for (const model of Object.values(this.dailyStats.llm) as any[]) {
            total += model[i]?.prompt_tokens || 0;
          }
          return total;
        }),
        label: 'Input Tokens',
        borderColor: colors.input,
        backgroundColor: colors.input,
        fill: false,
        tension: 0.1,
        borderWidth: 2,
        pointRadius: 3,
        pointHoverRadius: 5
      },
      {
        data: labels.map((_: any, i: number) => {
          let total = 0;
          for (const model of Object.values(this.dailyStats.llm) as any[]) {
            total += model[i]?.completion_tokens || 0;
          }
          return total;
        }),
        label: 'Output Tokens',
        borderColor: colors.output,
        backgroundColor: colors.output,
        fill: false,
        tension: 0.1,
        borderWidth: 2,
        pointRadius: 3,
        pointHoverRadius: 5
      },
      {
        data: labels.map((_: any, i: number) => {
          let total = 0;
          for (const model of Object.values(this.dailyStats.embedding) as any[]) {
            total += model[i]?.total_tokens || 0;
          }
          return total;
        }),
        label: 'Embedding Tokens',
        borderColor: colors.embedding,
        backgroundColor: colors.embedding,
        fill: false,
        tension: 0.1,
        borderWidth: 2,
        pointRadius: 3,
        pointHoverRadius: 5
      }
    ];

    this.tokensChartData = {
      labels: labels,
      datasets: tokenDatasets
    };

    // Durations Chart: avg llm and embedding durations
    const durationDatasets = [
      {
        data: labels.map((_: any, i: number) => {
          let totalDuration = 0;
          let totalCount = 0;
          for (const model of Object.values(this.dailyStats.llm) as any[]) {
            totalDuration += model[i]?.duration_seconds || 0;
            totalCount += model[i]?.count || 0;
          }
          return totalCount > 0 ? totalDuration / totalCount : 0;
        }),
        label: 'Avg LLM Duration (s)',
        borderColor: colors.llm_duration,
        backgroundColor: colors.llm_duration,
        fill: false,
        tension: 0.1,
        borderWidth: 2,
        pointRadius: 3,
        pointHoverRadius: 5
      },
      {
        data: labels.map((_: any, i: number) => {
          let totalDuration = 0;
          let totalCount = 0;
          for (const model of Object.values(this.dailyStats.embedding) as any[]) {
            totalDuration += model[i]?.duration_seconds || 0;
            totalCount += model[i]?.count || 0;
          }
          return totalCount > 0 ? totalDuration / totalCount : 0;
        }),
        label: 'Avg Embedding Duration (s)',
        borderColor: colors.embedding_duration,
        backgroundColor: colors.embedding_duration,
        fill: false,
        tension: 0.1,
        borderWidth: 2,
        pointRadius: 3,
        pointHoverRadius: 5
      }
    ];

    this.durationsChartData = {
      labels: labels,
      datasets: durationDatasets
    };

    // Runtimes Chart: total workflow runtimes grouped by agent name
    const runtimeDatasets = Object.keys(this.dailyStats.runs).map((agentName, idx) => {
      const agentData = this.dailyStats.runs[agentName];
      return {
        data: agentData.map((d: any) => d.duration_seconds || 0),
        label: agentName,
        borderColor: palette[idx % palette.length],
        backgroundColor: palette[idx % palette.length],
        fill: false,
        tension: 0.1,
        borderWidth: 2,
        pointRadius: 3,
        pointHoverRadius: 5
      };
    });

    this.runtimesChartData = {
      labels: labels,
      datasets: runtimeDatasets
    };
  }

  loadProjectMembers(projectId: string) {
    this.projectService.getMembers(projectId).subscribe({
      next: (members) => {
        this.projectMembers = members;
      },
      error: (err) => {
        console.error('Failed to load project members:', err);
      }
    });
  }

  loadAllUsers() {
    this.userService.getUsers().subscribe({
      next: (users) => {
        this.allUsers = users;
      },
      error: (err) => {
        console.error('Failed to load users:', err);
      }
    });
  }

  editProject() {
    if (this.selectedProject) {
      this.editableProject = { ...this.selectedProject };
      this.isEditing = true;
      this.accessDetailsCollapsed = false;
    }
  }

  cancelEdit() {
    this.isEditing = false;
    this.editableProject = {};
  }

  saveProject() {
    if (this.selectedProject && this.selectedProject.id) {
      this.projectService.update(this.selectedProject.id, this.editableProject).subscribe({
        next: (updated) => {
          this.selectedProject = { ...this.selectedProject, ...updated } as Project;
          this.isEditing = false;
          // Refresh list to show updated name if it changed
          this.loadProjects();
        },
        error: (err) => {
          console.error('Failed to update project:', err);
        }
      });
    }
  }

  confirmDeleteProject() {
    this.deleteConfirmationName = '';
    this.showDeleteModal = true;
  }

  closeDeleteModal() {
    this.showDeleteModal = false;
    this.deleteConfirmationName = '';
  }

  deleteProject() {
    if (this.selectedProject?.id && this.deleteConfirmationName === this.selectedProject.name) {
      this.isDeleting = true;
      this.projectService.delete(this.selectedProject.id).subscribe({
        next: () => {
          this.isDeleting = false;
          this.showDeleteModal = false;
          this.selectedProject = null;
          this.loadProjects();
        },
        error: (err) => {
          this.isDeleting = false;
          console.error('Failed to delete project:', err);
        }
      });
    }
  }

  openAddUserModal() {
    this.loadAllUsers();
    this.newMemberUserId = '';
    this.newMemberRole = 'viewer';
    this.showAddUserModal = true;
  }

  closeAddUserModal() {
    this.showAddUserModal = false;
  }

  addMember() {
    if (this.selectedProject && this.newMemberUserId) {
      this.projectService.addMember(this.selectedProject.id, this.newMemberUserId, this.newMemberRole).subscribe({
        next: () => {
          this.loadProjectMembers(this.selectedProject!.id);
          this.toastService.success('User added to project');
          this.closeAddUserModal();
        },
        error: (err) => {
          console.error('Failed to add member:', err);
          const errorMessage = err.error?.detail || 'Failed to add user to project';
          this.toastService.error(errorMessage);
        }
      });
    }
  }

  updateMemberRole(userId: string, newRole: string, event?: MouseEvent) {
    if (this.selectedProject) {
      this.projectService.updateMember(this.selectedProject.id, userId, newRole).subscribe({
        next: () => {
          this.loadProjectMembers(this.selectedProject!.id);
          this.toastService.success('Role updated');
          if (event) {
            (document.activeElement as HTMLElement).blur();
          }
        },
        error: (err) => {
          console.error('Failed to update member role:', err);
          const errorMessage = err.error?.detail || 'Failed to update member role';
          let position: { x: number; y: number } | undefined;
          if (event) {
            position = { x: event.clientX, y: event.clientY };
            (document.activeElement as HTMLElement).blur();
          }
          this.toastService.error(errorMessage, 3000, position);
        }
      });
    }
  }

  confirmRemoveMember(member: any) {
    this.memberToRemove = member;
    this.showRemoveMemberModal = true;
  }

  closeRemoveMemberModal() {
    this.showRemoveMemberModal = false;
    this.memberToRemove = null;
  }

  removeMember() {
    if (this.selectedProject && this.memberToRemove) {
      this.projectService.removeMember(this.selectedProject.id, this.memberToRemove.id).subscribe({
        next: () => {
          this.loadProjectMembers(this.selectedProject!.id);
          this.toastService.success('User removed from project');
          this.closeRemoveMemberModal();
        },
        error: (err) => {
          console.error('Failed to remove member:', err);
          const errorMessage = err.error?.detail || 'Failed to remove user from project';
          this.toastService.error(errorMessage);
        }
      });
    }
  }

  startNewProject() {
    this.router.navigate(['/project-edit', 'new']);
  }

  onProjectSelect(event: Event) {
    const target = event.target as HTMLSelectElement;
    const projectId = target.value;
    if (projectId) {
      this.userService.setActiveProject(projectId);
      const project = this.projects.find(p => p.id === projectId);
      if (project) {
        this.selectProject(project);
      }
    }
  }

  getDbUri(type: 'postgresql' | 'asyncpg' | 'jdbc' | 'env', maskPassword = true): string {
    if (!this.selectedProject) return '';
    const p = this.selectedProject;
    const user = p.db_user || 'user';
    const password = p.db_password || (maskPassword ? '••••••••' : 'password');
    const displayPassword = maskPassword ? '••••••••' : password;
    const host = p.db_host || 'localhost';
    const port = p.db_port || 5432;
    const dbName = p.db_name || 'kavalai';

    if (p.db_type === 'sqlite') {
      const uri = `sqlite:///${p.db_name || 'agents.db'}`;
      switch (type) {
        case 'postgresql':
          return uri;
        case 'asyncpg':
          return `sqlite+aiosqlite:///${p.db_name || 'agents.db'}`;
        case 'jdbc':
          return `jdbc:sqlite:${p.db_name || 'agents.db'}`;
        case 'env':
          return `KAVALAI_DB_URI=${uri}`;
        default:
          return '';
      }
    }

    switch (type) {
      case 'postgresql':
        return `postgresql://${user}:${displayPassword}@${host}:${port}/${dbName}`;
      case 'asyncpg':
        return `postgresql+asyncpg://${user}:${displayPassword}@${host}:${port}/${dbName}`;
      case 'jdbc':
        return `jdbc:postgresql://${host}:${port}/${dbName}?user=${user}&password=${displayPassword}`;
      case 'env':
        return `KAVALAI_DB_URI=postgresql://${user}:${displayPassword}@${host}:${port}/${dbName}`;
      default:
        return '';
    }
  }

  copyToClipboard(text: string, event?: MouseEvent) {
    navigator.clipboard.writeText(text).then(() => {
      let position: { x: number; y: number } | undefined;
      if (event) {
        position = { x: event.clientX, y: event.clientY };
      }
      this.toastService.success('Copied to clipboard', 1500, position);
    });
  }
}
