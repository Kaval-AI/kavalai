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
import { ActivatedRoute, Router, RouterModule } from '@angular/router';
import { AgentService } from '../../services/agent-service';
import { UserService } from '../../services/user-service';
import { Task } from '../../models/task';
import { LLMCallStat } from '../../models/llm-call-stat';
import { TasksList } from '../tasks-list/tasks-list';
import { TaskTimelineChart } from '../task-timeline-chart/task-timeline-chart';
import { NavigationService } from '../../services/navigation-service';

@Component({
  selector: 'app-run-tasks-page',
  standalone: true,
  imports: [RouterModule, TasksList, TaskTimelineChart],
  templateUrl: './run-tasks-page.html',
  styleUrl: './run-tasks-page.css',
})
export class RunTasksPage implements OnInit {
  private route = inject(ActivatedRoute);
  private router = inject(Router);
  private agentService = inject(AgentService);
  private userService = inject(UserService);
  private navigationService = inject(NavigationService);

  sessionId: string | null = null;
  runId: string | null = null;
  projectId: string | null = null;
  tasks: Task[] = [];
  /** The run's model calls, oldest first; at most `modelCallsLimit` of them. */
  modelCalls: LLMCallStat[] = [];
  readonly modelCallsLimit = 100;
  loading: boolean = false;
  error: string | null = null;

  ngOnInit(): void {
    this.route.paramMap.subscribe((params) => {
      this.sessionId = params.get('sessionId');
      this.runId = params.get('runId');
      this.navigationService.setBreadcrumbs([
        { label: 'Conversations', link: '/conversations' },
        { label: this.sessionId || 'Session', link: `/conversations/${this.sessionId}` },
        { label: 'Tasks' }
      ]);
      if (this.sessionId && this.runId) {
        this.tryLoad();
      }
    });

    this.userService.userDetails.subscribe((user) => {
      if (user && user.active_project_id) {
        const newProjectId = user.active_project_id !== 'None' ? user.active_project_id : null;
        if (newProjectId !== this.projectId) {
          this.projectId = newProjectId;
          this.tryLoad();
        }
      }
    });
  }

  private tryLoad(): void {
    if (!this.projectId || !this.sessionId || !this.runId) return;
    this.loadTasks();
    this.loadModelCalls();
  }

  private loadModelCalls(): void {
    if (!this.projectId || !this.runId) return;
    this.agentService
      .getLLMCallStats(this.projectId, undefined, this.modelCallsLimit, 0, { runId: this.runId })
      .subscribe({
        next: (calls) => {
          this.modelCalls = [...calls].reverse();
        },
        // The tasks are the page's subject; without the calls only their section is missing.
        error: (err) => {
          this.modelCalls = [];
          console.error(err);
        }
      });
  }

  get modelCallTokens(): number {
    return this.modelCalls.reduce((sum, call) => sum + (call.total_tokens || 0), 0);
  }

  formatDuration(seconds: number | null): string {
    return seconds === null ? '-' : `${(seconds * 1000).toFixed(0)} ms`;
  }

  formatTime(dateStr: string): string {
    return new Date(dateStr).toLocaleTimeString();
  }

  private loadTasks(): void {
    if (!this.projectId || !this.sessionId || !this.runId) return;
    this.loading = true;
    this.error = null;

    this.agentService.getSessionDetails(this.projectId, this.sessionId).subscribe({
      next: (details) => {
        this.tasks = details.tasks.filter(t => t.run_id === this.runId);
        this.loading = false;
      },
      error: (err) => {
        this.error = 'Failed to load tasks';
        console.error(err);
        this.loading = false;
      }
    });
  }

  goBack(): void {
    if (this.sessionId) {
      this.router.navigate(['/conversations', this.sessionId]);
    } else {
      this.router.navigate(['/conversations']);
    }
  }
}
