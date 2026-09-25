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
import { AgentService } from '../../services/agent-service';
import { UserService } from '../../services/user-service';
import { NavigationService } from '../../services/navigation-service';
import { Agent } from '../../models/agent';
import { SessionSummary, SessionDetails } from '../../models/session';
import { Run } from '../../models/run';
import { Task } from '../../models/task';
import { forkJoin, of } from 'rxjs';
import { catchError, map, switchMap } from 'rxjs/operators';

interface AgentWorkflow {
  agent: Agent;
  lanes: Lane[];
}

interface Lane {
  runs: RunWorkflow[];
}

interface SessionWorkflow {
  sessionId: string;
  runs: RunWorkflow[];
}

interface RunWorkflow {
  run: Run;
  sessionId: string;
  tasks: TaskWorkflow[];
  status: 'success' | 'error' | 'running';
}

interface TaskWorkflow {
  task: Task;
  status: 'success' | 'error' | 'running';
}

@Component({
  selector: 'app-workflows-page',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './workflows-page.html',
  styleUrl: './workflows-page.css',
})
export class WorkflowsPage implements OnInit {
  private agentService = inject(AgentService);
  private userService = inject(UserService);
  private navigationService = inject(NavigationService);
  private router = inject(Router);

  activeProjectId: string | null = null;
  agents: Agent[] = [];
  workflows: AgentWorkflow[] = [];
  loading: boolean = false;
  error: string | null = null;
  selectedTask: TaskWorkflow | null = null;
  selectedSessionId: string | null = null;

  ngOnInit(): void {
    this.navigationService.setBreadcrumbs([{ label: 'Workflows' }]);
    this.userService.userDetails.subscribe(user => {
      if (user && user.active_project_id) {
        const newProjectId = user.active_project_id !== 'None' ? user.active_project_id : null;
        if (newProjectId !== this.activeProjectId) {
          this.activeProjectId = newProjectId;
          this.loadWorkflows();
        }
      }
    });
  }

  loadWorkflows(): void {
    if (!this.activeProjectId) {
      this.error = 'No active project selected';
      return;
    }

    this.loading = true;
    this.error = null;

    this.agentService.getAgentsByProject(this.activeProjectId).pipe(
      switchMap(agents => {
        this.agents = agents;
        if (agents.length === 0) return of([]);

        const agentRequests = agents.map(agent =>
          this.agentService.getSessions(this.activeProjectId!, agent.id, undefined, undefined, undefined, 10, 0).pipe(
            switchMap(sessionData => {
              const sessions: SessionSummary[] = sessionData.sessions;
              if (sessions.length === 0) return of({ agent, sessions: [] });

              const sessionDetailsRequests = sessions.map(s =>
                this.agentService.getSessionDetails(this.activeProjectId!, s.session_id).pipe(
                  map(details => ({
                    sessionId: s.session_id,
                    runs: details.runs.map(run => ({
                      run,
                      sessionId: s.session_id,
                      tasks: details.tasks
                        .filter(t => t.run_id === run.id)
                        .map(t => ({
                          task: t,
                          status: (t.errors && t.errors.length > 0) ? 'error' as const : 'success' as const
                        })),
                      status: this.getRunStatus(details.tasks.filter(t => t.run_id === run.id))
                    }))
                  })),
                  catchError(() => of({ sessionId: s.session_id, runs: [] }))
                )
              );

              return forkJoin(sessionDetailsRequests).pipe(
                map(sessionWorkflows => ({
                  agent,
                  lanes: this.organizeIntoLanes(sessionWorkflows.flatMap(sw => sw.runs))
                }))
              );
            }),
            catchError(() => of({ agent, sessions: [] }))
          )
        );

        return forkJoin(agentRequests);
      })
    ).subscribe({
      next: (data: any) => {
        this.workflows = data;
        this.loading = false;
      },
      error: (err) => {
        this.error = 'Failed to load workflows';
        console.error(err);
        this.loading = false;
      }
    });
  }

  getRunStatus(tasks: Task[]): 'success' | 'error' | 'running' {
    if (tasks.length === 0) return 'running';
    const hasError = tasks.some(t => t.errors && t.errors.length > 0);
    return hasError ? 'error' : 'success';
  }

  organizeIntoLanes(runs: RunWorkflow[]): Lane[] {
    // Sort runs by created_at
    const sortedRuns = [...runs].sort((a, b) =>
      new Date(a.run.created_at).getTime() - new Date(b.run.created_at).getTime()
    );

    const lanes: Lane[] = [];

    sortedRuns.forEach(run => {
      let placed = false;
      const runStart = new Date(run.run.created_at).getTime();

      for (const lane of lanes) {
        const lastRunInLane = lane.runs[lane.runs.length - 1];
        const lastRunEnd = this.runEnd(lastRunInLane.run);

        if (runStart > lastRunEnd) {
          lane.runs.push(run);
          placed = true;
          break;
        }
      }

      if (!placed) {
        lanes.push({ runs: [run] });
      }
    });

    return lanes;
  }

  /**
   * When a run ended, for the lane overlap check. A run that recorded no
   * duration (still in progress, or written before runs carried one) is
   * assumed to have taken a minute.
   */
  runEnd(run: Run): number {
    const start = new Date(run.created_at).getTime();
    const seconds = run.duration_seconds ?? 60;
    return start + seconds * 1000;
  }

  selectTask(task: TaskWorkflow, sessionId: string): void {
    this.selectedTask = task;
    this.selectedSessionId = sessionId;
  }

  closeTaskOverview(): void {
    this.selectedTask = null;
    this.selectedSessionId = null;
  }

  openTaskInNewTab(): void {
    if (this.selectedTask && this.selectedSessionId) {
      const url = this.router.serializeUrl(
        this.router.createUrlTree(['/conversations', this.selectedSessionId], {
          fragment: `task-${this.selectedTask.task.id}`
        })
      );
      window.open(url, '_blank');
    }
  }

  viewSession(sessionId: string): void {
    this.router.navigate(['/conversations', sessionId]);
  }

  viewRunTasks(sessionId: string, runId: string): void {
    this.router.navigate(['/conversations', sessionId, 'runs', runId, 'tasks']);
  }

  formatDate(dateStr: string): string {
    const date = new Date(dateStr);
    return date.toLocaleString();
  }
}
