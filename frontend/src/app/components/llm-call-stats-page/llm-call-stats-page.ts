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
import { Component, OnDestroy, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ActivatedRoute, RouterModule } from '@angular/router';
import { Subscription } from 'rxjs';
import { AgentService } from '../../services/agent-service';
import { UserService } from '../../services/user-service';
import { LLMCallStat } from '../../models/llm-call-stat';
import { JsonTreeComponent } from '../json-tree/json-tree';

/**
 * Lists model calls, newest first. The `call_type`, `session_id` and `run_id`
 * query parameters narrow the list, so a run's calls are one link away from
 * the run and from any call it made.
 */
@Component({
  selector: 'app-llm-call-stats-page',
  standalone: true,
  imports: [CommonModule, RouterModule, JsonTreeComponent],
  templateUrl: './llm-call-stats-page.html',
  styleUrl: './llm-call-stats-page.css'
})
export class LlmCallStatsPage implements OnInit, OnDestroy {
  callType: string | null = null;
  sessionId: string | null = null;
  runId: string | null = null;
  projectId: string | null = null;
  stats: LLMCallStat[] = [];
  loading = false;
  error: string | null = null;
  limit = 20;
  offset = 0;
  hasMore = true;

  private subscriptions = new Subscription();
  private loadSubscription: Subscription | null = null;

  constructor(
    private route: ActivatedRoute,
    private agentService: AgentService,
    private userService: UserService
  ) {}

  ngOnInit(): void {
    this.subscriptions.add(this.route.queryParams.subscribe(params => {
      this.callType = params['call_type'] || null;
      this.sessionId = params['session_id'] || null;
      this.runId = params['run_id'] || null;
      this.reload();
    }));
    this.subscriptions.add(this.userService.userDetails.subscribe(user => {
      const projectId = user?.active_project_id && user.active_project_id !== 'None'
        ? user.active_project_id
        : null;
      if (projectId !== this.projectId) {
        this.projectId = projectId;
        this.reload();
      }
    }));
  }

  ngOnDestroy(): void {
    this.subscriptions.unsubscribe();
    this.loadSubscription?.unsubscribe();
  }

  get isFiltered(): boolean {
    return !!(this.callType || this.sessionId || this.runId);
  }

  reload(): void {
    this.offset = 0;
    this.stats = [];
    this.hasMore = true;
    this.loadStats();
  }

  loadStats(): void {
    if (!this.projectId) return;

    // A filter change supersedes a page still in flight.
    this.loadSubscription?.unsubscribe();
    this.loading = true;
    this.error = null;
    this.loadSubscription = this.agentService.getLLMCallStats(
      this.projectId,
      this.callType || undefined,
      this.limit,
      this.offset,
      { sessionId: this.sessionId || undefined, runId: this.runId || undefined }
    ).subscribe({
      next: (data) => {
        this.stats = [...this.stats, ...data];
        this.loading = false;
        this.hasMore = data.length === this.limit;
      },
      error: (err) => {
        this.error = 'Failed to load model calls';
        this.loading = false;
        console.error(err);
      }
    });
  }

  loadMore(): void {
    if (this.loading || !this.hasMore) return;
    this.offset += this.limit;
    this.loadStats();
  }

  formatDate(dateStr: string): string {
    return new Date(dateStr).toLocaleString();
  }
}
