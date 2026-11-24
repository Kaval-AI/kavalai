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
import { Component, DestroyRef, OnChanges, inject, input, signal } from '@angular/core';
import { DomSanitizer, SafeHtml } from '@angular/platform-browser';
import { Subscription } from 'rxjs';
import { AgentService } from '../../services/agent-service';
import { errorMessage } from '../../core/http-error';
import { UiAlert, UiButton, UiLoading } from '../../ui';

/**
 * Renders a workflow graph (its `nodes` + transitions) as an SVG diagram. The
 * SVG is generated on the backend from the agent's stored `workflow` JSON
 * (`kavalai.workflow.render_workflow_svg`), so node names, the arrows that
 * connect them and branch conditions are drawn consistently with the rest of
 * Kaval.AI — the documentation and the SDK use the same renderer.
 */
@Component({
  selector: 'app-workflow-graph',
  imports: [UiAlert, UiButton, UiLoading],
  templateUrl: './workflow-graph.html',
  styleUrl: './workflow-graph.css',
})
export class WorkflowGraphComponent implements OnChanges {
  private readonly agentService = inject(AgentService);
  private readonly sanitizer = inject(DomSanitizer);

  readonly workflow = input<unknown>(null);
  readonly svg = signal<SafeHtml | null>(null);
  readonly loading = signal(false);
  readonly error = signal<string | null>(null);

  private subscription: Subscription | null = null;

  constructor() {
    inject(DestroyRef).onDestroy(() => this.subscription?.unsubscribe());
  }

  ngOnChanges(): void {
    this.render();
  }

  render(): void {
    this.subscription?.unsubscribe();
    this.error.set(null);
    this.svg.set(null);
    const workflow = this.workflow();
    if (!workflow) return;
    this.loading.set(true);
    this.subscription = this.agentService.renderWorkflowSvg(workflow).subscribe({
      next: (svg) => {
        this.svg.set(this.sanitizer.bypassSecurityTrustHtml(svg));
        this.loading.set(false);
      },
      error: (err) => {
        this.error.set(errorMessage(err, 'Could not draw the workflow graph.'));
        this.loading.set(false);
      },
    });
  }
}
