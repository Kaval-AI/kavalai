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
import { NEVER, of, throwError } from 'rxjs';
import { WorkflowGraphComponent } from './workflow-graph';
import { AgentService } from '../../services/agent-service';
import { settle } from '../../../testing';

describe('WorkflowGraphComponent', () => {
  let fixture: ComponentFixture<WorkflowGraphComponent>;
  let el: HTMLElement;
  let agentService: jasmine.SpyObj<AgentService>;

  const workflow = {
    nodes: [
      { name: 'start', type: 'start', next: 'reply' },
      { name: 'reply', type: 'llm', next: 'end' },
      { name: 'end', type: 'end' },
    ],
  };

  function query(testId: string): HTMLElement | null {
    return el.querySelector<HTMLElement>(`[data-testid="${testId}"]`);
  }

  async function render(value: unknown): Promise<void> {
    fixture = TestBed.createComponent(WorkflowGraphComponent);
    el = fixture.nativeElement;
    fixture.componentRef.setInput('workflow', value);
    fixture.detectChanges();
    await settle(fixture);
  }

  beforeEach(async () => {
    agentService = jasmine.createSpyObj('AgentService', ['renderWorkflowSvg']);
    await TestBed.configureTestingModule({
      imports: [WorkflowGraphComponent],
      providers: [{ provide: AgentService, useValue: agentService }],
    }).compileComponents();
  });

  it('draws the SVG the backend renders for the workflow', async () => {
    agentService.renderWorkflowSvg.and.returnValue(of('<svg><text>reply</text></svg>'));
    await render(workflow);
    expect(agentService.renderWorkflowSvg).toHaveBeenCalledWith(workflow);
    expect(query('graph')!.querySelector('svg text')!.textContent).toBe('reply');
  });

  it('shows a spinner while the backend draws', async () => {
    agentService.renderWorkflowSvg.and.returnValue(NEVER);
    await render(workflow);
    expect(el.querySelector('ui-loading')).not.toBeNull();
    expect(query('graph')).toBeNull();
  });

  it("shows the server's error with Retry, which draws again", async () => {
    agentService.renderWorkflowSvg.and.returnValue(
      throwError(() => new HttpErrorResponse({ status: 422, error: { detail: 'Unknown node type' } }))
    );
    await render(workflow);
    expect(query('graph-error')!.textContent).toContain('Unknown node type');

    agentService.renderWorkflowSvg.and.returnValue(of('<svg></svg>'));
    query('graph-retry')!.click();
    await settle(fixture);
    expect(query('graph-error')).toBeNull();
    expect(query('graph')!.querySelector('svg')).not.toBeNull();
  });

  it('says when there is no workflow, without calling the backend', async () => {
    await render(null);
    expect(query('graph-none')!.textContent).toContain('no workflow recorded');
    expect(agentService.renderWorkflowSvg).not.toHaveBeenCalled();
  });

  it('draws again when the workflow changes', async () => {
    agentService.renderWorkflowSvg.and.returnValue(of('<svg></svg>'));
    await render(workflow);
    fixture.componentRef.setInput('workflow', { nodes: [] });
    fixture.detectChanges();
    await settle(fixture);
    expect(agentService.renderWorkflowSvg).toHaveBeenCalledTimes(2);
  });
});
