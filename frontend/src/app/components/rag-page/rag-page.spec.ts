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
import { BehaviorSubject, NEVER, Subject, of, throwError } from 'rxjs';
import { RagPage } from './rag-page';
import { RagService } from '../../services/rag-service';
import { UserService } from '../../services/user-service';
import { PcaData, RagCollection, RagResult } from '../../models/rag';
import { UserDetails } from '../../models/user-details';
import { makeUser, settle } from '../../../testing';

function makeCollection(overrides: Partial<RagCollection> = {}): RagCollection {
  return {
    name: 'default',
    model: 'openai/text-embedding-3-small',
    embedding_size: 1536,
    schema_version: 2,
    count: 120,
    ...overrides,
  };
}

function makeResult(overrides: Partial<RagResult> = {}): RagResult {
  return {
    id: 'e1',
    model: 'openai/text-embedding-3-small',
    collection_name: 'default',
    source_id: 'claims-handbook-2026-section-4',
    content: 'A claim is opened when the first notice of loss arrives.',
    embedding_size: 1536,
    rag_metadata: { page: 4 },
    created_at: '2026-11-24T10:00:00Z',
    updated_at: '2026-11-24T10:00:00Z',
    similarity: 0.8734,
    ...overrides,
  };
}

const PCA: PcaData = {
  query: { label: 'claims?', x: 0, y: 0 },
  results: [{ label: 'handbook', x: 1, y: 2 }],
  samples: [{ label: 'sample', x: -1, y: -2 }],
};

describe('RagPage', () => {
  let fixture: ComponentFixture<RagPage>;
  let el: HTMLElement;
  let ragService: jasmine.SpyObj<RagService>;
  let user: BehaviorSubject<UserDetails | null>;

  function query<T extends Element = HTMLElement>(testId: string): T | null {
    return el.querySelector<T>(`[data-testid="${testId}"]`);
  }

  async function render(): Promise<void> {
    fixture = TestBed.createComponent(RagPage);
    el = fixture.nativeElement;
    fixture.detectChanges();
    await settle(fixture);
  }

  async function type(testId: string, value: string): Promise<void> {
    const input = query<HTMLInputElement>(testId)!;
    input.value = value;
    input.dispatchEvent(new Event('input'));
    await settle(fixture);
  }

  async function search(text = 'How is a claim opened?'): Promise<void> {
    await type('rag-query', text);
    query<HTMLButtonElement>('rag-search')!.click();
    await settle(fixture);
  }

  beforeEach(async () => {
    ragService = jasmine.createSpyObj('RagService', ['getRagStats', 'getRagCollections', 'queryRag', 'trainPca']);
    ragService.getRagStats.and.returnValue(of({ total_entries: 1450, total_collections: 2, collections: [] }));
    ragService.getRagCollections.and.returnValue(
      of([makeCollection({ name: 'faq', model: 'fastembed/bge-small' }), makeCollection()])
    );
    ragService.queryRag.and.returnValue(of({ results: [makeResult()], pca_data: PCA }));
    user = new BehaviorSubject<UserDetails | null>(makeUser());

    await TestBed.configureTestingModule({
      imports: [RagPage],
      providers: [
        { provide: RagService, useValue: ragService },
        { provide: UserService, useValue: { userDetails: user.asObservable() } },
      ],
    }).compileComponents();
  });

  it('opens with its page header', async () => {
    await render();
    expect(el.querySelector('ui-page-header h1')!.textContent).toContain('RAG');
  });

  it('asks for a project when none is active', async () => {
    user.next(makeUser({ active_project_id: 'None' }));
    await render();
    expect(query('rag-no-project')).not.toBeNull();
    expect(ragService.getRagStats).not.toHaveBeenCalled();
  });

  it('waits for a signed-in user', async () => {
    user.next(null);
    await render();
    expect(ragService.getRagStats).not.toHaveBeenCalled();
  });

  it('shows a spinner while the collections load', async () => {
    ragService.getRagCollections.and.returnValue(NEVER);
    await render();
    expect(el.querySelector('ui-loading')).not.toBeNull();
  });

  it("shows the server's error with Retry, which loads again", async () => {
    ragService.getRagStats.and.returnValue(
      throwError(() => new HttpErrorResponse({ status: 500, error: { detail: 'pgvector is not installed' } }))
    );
    await render();
    expect(query('rag-error')!.textContent).toContain('pgvector is not installed');

    ragService.getRagStats.and.returnValue(of({ total_entries: 1, total_collections: 1, collections: [] }));
    query<HTMLButtonElement>('rag-retry')!.click();
    await settle(fixture);
    expect(query('rag-error')).toBeNull();
    expect(query('rag-form')).not.toBeNull();
  });

  it('says when the project has no collections', async () => {
    ragService.getRagCollections.and.returnValue(of([]));
    await render();
    expect(query('rag-empty')!.textContent).toContain('no RAG collection');
    expect(query('rag-form')).toBeNull();
  });

  it('shows the totals and preselects the default collection with its model', async () => {
    await render();
    expect(query('rag-stats')!.textContent).toContain('1,450');
    const select = query<HTMLSelectElement>('rag-collection')!;
    expect(select.options[select.selectedIndex].textContent).toContain('default (120)');
    expect(query('rag-model')!.textContent).toContain('openai/text-embedding-3-small');
  });

  it('preselects the first collection when there is no default one', async () => {
    ragService.getRagCollections.and.returnValue(of([makeCollection({ name: 'faq', model: 'fastembed/bge-small' })]));
    await render();
    expect(query('rag-model')!.textContent).toContain('fastembed/bge-small');
  });

  it('refuses an empty query', async () => {
    await render();
    expect(query<HTMLButtonElement>('rag-search')!.disabled).toBeTrue();
    await type('rag-query', '   ');
    query<HTMLFormElement>('rag-form')!.dispatchEvent(new Event('submit'));
    await settle(fixture);
    expect(ragService.queryRag).not.toHaveBeenCalled();
  });

  it('holds the Search button busy while the query runs', async () => {
    ragService.queryRag.and.returnValue(NEVER);
    await render();
    await search();
    const button = query<HTMLButtonElement>('rag-search')!;
    expect(button.disabled).toBeTrue();
    expect(button.querySelector('ui-spinner')).not.toBeNull();
  });

  it('sends the query with its options and lists the results with a similarity bar', async () => {
    await render();
    await type('rag-query', '  How is a claim opened?  ');
    const sourceIds = el.querySelector<HTMLInputElement>('#rag-source-ids')!;
    sourceIds.value = 'a, ,b';
    sourceIds.dispatchEvent(new Event('input'));
    const normalizer = el.querySelector<HTMLTextAreaElement>('#rag-normalizer')!;
    normalizer.value = 'l2: true';
    normalizer.dispatchEvent(new Event('input'));
    query<HTMLInputElement>('rag-keep-best')!.click();
    const topK = query<HTMLSelectElement>('rag-top-k')!;
    topK.selectedIndex = 0;
    topK.dispatchEvent(new Event('change'));
    await settle(fixture);
    query<HTMLButtonElement>('rag-search')!.click();
    await settle(fixture);

    expect(ragService.queryRag).toHaveBeenCalledWith('p1', {
      text: 'How is a claim opened?',
      collection_name: 'default',
      top_k: 5,
      source_ids: ['a', 'b'],
      keep_best: true,
      normalizer_yaml: 'l2: true',
    });
    const row = query('rag-result')!;
    expect(row.textContent).toContain('87%');
    expect(row.querySelector('ui-progress')!.getAttribute('aria-valuenow')).toBe('87');
    expect(row.querySelector('.rag-source')!.textContent!.trim()).toBe('claims-h…');
    expect(row.querySelector('.rag-source')!.getAttribute('title')).toBe('claims-handbook-2026-section-4');
    expect(row.textContent).toContain('"page": 4');
  });

  it('sends no source IDs or normalizer when none are given', async () => {
    await render();
    await search();
    const body = ragService.queryRag.calls.mostRecent().args[1];
    expect(body.source_ids).toBeUndefined();
    expect(body.normalizer_yaml).toBeUndefined();
  });

  it('draws the projection of the query, its matches and the sample', async () => {
    await render();
    await search();
    const projection = query('rag-projection')!;
    const circles = projection.querySelectorAll('circle');
    expect(circles.length).toBe(3);
    expect(circles[2].querySelector('title')!.textContent).toBe('Query: claims?');
    expect(Number(circles[1].getAttribute('cx'))).toBe(560);
    expect(Number(circles[0].getAttribute('cy'))).toBe(360);
  });

  it('draws a single point in the middle of the margin', async () => {
    ragService.queryRag.and.returnValue(
      of({ results: [makeResult()], pca_data: { query: { label: 'q', x: 3, y: 3 }, results: [], samples: [] } })
    );
    await render();
    await search();
    const circle = query('rag-projection')!.querySelector('circle')!;
    expect(circle.getAttribute('cx')).toBe('40');
    expect(circle.getAttribute('cy')).toBe('360');
  });

  it('says when the collection has no PCA model', async () => {
    ragService.queryRag.and.returnValue(of({ results: [makeResult({ rag_metadata: null })], pca_data: null }));
    await render();
    await search();
    expect(query('rag-no-pca')).not.toBeNull();
    expect(query('rag-projection')).toBeNull();
  });

  it('says when nothing matched', async () => {
    ragService.queryRag.and.returnValue(of({ results: [], pca_data: null }));
    await render();
    await search();
    expect(query('rag-no-results')!.textContent).toContain('Nothing matched.');
    expect(query('rag-no-pca')).toBeNull();
  });

  it("shows the server's refusal of a query", async () => {
    ragService.queryRag.and.returnValue(
      throwError(() => new HttpErrorResponse({ status: 400, error: { detail: 'Invalid normalizer YAML' } }))
    );
    await render();
    await search();
    expect(query('rag-query-error')!.textContent).toContain('Invalid normalizer YAML');
    expect(query<HTMLButtonElement>('rag-search')!.disabled).toBeFalse();
  });

  describe('PCA training', () => {
    let events: Subject<string>;

    beforeEach(() => {
      events = new Subject<string>();
      ragService.trainPca.and.returnValue(events.asObservable());
    });

    async function train(): Promise<HTMLDialogElement> {
      await render();
      query<HTMLButtonElement>('rag-train-pca')!.click();
      await settle(fixture);
      return el.querySelector<HTMLDialogElement>('ui-dialog dialog')!;
    }

    it('streams the log into a dialog until training completes', async () => {
      const dialog = await train();
      expect(ragService.trainPca).toHaveBeenCalledWith('p1', 'default');
      expect(dialog.open).toBeTrue();
      expect(query('pca-log')!.textContent).toContain('Waiting for the server');
      expect(query<HTMLButtonElement>('rag-train-pca')!.disabled).toBeTrue();
      expect(query('pca-close')!.textContent).toContain('Stop');

      events.next(JSON.stringify({ value: 'Sampling 1000 entries' }));
      events.next('plain text line');
      events.next(JSON.stringify({ type: 'complete', value: 'PCA training completed successfully' }));
      await settle(fixture);
      expect(query('pca-log')!.textContent).toContain('Sampling 1000 entries');
      expect(query('pca-log')!.textContent).toContain('plain text line');
      expect(query('pca-close')!.textContent).toContain('Close');
      expect(events.observed).toBeFalse();

      query<HTMLButtonElement>('pca-close')!.click();
      await settle(fixture);
      expect(dialog.open).toBeFalse();
    });

    it('stops when the server reports an error', async () => {
      await train();
      events.next(JSON.stringify({ status: 'error', value: 'Collection too small' }));
      await settle(fixture);
      expect(query('pca-log')!.textContent).toContain('Collection too small');
      expect(query('pca-close')!.textContent).toContain('Close');
    });

    it('says when the connection is lost', async () => {
      await train();
      events.error(new Event('error'));
      await settle(fixture);
      expect(query('pca-log')!.textContent).toContain('the connection to the server was lost');
      expect(query('pca-close')!.textContent).toContain('Close');
    });

    it('ignores a connection error after training has finished', async () => {
      await train();
      events.next(JSON.stringify({ type: 'error' }));
      events.error(new Event('error'));
      await settle(fixture);
      expect(query('pca-log')!.textContent).not.toContain('connection');
    });

    it('stops the stream when the dialog is dismissed', async () => {
      const dialog = await train();
      dialog.close();
      // The native close event is queued as a task of its own.
      await new Promise((resolve) => setTimeout(resolve, 20));
      await settle(fixture);
      expect(events.observed).toBeFalse();
      expect(query<HTMLButtonElement>('rag-train-pca')!.disabled).toBeFalse();
    });
  });

  it('starts over when the active project changes, and only then', async () => {
    await render();
    await search();
    user.next(makeUser());
    await settle(fixture);
    expect(ragService.getRagStats).toHaveBeenCalledTimes(1);

    user.next(makeUser({ active_project_id: 'p2' }));
    await settle(fixture);
    expect(ragService.getRagStats).toHaveBeenCalledWith('p2');
    expect(query('rag-results')).toBeNull();
  });
});
