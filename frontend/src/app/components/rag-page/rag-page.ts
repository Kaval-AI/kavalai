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
import { Component, DestroyRef, NgZone, computed, inject, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { DecimalPipe, JsonPipe } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Subscription, forkJoin } from 'rxjs';
import { RagService } from '../../services/rag-service';
import { UserService } from '../../services/user-service';
import { PcaData, RagCollection, RagResult, RagStats } from '../../models/rag';
import { errorMessage } from '../../core/http-error';
import { CHART_COLORS } from '../../core/daily-stats';
import {
  UiAlert,
  UiButton,
  UiCard,
  UiDialog,
  UiEmptyState,
  UiField,
  UiLoading,
  UiPageHeader,
  UiProgress,
  UiSpinner,
  UiStat,
  UiStats,
  UiTable,
  UiToolbar,
} from '../../ui';

/** The projection's drawing area, in SVG units: the plot inside a margin. */
const PLOT = { width: 600, height: 400, margin: 40 };

/** The `top_k` choices of the query form. */
const TOP_K_OPTIONS = [5, 10, 20, 50, 100];

/**
 * The RAG collections of the active project: their totals, a similarity
 * query against one of them (the results with their similarity as a bar),
 * and the PCA projection of the query, its matches and a sample of the
 * collection, trained on demand over server-sent events.
 */
@Component({
  selector: 'app-rag-page',
  imports: [
    DecimalPipe,
    JsonPipe,
    FormsModule,
    UiAlert,
    UiButton,
    UiCard,
    UiDialog,
    UiEmptyState,
    UiField,
    UiLoading,
    UiPageHeader,
    UiProgress,
    UiSpinner,
    UiStat,
    UiStats,
    UiTable,
    UiToolbar,
  ],
  templateUrl: './rag-page.html',
  styleUrl: './rag-page.css',
})
export class RagPage {
  private readonly ragService = inject(RagService);
  private readonly userService = inject(UserService);
  private readonly ngZone = inject(NgZone);
  private readonly destroyRef = inject(DestroyRef);

  readonly colors = CHART_COLORS;
  readonly plot = PLOT;
  readonly topKOptions = TOP_K_OPTIONS;

  readonly projectId = signal<string | null>(null);
  readonly stats = signal<RagStats | null>(null);
  readonly collections = signal<RagCollection[]>([]);
  readonly loading = signal(false);
  readonly error = signal<string | null>(null);

  queryText = '';
  collectionName = '';
  sourceIdsInput = '';
  topK = 10;
  keepBest = false;
  normalizerYaml = '';

  readonly querying = signal(false);
  readonly queryError = signal<string | null>(null);
  /** `null` until the first query has answered. */
  readonly results = signal<RagResult[] | null>(null);
  readonly pcaData = signal<PcaData | null>(null);

  readonly trainingPca = signal(false);
  readonly showPca = signal(false);
  readonly pcaMessages = signal<string[]>([]);
  private pcaSubscription: Subscription | null = null;
  private started = false;

  readonly points = computed(() => {
    const pca = this.pcaData();
    return pca ? [pca.query, ...pca.results, ...pca.samples] : [];
  });
  private readonly bounds = computed(() => {
    const points = this.points();
    const xs = points.map((p) => p.x);
    const ys = points.map((p) => p.y);
    return {
      minX: Math.min(...xs),
      maxX: Math.max(...xs),
      minY: Math.min(...ys),
      maxY: Math.max(...ys),
    };
  });

  constructor() {
    this.userService.userDetails.pipe(takeUntilDestroyed()).subscribe((user) => {
      if (!user) return;
      const projectId =
        user.active_project_id && user.active_project_id !== 'None' ? user.active_project_id : null;
      if (this.started && projectId === this.projectId()) return;
      this.started = true;
      this.projectId.set(projectId);
      this.results.set(null);
      this.pcaData.set(null);
      this.collectionName = '';
      this.load();
    });
    this.destroyRef.onDestroy(() => this.pcaSubscription?.unsubscribe());
  }

  load(): void {
    const projectId = this.projectId();
    if (!projectId) return;
    this.loading.set(true);
    this.error.set(null);
    forkJoin({
      stats: this.ragService.getRagStats(projectId),
      collections: this.ragService.getRagCollections(projectId),
    })
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: ({ stats, collections }) => {
          this.stats.set(stats);
          this.collections.set(collections);
          // There is no cross-collection search, so a query always names one.
          if (!collections.some((c) => c.name === this.collectionName)) {
            const preferred = collections.find((c) => c.name === 'default') ?? collections[0];
            this.collectionName = preferred?.name ?? '';
          }
          this.loading.set(false);
        },
        error: (err) => {
          this.error.set(errorMessage(err, 'Could not load the RAG collections.'));
          this.loading.set(false);
        },
      });
  }

  /**
   * The model the selected collection was indexed with. The service embeds
   * every query against the collection with it, so the page shows it rather
   * than asking for one.
   */
  collectionModel(): string | null {
    return this.collections().find((c) => c.name === this.collectionName)?.model ?? null;
  }

  query(): void {
    const projectId = this.projectId();
    const text = this.queryText.trim();
    if (!projectId || !text) return;
    this.querying.set(true);
    this.queryError.set(null);
    const sourceIds = this.sourceIdsInput
      .split(',')
      .map((id) => id.trim())
      .filter((id) => !!id);
    this.ragService
      .queryRag(projectId, {
        text,
        collection_name: this.collectionName || undefined,
        top_k: this.topK,
        source_ids: sourceIds.length ? sourceIds : undefined,
        keep_best: this.keepBest,
        normalizer_yaml: this.normalizerYaml.trim() || undefined,
      })
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (response) => {
          this.results.set(response.results);
          this.pcaData.set(response.pca_data);
          this.querying.set(false);
        },
        error: (err) => {
          this.queryError.set(errorMessage(err, 'The query failed.'));
          this.querying.set(false);
        },
      });
  }

  computePca(): void {
    const projectId = this.projectId();
    if (projectId && this.collectionName) this.train(projectId, this.collectionName);
  }

  private train(projectId: string, collectionName: string): void {
    this.trainingPca.set(true);
    this.pcaMessages.set([]);
    this.showPca.set(true);
    const lost = () =>
      this.ngZone.run(() => {
        if (!this.trainingPca()) return;
        this.log('Error: the connection to the server was lost.');
        this.trainingPca.set(false);
      });
    this.pcaSubscription = this.ragService.trainPca(projectId, collectionName).subscribe({
      next: (data) => this.ngZone.run(() => this.onPcaMessage(data)),
      error: lost,
      complete: lost,
    });
  }

  private onPcaMessage(data: string): void {
    let message: { value?: string; status?: string; type?: string };
    try {
      message = JSON.parse(data);
    } catch {
      this.log(data);
      return;
    }
    if (message.value) this.log(message.value);
    if (message.status === 'error' || message.type === 'error') {
      this.trainingPca.set(false);
    }
    if (message.type === 'complete' || message.value?.includes('completed successfully')) {
      this.trainingPca.set(false);
      this.pcaSubscription?.unsubscribe();
      this.pcaSubscription = null;
    }
  }

  private log(line: string): void {
    this.pcaMessages.update((lines) => [...lines, line]);
  }

  stopPca(): void {
    this.pcaSubscription?.unsubscribe();
    this.pcaSubscription = null;
    this.trainingPca.set(false);
    this.showPca.set(false);
  }

  percent(similarity: number): number {
    return Math.round(similarity * 100);
  }

  shortId(id: string): string {
    return id.length > 12 ? `${id.slice(0, 8)}…` : id;
  }

  scaleX(value: number): number {
    const { minX, maxX } = this.bounds();
    const range = maxX - minX || 1;
    return PLOT.margin + ((value - minX) / range) * (PLOT.width - 2 * PLOT.margin);
  }

  scaleY(value: number): number {
    const { minY, maxY } = this.bounds();
    const range = maxY - minY || 1;
    return PLOT.height - PLOT.margin - ((value - minY) / range) * (PLOT.height - 2 * PLOT.margin);
  }
}
