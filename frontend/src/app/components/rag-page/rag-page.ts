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
import { Component, OnInit, NgZone } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { RagService } from '../../services/rag-service';
import { UserService } from '../../services/user-service';
import { RagCollection, RagResult, RagStats, PcaData, PcaPoint } from '../../models/rag';

@Component({
  selector: 'app-rag-page',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './rag-page.html',
  styleUrl: './rag-page.css',
})
export class RagPage implements OnInit {
  projectId: string | null = null;
  results: RagResult[] = [];
  pcaData: PcaData | null = null;
  ragStats: RagStats | null = null;
  collections: RagCollection[] = [];

  queryText: string = '';
  collectionName: string = '';
  sourceIdsInput: string = '';
  topK: number = 10;
  keepBest: boolean = false;
  normalizerYaml: string = '';

  loading: boolean = false;
  trainingPca: boolean = false;
  pcaMessages: string[] = [];
  showPcaModal: boolean = false;
  error: string | null = null;
  private pcaSubscription: any = null;

  hoveredPoint: PcaPoint | null = null;
  tooltipPos = { x: 0, y: 0 };
  isPcaExpanded: boolean = true;

  constructor(
    private ragService: RagService,
    private userService: UserService,
    private ngZone: NgZone
  ) {}

  ngOnInit(): void {
    this.userService.userDetails.subscribe(user => {
      if (user && user.active_project_id) {
        const newProjectId = user.active_project_id !== 'None' ? user.active_project_id : null;
        if (newProjectId !== this.projectId) {
          this.projectId = newProjectId;
          this.results = [];
          this.collections = [];
          this.collectionName = '';
          this.loadRagStats();
        }
      }
    });
  }

  loadRagStats(): void {
    if (!this.projectId) return;

    this.ragService.getRagStats(this.projectId).subscribe({
      next: (stats) => {
        this.ragStats = stats;
      },
      error: (err) => {
        console.error('Error loading RAG stats', err);
      }
    });

    this.ragService.getRagCollections(this.projectId).subscribe({
      next: (collections) => {
        this.collections = collections;
        // There is no cross-collection search, so a query always names one.
        if (!collections.some(c => c.name === this.collectionName)) {
          const preferred = collections.find(c => c.name === 'default') ?? collections[0];
          this.collectionName = preferred?.name ?? '';
        }
      },
      error: (err) => {
        console.error('Error loading RAG collections', err);
      }
    });
  }

  /**
   * The model the selected collection was indexed with. The service embeds
   * every query against the collection with it, so the page shows it rather
   * than asking for one.
   */
  get collectionModel(): string | null {
    return this.collections.find(c => c.name === this.collectionName)?.model ?? null;
  }

  onQuery(): void {
    if (!this.projectId || !this.queryText) {
      return;
    }

    this.loading = true;
    this.error = null;
    const sourceIds = this.sourceIdsInput
      ? this.sourceIdsInput.split(',').map(id => id.trim()).filter(id => !!id)
      : undefined;

    this.ragService.queryRag(this.projectId, {
      text: this.queryText,
      collection_name: this.collectionName || undefined,
      top_k: this.topK,
      source_ids: sourceIds,
      keep_best: this.keepBest,
      normalizer_yaml: this.normalizerYaml || undefined
    }).subscribe({
      next: (response) => {
        this.results = response.results;
        this.pcaData = response.pca_data;
        this.loading = false;
      },
      error: (err) => {
        console.error('Error querying RAG', err);
        this.error = 'Failed to execute RAG query.';
        this.loading = false;
      }
    });
  }

  onComputePca(): void {
    if (!this.projectId || !this.collectionName) {
      this.error = 'Please select a collection to compute PCA.';
      return;
    }

    this.trainingPca = true;
    this.pcaMessages = [];
    this.showPcaModal = true;
    this.error = null;

    this.pcaSubscription = this.ragService.trainPca(this.projectId, this.collectionName).subscribe({
      next: (data) => {
        this.ngZone.run(() => {
          try {
            const msg = JSON.parse(data);
            if (msg.value) {
              this.pcaMessages.push(msg.value);
            }
            if (msg.status === 'error' || msg.type === 'error') {
              this.trainingPca = false;
            }
            if (msg.type === 'complete' || (msg.value && msg.value.includes('completed successfully'))) {
              this.trainingPca = false;
              if (this.pcaSubscription) {
                this.pcaSubscription.unsubscribe();
                this.pcaSubscription = null;
              }
            }
          } catch (e) {
            this.pcaMessages.push(data);
          }
        });
      },
      error: (err) => {
        this.ngZone.run(() => {
          if (this.trainingPca) {
            console.error('Error training PCA', err);
            this.pcaMessages.push('Error: Failed to connect to server.');
            this.trainingPca = false;
          }
        });
      }
    });
  }

  stopPca(): void {
    if (this.pcaSubscription) {
      this.pcaSubscription.unsubscribe();
      this.pcaSubscription = null;
    }
    this.trainingPca = false;
    this.showPcaModal = false;
  }

  get allPoints(): PcaPoint[] {
    if (!this.pcaData) return [];
    return [this.pcaData.query, ...this.pcaData.results, ...this.pcaData.samples];
  }

  get minX(): number {
    const pts = this.allPoints;
    return pts.length ? Math.min(...pts.map(p => p.x)) : 0;
  }

  get maxX(): number {
    const pts = this.allPoints;
    return pts.length ? Math.max(...pts.map(p => p.x)) : 1;
  }

  get minY(): number {
    const pts = this.allPoints;
    return pts.length ? Math.min(...pts.map(p => p.y)) : 0;
  }

  get maxY(): number {
    const pts = this.allPoints;
    return pts.length ? Math.max(...pts.map(p => p.y)) : 1;
  }

  getScaleX(val: number): number {
    const range = this.maxX - this.minX || 1;
    return 50 + ((val - this.minX) / range) * 500;
  }

  getScaleY(val: number): number {
    const range = this.maxY - this.minY || 1;
    return 350 - ((val - this.minY) / range) * 300;
  }

  onPointHover(point: PcaPoint, event: MouseEvent): void {
    this.hoveredPoint = point;
    this.updateTooltipPos(event);
  }

  onPointMove(event: MouseEvent): void {
    this.updateTooltipPos(event);
  }

  onPointLeave(): void {
    this.hoveredPoint = null;
  }

  private updateTooltipPos(event: MouseEvent): void {
    this.tooltipPos = { x: event.clientX + 10, y: event.clientY + 10 };
  }
}
