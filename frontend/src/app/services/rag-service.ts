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

import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { RagCollection, RagStats, RagQueryResponse } from '../models/rag';

@Injectable({
  providedIn: 'root'
})
export class RagService {
  constructor(private http: HttpClient) {}

  /**
   * Query a collection. `model` may be omitted: an existing collection is
   * embedded with the model recorded for it.
   */
  queryRag(projectId: string, queryData: {
    model?: string,
    text: string,
    collection_name?: string,
    top_k?: number,
    source_ids?: string[],
    keep_best?: boolean,
    normalizer_yaml?: string
  }): Observable<RagQueryResponse> {
    return this.http.post<RagQueryResponse>(`/api/projects/${projectId}/rag/query`, queryData);
  }

  getRagStats(projectId: string): Observable<RagStats> {
    return this.http.get<RagStats>(`/api/projects/${projectId}/rag/stats`);
  }

  getRagCollections(projectId: string): Observable<RagCollection[]> {
    return this.http.get<RagCollection[]>(`/api/projects/${projectId}/rag/collections`);
  }

  trainPca(projectId: string, collectionName: string): Observable<string> {
    return new Observable<string>(observer => {
      const eventSource = new EventSource(`/api/projects/${projectId}/rag/train-pca?collection_name=${encodeURIComponent(collectionName)}`);

      eventSource.onmessage = (event) => {
        observer.next(event.data);
      };

      eventSource.onerror = (error) => {
        observer.error(error);
        eventSource.close();
      };

      return () => {
        eventSource.close();
      };
    });
  }
}
