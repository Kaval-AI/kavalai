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
import { HttpClient, HttpDownloadProgressEvent, HttpEventType } from '@angular/common/http';
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

  /**
   * Trains the PCA projection of a collection and emits the `data` of each
   * server-sent event as it arrives; completes when the response ends.
   *
   * A POST read as a stream rather than an `EventSource`, which can only
   * GET: training writes to the database, and a GET would be started by any
   * image pointing at it. Unsubscribing aborts the request.
   */
  trainPca(projectId: string, collectionName: string): Observable<string> {
    return new Observable<string>(observer => {
      let consumed = 0;
      const read = (text: string) => {
        const normalised = text.replace(/\r\n?/g, '\n');
        const { events, rest } = splitSseEvents(normalised.slice(consumed));
        consumed = normalised.length - rest.length;
        events.forEach(data => observer.next(data));
      };
      const subscription = this.http
        .post(
          `/api/projects/${projectId}/rag/train-pca`,
          { collection_name: collectionName },
          { observe: 'events', reportProgress: true, responseType: 'text' },
        )
        .subscribe({
          next: event => {
            if (event.type === HttpEventType.DownloadProgress) {
              read((event as HttpDownloadProgressEvent).partialText ?? '');
            } else if (event.type === HttpEventType.Response) {
              read(`${event.body ?? ''}\n\n`);
              observer.complete();
            }
          },
          error: err => observer.error(err),
        });
      return () => subscription.unsubscribe();
    });
  }
}

/**
 * The `data` of each complete event in a server-sent event stream whose line
 * ends are `\n`, and the unfinished remainder. Comments (`: ping`) and
 * events without data yield nothing.
 */
export function splitSseEvents(text: string): { events: string[]; rest: string } {
  const frames = text.split('\n\n');
  const rest = frames.pop() ?? '';
  const events = frames.flatMap(frame => {
    const data = frame
      .split('\n')
      .filter(line => line.startsWith('data:'))
      .map(line => line.slice(5).replace(/^ /, ''));
    return data.length ? [data.join('\n')] : [];
  });
  return { events, rest };
}
