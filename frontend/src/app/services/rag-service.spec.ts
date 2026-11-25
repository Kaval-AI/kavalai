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

import { TestBed } from '@angular/core/testing';
import { HttpClientTestingModule, HttpTestingController } from '@angular/common/http/testing';
import { HttpErrorResponse, HttpEventType } from '@angular/common/http';
import { RagService, splitSseEvents } from './rag-service';
import { RagCollection, RagResult, RagStats, RagQueryResponse } from '../models/rag';

describe('RagService', () => {
  let service: RagService;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      imports: [HttpClientTestingModule],
      providers: [RagService]
    });
    service = TestBed.inject(RagService);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    httpMock.verify();
  });

  describe('trainPca', () => {
    const url = '/api/projects/p1/rag/train-pca';

    it('posts the collection and emits each event as it arrives', () => {
      const received: string[] = [];
      let completed = false;
      service.trainPca('p1', 'songs').subscribe({
        next: (data) => received.push(data),
        complete: () => (completed = true),
      });

      const req = httpMock.expectOne(url);
      expect(req.request.method).toBe('POST');
      expect(req.request.body).toEqual({ collection_name: 'songs' });
      expect(req.request.responseType).toBe('text');
      expect(req.request.reportProgress).toBeTrue();

      const first = 'data: {"value":"one"}\r\n\r\n: ping\r\n\r\ndata: {"val';
      req.event({ type: HttpEventType.DownloadProgress, loaded: 1, partialText: first });
      expect(received).toEqual(['{"value":"one"}']);

      const all = `${first}ue":"two"}\r\n\r\ndata:three\r\ndata: lines`;
      req.event({ type: HttpEventType.DownloadProgress, loaded: 2, partialText: all });
      expect(received).toEqual(['{"value":"one"}', '{"value":"two"}']);
      expect(completed).toBeFalse();

      req.flush(all);
      expect(received).toEqual(['{"value":"one"}', '{"value":"two"}', 'three\nlines']);
      expect(completed).toBeTrue();
    });

    it('reports a refused request as an error', () => {
      let failed: unknown;
      service.trainPca('p1', 'songs').subscribe({ error: (err) => (failed = err) });

      httpMock
        .expectOne(url)
        .flush({ detail: 'Forbidden' }, { status: 403, statusText: 'Forbidden' });
      expect((failed as HttpErrorResponse).status).toBe(403);
    });

    it('aborts the request when unsubscribed', () => {
      const subscription = service.trainPca('p1', 'songs').subscribe();
      const req = httpMock.expectOne(url);
      subscription.unsubscribe();
      expect(req.cancelled).toBeTrue();
    });
  });

  it('splits a server-sent event stream into complete events and the rest', () => {
    expect(splitSseEvents('data: a\n\nevent: x\n\n: ping\n\ndata: b')).toEqual({
      events: ['a'],
      rest: 'data: b',
    });
    expect(splitSseEvents('')).toEqual({ events: [], rest: '' });
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });

  it('should query RAG', () => {
    const mockResults: RagResult[] = [
      { content: 'Result 1' } as RagResult
    ];
    const mockResponse: RagQueryResponse = {
      results: mockResults,
      pca_data: null
    };
    const projectId = 'proj123';
    const queryData = { model: 'text-embedding-3-small', text: 'query' };

    service.queryRag(projectId, queryData).subscribe(response => {
      expect(response).toEqual(mockResponse);
    });

    const req = httpMock.expectOne(`/api/projects/${projectId}/rag/query`);
    expect(req.request.method).toBe('POST');
    expect(req.request.body).toEqual(queryData);
    req.flush(mockResponse);
  });

  it('should query RAG with source_ids', () => {
    const mockResults: RagResult[] = [
      { content: 'Result 1' } as RagResult
    ];
    const mockResponse: RagQueryResponse = {
      results: mockResults,
      pca_data: null
    };
    const projectId = 'proj123';
    const queryData = {
      model: 'text-embedding-3-small',
      text: 'query',
      source_ids: ['id1', 'id2']
    };

    service.queryRag(projectId, queryData).subscribe(response => {
      expect(response).toEqual(mockResponse);
    });

    const req = httpMock.expectOne(`/api/projects/${projectId}/rag/query`);
    expect(req.request.method).toBe('POST');
    expect(req.request.body).toEqual(queryData);
    req.flush(mockResponse);
  });

  it('should fetch RAG stats', () => {
    const mockStats: RagStats = { total_entries: 10, total_collections: 1, collections: ['default'] };
    const projectId = 'proj123';

    service.getRagStats(projectId).subscribe(stats => {
      expect(stats).toEqual(mockStats);
    });

    const req = httpMock.expectOne(`/api/projects/${projectId}/rag/stats`);
    expect(req.request.method).toBe('GET');
    req.flush(mockStats);
  });

  it('should query RAG without a model', () => {
    const queryData = { text: 'query', collection_name: 'faq' };

    service.queryRag('proj123', queryData).subscribe(response => {
      expect(response.results).toEqual([]);
    });

    const req = httpMock.expectOne('/api/projects/proj123/rag/query');
    expect(req.request.body).toEqual(queryData);
    req.flush({ results: [], pca_data: null });
  });

  it('should list RAG collections with their models', () => {
    const mockCollections: RagCollection[] = [
      { name: 'faq', model: 'openai/text-embedding-3-small', embedding_size: 1536, schema_version: 2, count: 12 }
    ];

    service.getRagCollections('proj123').subscribe(collections => {
      expect(collections).toEqual(mockCollections);
    });

    const req = httpMock.expectOne('/api/projects/proj123/rag/collections');
    expect(req.request.method).toBe('GET');
    req.flush(mockCollections);
  });
});
