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
import { RagService } from './rag-service';
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
