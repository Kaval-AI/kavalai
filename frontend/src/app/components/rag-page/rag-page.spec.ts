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
import { BehaviorSubject, of, throwError } from 'rxjs';

import { RagPage } from './rag-page';
import { RagService } from '../../services/rag-service';
import { UserService } from '../../services/user-service';
import { RagCollection } from '../../models/rag';

describe('RagPage', () => {
  let component: RagPage;
  let fixture: ComponentFixture<RagPage>;
  let ragServiceSpy: jasmine.SpyObj<RagService>;
  let userDetails: BehaviorSubject<any>;

  const faq: RagCollection = {
    name: 'faq', model: 'openai/text-embedding-3-small', embedding_size: 1536, schema_version: 2, count: 12
  };
  const fallback: RagCollection = {
    name: 'default', model: 'fastembed/BAAI/bge-small-en-v1.5', embedding_size: 384, schema_version: 2, count: 30
  };

  beforeEach(async () => {
    ragServiceSpy = jasmine.createSpyObj('RagService', ['getRagStats', 'getRagCollections', 'queryRag', 'trainPca']);
    ragServiceSpy.getRagStats.and.returnValue(of({ total_entries: 42, total_collections: 2, collections: ['default', 'faq'] }));
    ragServiceSpy.getRagCollections.and.returnValue(of([faq, fallback]));
    ragServiceSpy.queryRag.and.returnValue(of({ results: [], pca_data: null }));
    userDetails = new BehaviorSubject<any>({ active_project_id: 'proj1' });

    await TestBed.configureTestingModule({
      imports: [RagPage],
      providers: [
        { provide: RagService, useValue: ragServiceSpy },
        { provide: UserService, useValue: { userDetails: userDetails.asObservable() } },
      ]
    }).compileComponents();

    fixture = TestBed.createComponent(RagPage);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('selects the default collection and shows the model it was indexed with', () => {
    expect(ragServiceSpy.getRagCollections).toHaveBeenCalledWith('proj1');
    expect(component.collectionName).toBe('default');
    expect(component.collectionModel).toBe('fastembed/BAAI/bge-small-en-v1.5');
  });

  it('follows the selected collection with its model', () => {
    component.collectionName = 'faq';
    expect(component.collectionModel).toBe('openai/text-embedding-3-small');
  });

  it('falls back to the first collection when there is no default one', () => {
    ragServiceSpy.getRagCollections.and.returnValue(of([faq]));
    component.collectionName = 'gone';
    component.loadRagStats();
    expect(component.collectionName).toBe('faq');
  });

  it('keeps a selection that still exists', () => {
    component.collectionName = 'faq';
    component.loadRagStats();
    expect(component.collectionName).toBe('faq');
  });

  it('selects nothing when the project has no collection', () => {
    ragServiceSpy.getRagCollections.and.returnValue(of([]));
    component.loadRagStats();
    expect(component.collectionName).toBe('');
    expect(component.collectionModel).toBeNull();
  });

  it('keeps the page usable when the collections cannot be listed', () => {
    spyOn(console, 'error');
    ragServiceSpy.getRagCollections.and.returnValue(throwError(() => new Error('boom')));
    userDetails.next({ active_project_id: 'proj2' });
    expect(component.collections).toEqual([]);
    expect(component.collectionName).toBe('');
    expect(console.error).toHaveBeenCalled();
  });

  it('sends nothing without query text', () => {
    component.queryText = '';
    component.onQuery();
    expect(ragServiceSpy.queryRag).not.toHaveBeenCalled();
  });

  it('passes the source identifiers as a trimmed list', () => {
    component.queryText = 'opening hours';
    component.sourceIdsInput = ' a, b ,,';
    component.onQuery();
    expect(ragServiceSpy.queryRag.calls.mostRecent().args[1].source_ids).toEqual(['a', 'b']);
  });

  it('queries without naming a model', () => {
    component.queryText = 'opening hours';
    component.onQuery();

    expect(ragServiceSpy.queryRag).toHaveBeenCalledWith('proj1', jasmine.objectContaining({
      text: 'opening hours',
      collection_name: 'default',
    }));
    expect('model' in ragServiceSpy.queryRag.calls.mostRecent().args[1]).toBeFalse();
  });
});
