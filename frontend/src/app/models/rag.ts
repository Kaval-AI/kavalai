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

export interface RagResult {
  id: string;
  model: string;
  collection_name: string;
  source_id: string;
  content: string;
  embedding_size: number;
  rag_metadata: any;
  created_at: string;
  updated_at: string;
  similarity: number;
}

export interface PcaPoint {
  label: string;
  x: number;
  y: number;
}

export interface PcaData {
  query: PcaPoint;
  results: PcaPoint[];
  samples: PcaPoint[];
}

export interface RagQueryResponse {
  results: RagResult[];
  pca_data: PcaData | null;
}

export interface RagStats {
  total_entries: number;
  total_collections: number;
  collections: string[];
}

/** A registered collection; `model` is the one its entries were embedded with. */
export interface RagCollection {
  name: string;
  model: string;
  embedding_size: number;
  schema_version: number;
  count: number;
}
