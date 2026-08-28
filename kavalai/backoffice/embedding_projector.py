"""
Copyright 2026 OÜ KAVAL AI (registry code 17393877)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
you may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import csv
import pickle
import base64
import json
import tempfile
import os
from typing import Optional
from uuid import UUID
from sqlalchemy import select
from sklearn.decomposition import IncrementalPCA
import numpy as np
from loguru import logger

from kavalai.backoffice.db import Project, ProjectCache
from kavalai.llm_clients.streamer import ValueStreamer
from kavalai.rag.base import BaseRagService

SAMPLE_SIZE = 500


async def download_rag_index(
    rag_service: BaseRagService,
    collection_name: str,
    output_csv_path: str,
    streamer: Optional[ValueStreamer] = None,
):
    """
    Downloads a RAG collection to the specified CSV file via the service's
    bulk-export API. The first column is the label (content, falling back to
    source_id); the remaining columns are the embedding components.

    Args:
        rag_service: RAG service to export from (storage is backend-owned).
        collection_name: The name of the collection to download.
        output_csv_path: The path where the CSV file will be saved.
        streamer: Optional streamer for progress reporting.
    """
    total_count = await rag_service.count_entries(collection_name)

    with open(output_csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        count = 0
        async for item in rag_service.iter_entries(collection_name):
            if not item["embedding"]:
                continue
            label = item["content"] or item["source_id"]
            writer.writerow([label, *item["embedding"]])
            count += 1
            if count % 100 == 0 and streamer:
                await streamer.stream_partial(
                    f"Downloaded {count}/{total_count} items..."
                )

    msg = f"Finished downloading {count}/{total_count} items."
    logger.info(msg)
    if streamer:
        await streamer.stream_partial(msg)


async def compute_pca(
    csv_path: str,
    n_components: int = 2,
    batch_size: int = 100,
    streamer: Optional[ValueStreamer] = None,
) -> IncrementalPCA:
    """
    Fit an IncrementalPCA model to the embeddings in the CSV, batch by batch.

    Args:
        csv_path: Path to the CSV file containing labels and embeddings.
        n_components: Number of principal components to compute.
        batch_size: The number of rows to process at once for incremental training.
        streamer: Optional streamer for progress reporting.

    Returns:
        IncrementalPCA: The fitted PCA model.
    """
    ipca = IncrementalPCA(n_components=n_components)
    batch = []
    row_count = 0
    batch_count = 0

    with open(csv_path, "r", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if not row:
                continue
            batch.append([float(x) for x in row[1:]])
            row_count += 1
            if len(batch) >= batch_size:
                ipca.partial_fit(np.array(batch))
                batch = []
                batch_count += 1
                if batch_count % 5 == 0:
                    msg = f"Processed {row_count} rows for PCA..."
                    if streamer:
                        await streamer.stream_partial(msg)

        # IncrementalPCA needs at least n_components rows per partial_fit; a
        # smaller remainder is dropped.
        if len(batch) >= n_components:
            ipca.partial_fit(np.array(batch))
            msg = f"Processed final {row_count} rows for PCA."
            logger.info(msg)
            if streamer:
                await streamer.stream_partial(msg)

    if row_count == 0:
        raise ValueError("No data found in CSV for PCA computation.")

    return ipca


def _project_sample(csv_path: str, ipca: IncrementalPCA, size: int) -> list[dict]:
    """Project the first ``size`` rows of the CSV onto the PCA plane."""
    labels = []
    batch = []
    with open(csv_path, "r", encoding="utf-8") as csvfile:
        for row in csv.reader(csvfile):
            if not row:
                continue
            labels.append(row[0])
            batch.append([float(x) for x in row[1:]])
            if len(batch) >= size:
                break
    if not batch:
        return []
    transformed = ipca.transform(np.array(batch))
    return [
        {"label": label, "x": point[0], "y": point[1]}
        for label, point in zip(labels, transformed)
    ]


async def _upsert_cache(bo_session, project_id, name: str, value: str) -> None:
    """Set one ``project_cache`` entry, creating it when absent."""
    stmt = select(ProjectCache).where(
        ProjectCache.project_id == project_id, ProjectCache.name == name
    )
    entry = (await bo_session.execute(stmt)).scalar_one_or_none()
    if entry:
        entry.value = value
    else:
        bo_session.add(ProjectCache(project_id=project_id, name=name, value=value))


async def train_pca(
    bo_session_maker,
    rag_service: BaseRagService,
    project_id: UUID,
    collection_name: str,
    streamer: Optional[ValueStreamer] = None,
):
    """
    Trains PCA model for a given collection and stores it in the project cache.

    Args:
        bo_session_maker: Callable returning an async backoffice session.
        rag_service: RAG service holding the collection (storage backend-owned).
        project_id: Id of the backoffice project that owns the cache entries.
        collection_name: Name of the RAG collection.
        streamer: Optional streamer for progress reporting.
    """

    if streamer:
        await streamer.stream_partial(
            f"Starting PCA training for collection: {collection_name}"
        )

    async with bo_session_maker() as bo_session:
        if await bo_session.get(Project, project_id) is None:
            raise ValueError(f"Project '{project_id}' not found.")

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp_csv_path = tmp.name

    try:
        if streamer:
            await streamer.stream_partial("Downloading embeddings...")
        await download_rag_index(
            rag_service,
            collection_name,
            output_csv_path=tmp_csv_path,
            streamer=streamer,
        )

        if streamer:
            await streamer.stream_partial("Computing PCA model...")
        ipca = await compute_pca(
            tmp_csv_path,
            n_components=2,
            batch_size=100,
            streamer=streamer,
        )

        # The first SAMPLE_SIZE points, projected, give the explorer its
        # background.
        if streamer:
            await streamer.stream_partial("Generating sample points...")
        sample_points = _project_sample(tmp_csv_path, ipca, SAMPLE_SIZE)

        if streamer:
            await streamer.stream_partial("Storing results in cache...")
        async with bo_session_maker() as bo_session:
            await _upsert_cache(
                bo_session,
                project_id,
                f"pca_model_{collection_name}",
                base64.b64encode(pickle.dumps(ipca)).decode("utf-8"),
            )
            await _upsert_cache(
                bo_session,
                project_id,
                f"pca_sample_train_data_{collection_name}",
                json.dumps(sample_points),
            )
            await bo_session.commit()

        if streamer:
            await streamer.stream_partial("PCA training completed successfully.")
            await streamer.stream_complete()
        logger.info(f"PCA training completed for collection {collection_name}")

    finally:
        if os.path.exists(tmp_csv_path):
            os.remove(tmp_csv_path)
