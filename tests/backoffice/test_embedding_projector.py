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

import os
import csv
from unittest.mock import AsyncMock, MagicMock

import pytest
import numpy as np
from sklearn.decomposition import IncrementalPCA

from kavalai.db import ModelCallStat
from kavalai.backoffice.embedding_projector import download_rag_index, compute_pca
from kavalai.llm_clients.streamer import Streamer
from kavalai.rag import PostgresRagService


def make_seeded_rag_service(session_maker, schema, embeddings_by_text):
    """RAG service with a mocked embedding client (storage is backend-owned)."""
    service = PostgresRagService(
        session_maker=session_maker, model="test/embedding", schema=schema
    )

    async def fake_compute_embeddings(
        texts, normalize=False, normalizer=None, **kwargs
    ):
        stats = ModelCallStat(call_type="embedding", model="test/embedding")
        return [embeddings_by_text[t] for t in texts], stats

    client = MagicMock()
    client.compute_embeddings = AsyncMock(side_effect=fake_compute_embeddings)
    service.embedding_client = client
    return service


@pytest.mark.asyncio
async def test_download_rag_index(agents_db: object, agents_session_maker, tmp_path):
    collection = "test_collection"
    embeddings_by_text = {
        "content1": [1.0, 0.0, 0.0],
        "content2": [0.0, 1.0, 0.0],
        "content3": [0.0, 0.0, 1.0],
    }
    service = make_seeded_rag_service(
        agents_session_maker, "test_agents", embeddings_by_text
    )
    await service.index_batch(
        texts=list(embeddings_by_text.keys()),
        metadata_list=[{}] * 3,
        source_ids=["label1", "label2", "label3"],
        collection_name=collection,
    )

    csv_path = tmp_path / "rag_index.csv"
    streamer = Streamer(stream_delta=True).get_value_streamer("test")

    # Execute
    await download_rag_index(service, collection, str(csv_path), streamer=streamer)

    # Verify
    assert os.path.exists(csv_path)
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
        assert len(rows) == 3
        # Export order follows entry ids (random UUIDs) — compare as a set.
        by_label = {row[0]: [float(x) for x in row[1:]] for row in rows}
        assert by_label == embeddings_by_text

    await service.drop_collection(collection)


@pytest.mark.asyncio
async def test_compute_pca(tmp_path):
    # Setup: Create a CSV file with embeddings
    csv_path = tmp_path / "pca_input.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Create some data that is easy to project
        # 4 points in 3D, mainly varying in X and Y
        writer.writerow(["p1", 1.0, 1.0, 0.01])
        writer.writerow(["p2", 1.1, 0.9, -0.01])
        writer.writerow(["p3", -1.0, -1.0, 0.02])
        writer.writerow(["p4", -0.9, -1.1, -0.02])

    # Execute
    streamer = Streamer(stream_delta=True).get_value_streamer("test")
    ipca = await compute_pca(
        str(csv_path), n_components=2, batch_size=2, streamer=streamer
    )

    # Verify
    assert isinstance(ipca, IncrementalPCA)
    assert ipca.n_components == 2

    # Check that we can transform data with it
    data = np.array(
        [[1.0, 1.0, 0.01], [1.1, 0.9, -0.01], [-1.0, -1.0, 0.02], [-0.9, -1.1, -0.02]]
    )
    transformed = ipca.transform(data)
    assert transformed.shape == (4, 2)

    # The first component should capture most of the variance (from 1,1 to -1,-1)
    # Check that it's not all zeros
    assert np.any(np.abs(transformed) > 0.1)


@pytest.mark.asyncio
async def test_compute_pca_empty(tmp_path):
    csv_path = tmp_path / "empty.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as _:
        pass

    with pytest.raises(ValueError, match="No data found in CSV for PCA computation."):
        await compute_pca(str(csv_path))
