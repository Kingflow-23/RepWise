# Local hybrid retrieval

## What each retriever does

- **BM25** counts query terms in each chunk and weights terms that are rare across the corpus. It is the lexical baseline and needs no model download.
- **Semantic** embeds passages once, then embeds each query and ranks by cosine similarity. It can find paraphrases even when the wording differs.
- **Hybrid** runs both rankers and combines rank positions with Reciprocal Rank Fusion:

```text
RRF(chunk) = 1 / (60 + BM25 rank) + 1 / (60 + semantic rank)
```

The constant 60 is fixed and documented in `HybridRetriever`. Fusion uses ranks because BM25 scores and cosine similarities are on different scales. Chunks absent from one list receive no contribution from that list.

All local retrievers exclude chunks flagged `sponsor` or `cta`; evidence selection applies the same exclusion again before generation. The retrieval benchmark also uses the retriever's filtered ranking.

## Model and local index

The default embedding model is [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small). Its model card lists an MIT license and multilingual support; the safetensors weights are about 471 MB. Sentence Transformers and PyTorch add disk use, so the complete Python environment is substantially larger than the weights. Encoding a large corpus needs model memory plus one vector per chunk; allow at least 1–2 GB free RAM for this small model and more for large corpora.

Download the model once, then build the index:

```powershell
python -m pip install -r requirements.txt
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='intfloat/multilingual-e5-small')"
python -m src.retrieval.index --build --chunks data/corpus/chunks --index data/corpus/index/semantic.npz
```

The index is a compressed NumPy file containing normalized vectors, ordered chunk IDs, model name, and a SHA-256 fingerprint of chunk text and IDs. Text and citation metadata remain in the JSONL files. Search verifies the model, IDs, and fingerprint; rebuild the index after changing the corpus or model. Generated indexes and corpus chunks are local artifacts and must not be committed.

On load, RepWise also checks that the index matrix has one row per stored chunk ID and that its embedding dimension matches the configured model. If the model or index cannot be loaded locally, the API reports the setup error rather than falling back to another retrieval method. Install dependencies with `python -m pip install -r requirements.txt`, download the model using the command above, and rebuild a stale or incompatible index with the index-build command.

## Stable source IDs

Ingestion sets `chunk_id` from the video ID and the first/last source passage IDs. The same record retains `passage_ids`, title, URL, start/end seconds, and timestamp URL. Rebuilding with the same source passages preserves the ID. If chunk boundaries change, the changed passage range gets a different ID and the embedding index must be rebuilt.

## Evaluation

Use the same relevance labels for each retrieval mode:

```powershell
python -m src.evaluation.runner --suite retrieval --retriever bm25
python -m src.evaluation.runner --suite retrieval --retriever semantic
python -m src.evaluation.runner --suite retrieval --retriever hybrid
```

Recall@k measures how many labelled evidence spans are covered in the top k; MRR rewards placing the first relevant result near the top. A span can overlap more than one chunk, but is counted once. No claim that hybrid is better should be made until the measured report shows that on the same labelled queries.
