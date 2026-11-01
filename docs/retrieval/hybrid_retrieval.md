# Local hybrid retrieval

## What each retriever does

- **BM25** counts query terms in each chunk and weights terms that are rare across the corpus. It is the lexical baseline and needs no model download.
- **Semantic** embeds passages once, then embeds each query and ranks by cosine similarity. It can find paraphrases even when the wording differs.
- **Hybrid** runs both rankers and combines rank positions with Reciprocal Rank Fusion:

```text
RRF(chunk) = 1 / (60 + BM25 rank) + 1 / (60 + semantic rank)
```

The constant 60 is fixed and documented in `HybridRetriever`. Fusion uses ranks because BM25 scores and cosine similarities are on different scales. Chunks absent from one list receive no contribution from that list.

The chunker excludes any passage with flags before forming chunks. Retrieval and evidence selection therefore operate on clean chunks and do not repeat flag checks. Rebuild chunks and the semantic index after changing this ingestion behavior.

## French query translation and expansion

The corpus transcripts are French, so the application first turns each request
into one clean, plain-text French query. The configured local Ollama model
translates the request when needed, puts its main topic and outcome first, and
adds only closely related training terms that could match transcript wording.
Retrieval then runs once using that expanded query. With the default hybrid
strategy, that one call runs BM25 and semantic search internally and fuses their
rankings; it does not run separate searches for the original and expanded
queries. Source chunks, text, and citations are not rewritten.

The rewrite prompt treats request text as untrusted search content, preserves
the requested topic and constraints, and explicitly forbids answering or
adding assumptions. The model returns plain text, not JSON. This requires an
additional local Ollama call per new request, so it increases retrieval latency.
If Ollama is unavailable or returns invalid output, retrieval reports an error
instead of silently falling back to the unexpanded request.

Set `REPWISE_QUERY_EXPANSION=false` in `.env` to disable query rewriting. The
retrieval benchmark uses the same setting, allowing direct comparisons with
rewriting enabled and disabled. Keep the corpus, index, and evaluation cases
unchanged between those runs, and compare Recall@k, MRR, and latency; expansion
should not be assumed to improve results until measured.

The retrieval benchmark reports whether query rewriting was enabled and which
model was used.

## Body-part targeting

The chunker tags each chunk with canonical `body_parts` from
`config/body_part_lexicon.json`. The bilingual lexicon includes broad regions
(such as legs and arms) and specific muscles (such as quadriceps, hamstrings,
chest, and triceps). Tags are based on words explicitly present in the chunk;
they are retrieval hints, not human-verified claims that the passage contains
useful advice for that region.

The request query is expanded with matching canonical tags. BM25 indexes those
tags as searchable terms, and semantic embeddings include them alongside the
chunk title and transcript text. To tune the vocabulary, update the lexicon,
then rebuild chunks and the semantic index:

```powershell
python -m src.ingestion.chunker
python -m src.retrieval.index --build --chunks data/corpus/chunks --index data/corpus/index/semantic.npz
```

Changing chunk tags changes the semantic corpus fingerprint, so the existing
index must be rebuilt before semantic or hybrid retrieval can start.

The API defaults to `hybrid` through `REPWISE_RETRIEVAL`; it therefore fuses
lexical and semantic rankings for the normal request path. Query expansion is
performed once before fusion, not once per component. Benchmark comparisons
rewrite each answerable query once and share that same French query across
strategies. Plan-request rewrites are also prepared before both RAG generation
arms, so cache hits cannot advantage the arm evaluated second. Retrieval and
generation latency exclude the separate rewrite time, which is reported
independently. The comparison uses the configured production strategy for
generation (hybrid by default) while still reporting the ranking of each
retrieval strategy.

## Model and local index

The default embedding model is [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small). Its model card lists an MIT license and multilingual support; the safetensors weights are about 471 MB. Sentence Transformers and PyTorch add disk use, so the complete Python environment is substantially larger than the weights. Encoding a large corpus needs model memory plus one vector per chunk; allow at least 1–2 GB free RAM for this small model and more for large corpora.

Download the model once, then build the index:

```powershell
python -m pip install -r requirements.txt
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='intfloat/multilingual-e5-small')"
python -m src.retrieval.index --build --chunks data/corpus/chunks --index data/corpus/index/semantic.npz
```

The index is a compressed NumPy file containing normalized vectors, ordered chunk IDs, model name, and a SHA-256 fingerprint of chunk IDs, text, titles, and body-part tags. Text and citation metadata remain in the JSONL files. Search verifies the model, IDs, and fingerprint; rebuild the index after changing the corpus or model. Generated indexes and corpus chunks are local artifacts and must not be committed.

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

Recall@k measures how many relevant chunks appear in the top k; MRR rewards placing the first relevant chunk near the top. Human labels remain source-video time spans; all chunks overlapping a span by at least one second become relevant. BM25, semantic, and hybrid are therefore ranked against the same source-grounded chunk labels, without requiring passage retrieval. No claim that hybrid is better should be made until the measured report shows that on the same labelled queries.
