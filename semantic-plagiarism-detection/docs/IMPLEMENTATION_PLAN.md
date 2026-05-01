# AI-Based Semantic Plagiarism Detection — Detailed Implementation Plan

## 1. Purpose and scope of this document

This document defines **what you will deliver**, **how the pipeline will work end-to-end**, **design decisions**, **evaluation protocol**, and **project milestones**. It supports implementation in Python with a small UI (e.g. Streamlit) and defense in viva.

### Recommended deliverables (minimum for a Master’s submission)

1. **Runnable prototype**: load two texts (or two files), output similarity and an interpretable verdict band.
2. **Core algorithm module**: preprocessing options, embedding, similarity, optional chunking/alignment.
3. **Evaluation notebook/script**: baseline vs semantic method, metrics on a small labeled set.
4. **Documentation**: architecture diagram, limitations, ethics, reproducibility (`requirements.txt`, fixed model name, seeds).
5. **Short demo**: 5–10 minute walkthrough (screens + 2–3 examples).

---

## 2. Problem definition (what “semantic plagiarism” means in this project)

This system is a **semantic similarity detector**, not a legal instrument. It answers:

> How semantically similar are these two texts?

### Inputs

- **Document A** (suspect text)
- **Document B** (reference text or another document)

### Outputs

- A **continuous similarity score** in [0, 1] (cosine similarity of embeddings)
- A **categorical label** from thresholds (e.g. Low / Moderate / High)
- Optional: **evidence** (which chunks in A align most closely to chunks in B)

### Boundary condition (include in the report)

High similarity can also mean same topic, template reuse, common definitions, or legitimate citation overlap. Frame the tool as **decision support**, not autonomous accusation.

---

## 3. Overall approach (semantic pipeline)

### 3.1 Core idea

1. Split long documents into **meaningful units** (sentences or sliding windows).
2. Encode units into **dense vectors** using a **sentence embedding model**.
3. Compare vectors with **cosine similarity**.
4. Aggregate many pairwise scores into **one document-level score** (max / mean / percentile).
5. **Calibrate thresholds** using your evaluation set (avoid relying only on generic cutoffs like 0.8).

### 3.2 Why chunking matters

A single embedding for an entire long document often **averages out** local structure; localized copying can be missed.

**Default plan:** sentence segmentation + **max chunk similarity** (best matching chunk pair), because plagiarism is often localized.

**Fallback:** whole-document embedding for very short inputs (e.g. single paragraphs or abstracts).

---

## 4. System architecture (modules and responsibilities)

### 4.1 Logical modules (repo layout: `frontend/` then `backend/`)

| Module | Responsibility |
|--------|----------------|
| `backend/plagiarism/config.py` | Model name, thresholds, chunk sizes, device |
| `backend/plagiarism/preprocess.py` | Cleaning, segmentation, optional normalization |
| `backend/plagiarism/embedder.py` | Load SentenceTransformer, batch encoding |
| `backend/plagiarism/similarity.py` | Cosine similarity, aggregation strategies |
| `backend/plagiarism/detector.py` | Orchestrate: preprocess → embed → score → verdict |
| `backend/plagiarism/eval.py` | Metrics, dataset loading, baselines |
| `frontend/streamlit_app.py` | UI (optional but strong for demos) |

### 4.2 Data flow

```
User text/files
  → preprocess
  → list of chunks
  → embeddings (matrix)
  → similarity matrix
  → aggregated score
  → verdict + optional chunk highlights
  → UI / export
```

### 4.3 Deployment shape

- **Local-first**: laptop CPU or GPU.
- **Model caching**: first run downloads weights; later runs can be offline if cached.
- **Reproducibility**: pin library versions and model identifier.

---

## 5. Technology choices (with justification)

### 5.1 Language and environment

- **Python 3.10+**
- **Virtual environment** (`venv`) to freeze dependencies

**Windows PowerShell — environment setup:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 5.2 Embedding model (primary recommendation)

- **`sentence-transformers/all-MiniLM-L6-v2`** (or `all-MiniLM-L6-v384` if you standardize on that variant)

**Why sentence transformers over “raw BERT”:** sentence-level semantics and similarity-friendly training; simpler than hand-built pooling over token-level BERT.

### 5.3 Similarity metric

- **Cosine similarity** on L2-normalized embeddings (standard for semantic search).

### 5.4 Baselines (for the evaluation chapter)

Implement at least one cheap baseline on the same chunking:

- **Jaccard** on word sets after tokenization, or
- **TF-IDF + cosine** (scikit-learn)

Compare to embedding cosine on the **same** dataset and splits.

---

## 6. Preprocessing plan

### 6.1 Default preprocessing (conservative for embeddings)

- Normalize whitespace; remove excessive blank lines.
- Segment into sentences (e.g. NLTK Punkt or spaCy).
- Optional: strip URLs or a defined “references” block **only** if the rule is applied consistently everywhere.

**Avoid aggressive stopword removal as the default:** embeddings usually work better on natural sentences; stopword stripping can hurt semantics unless you show a measured gain on your data.

### 6.2 Chunking rules (define precisely in the report)

**Sentence chunking (default):**

- Split A into sentences `S^A_1 … S^A_n`
- Split B into sentences `S^B_1 … S^B_m`
- Encode all sentences

**Sliding window (optional extension):**

- Window size: e.g. 128–256 tokens (character-based approximation is acceptable for a prototype).
- Stride: ~50% overlap.

### 6.3 Document-level aggregation

Pick **one primary** score for thresholding; optionally report a second for analysis.

- **Localized score:** `max_{i,j} cos(S^A_i, S^B_j)` (sensitive to partial copying).
- **Global score:** cosine between mean-pooled embeddings of A and B.
- **Richer reporting:** e.g. mean of top-k chunk similarities (good narrative for “partial overlap”).

---

## 7. Similarity computation (implementation detail)

### 7.1 Matrices

Let `E_A ∈ R^{n×d}`, `E_B ∈ R^{m×d}` be row embeddings for chunks of A and B.

L2-normalize rows to `Ê_A`, `Ê_B`. Similarity matrix:

`M = Ê_A @ Ê_B^T`  (shape `n × m`)

### 7.2 Document scores (examples)

- **Localized:** `score = max(M)`
- **Global:** `score = cos(mean(Ê_A), mean(Ê_B))`

### 7.3 “Plagiarism percentage” in the UI

Cosine similarity is **not** a probability. Define explicitly:

- `similarity_percent = round(100 * score, 2)` as a **similarity index**, not statistical confidence.

Verdict bands should come from **calibrated** thresholds (Section 8).

---

## 8. Thresholds and calibration

Fixed thresholds (e.g. >0.8 = high) depend on model, chunking, and genre.

### Planned approach

1. Build a **small labeled set** (50–200 pairs is acceptable if methodology is clear): labels such as plagiarized / paraphrased / same-topic-not-plagiarism / unrelated.
2. Sweep thresholds; plot **ROC** (and **PR** if class imbalance).
3. Choose threshold optimizing **F1** or balanced accuracy; report the value.
4. Optionally report two operating points: high-precision vs high-recall.

---

## 9. Datasets and evaluation protocol

### 9.1 Practical data sources

- PAN-style plagiarism corpora (if license/access permits; cite correctly).
- **Synthetic paraphrases:** manual rewrites or controlled paraphrase generation; label consistently.
- **Negative controls:** same topic, different wording (e.g. two Wikipedia summaries).
- **Hard negatives:** same section headings or boilerplate, different body text.

### 9.2 Metrics

- **ROC-AUC** (ranking quality across thresholds).
- At chosen threshold: **precision, recall, F1**, confusion matrix.
- Emphasize **false positives** (ethical and practical impact).

### 9.3 Ablations (short but valuable)

- Sentence chunking vs whole document.
- Max vs mean aggregation.
- Light cleaning on vs off.

---

## 10. UI plan (Streamlit)

### 10.1 Screens

1. Inputs: two text areas; optional file upload (`.txt`; PDF only if you scope parsing).
2. Controls: model id (fixed dropdown), aggregation mode, optional heatmap toggle.
3. Outputs: primary score, verdict band + disclaimer, optional top-k aligned chunk pairs.

### 10.2 Performance

- Cache model with `@st.cache_resource` (load heavy `sentence_transformers` imports inside cached functions where possible).
- **Batch** encode chunks; avoid per-chunk Python loops that reload the model.

---

## 11. Non-functional requirements

### 11.1 Reproducibility

- `requirements.txt` with pinned versions.
- Document: Python version, OS, CPU/GPU, model id, any random seeds.

### 11.2 Ethics and misuse (report section)

- Similarity ≠ intent; ≠ legal finding.
- False positives: citations, definitions, templates.
- Optional mitigations only if validated (e.g. simple citation detection).

---

## 12. Implementation timeline (suggested)

| Week | Focus |
|------|--------|
| 1 | venv, dependencies, embedder, cosine, sentence split, CLI: two texts → score |
| 2 | Similarity matrix, top matches, Streamlit + caching |
| 3 | Labeled set, metrics, ROC, baselines |
| 4 | Thesis packaging: diagrams, tables, limitations, demo recording |

---

## 13. Testing (engineering)

- Empty input, single sentence, long text, Unicode.
- Identical texts → score ≈ 1.0 (within float tolerance).
- Record runtime for a representative “5-page” input on your hardware.

---------

## 14. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| False positives on generic text | hard negatives in training set; tune for precision |
| Unstable scores on very short text | minimum length + user warning |
| Slow long documents | batching; optional cap on max chunks (disclosed) |
| Domain mismatch | state domain limits; future work: domain-specific or multilingual model |

------

## 15. Dependencies (install after venv activation)

Typical packages:

- `sentence-transformers`
- `torch` (pulled in; note CPU vs CUDA)
- `numpy`, `scikit-learn`
- `streamlit`
- `nltk` (tokenization) and/or `spacy`
- `pandas` (optional, for evaluation tables)

```powershell
pip install sentence-transformers scikit-learn numpy streamlit nltk
```

Pin versions in `requirements.txt` before submission.

--------

## 16. Report mapping (quick index)

| Report section | This document |
|----------------|---------------|
| Introduction / problem | §2, ethics in §11.2 |
| Related work / gap | keyword vs semantic (thesis prose) |
| Methodology | §3, §6–§8 |
| System design | §4 |
| Experiments | §9 |
| Results | your tables + ROC from §9 |
| Conclusion | strengths, limitations, responsible use |

-------

## 17. Optional extensions (if time allows)

- Multilingual embedding model for cross-lingual similarity.
- Corpus retrieval: embed references and return top-k semantic matches.
- UI heatmap over chunk–chunk similarity matrix.

-------

*Document generated for project planning. Implementation uses `frontend/` (UI) and `backend/plagiarism/` (core).*
