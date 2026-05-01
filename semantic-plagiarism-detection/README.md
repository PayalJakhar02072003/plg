# Semantic plagiarism detection (prototype)

**Layout**

| Folder | Role |
|--------|------|
| `frontend/` | Streamlit UI only (`streamlit_app.py`, `components/`) |
| `backend/plagiarism/` | Scoring pipeline: preprocess → embeddings → cosine → verdict |
| `docs/` | Implementation plan and thesis notes |

Newer `transformers` stacks may import vision helpers that require **`torchvision`**. It is listed in `requirements.txt` so `ModuleNotFoundError: No module named 'torchvision'` should not occur after reinstall.

## Setup (Windows PowerShell)

```powershell
cd D:\plg\semantic-plagiarism-detection
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If NLTK sentence splitting fails once:

```powershell
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

## Start the frontend (after backend code is in place)

From the **project root** (folder that contains `frontend` and `backend`):

```powershell
cd D:\plg\semantic-plagiarism-detection
.\.venv\Scripts\Activate.ps1
streamlit run frontend\streamlit_app.py
```

`streamlit_app.py` adds `backend` to `sys.path`, so you do **not** need `PYTHONPATH` for the UI.

## Run the backend from the CLI

**You must be inside the project folder** (`semantic-plagiarism-detection`), not `D:\plg` alone. The `plagiarism` package lives at `backend\plagiarism\`.

**Easiest (no `PYTHONPATH` typos):**

```powershell
cd D:\plg\semantic-plagiarism-detection
.\.venv\Scripts\Activate.ps1
.\run_cli.ps1 "First text." "Second text."
```

**Manual:** use **`$env:PYTHONPATH`** (do not prefix with `p` — that breaks PowerShell).

```powershell
cd D:\plg\semantic-plagiarism-detection
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "$PWD\backend"
python -m plagiarism "First text." "Second text."
```

Optional: `--file-a`, `--file-b`, `--aggregation`, `--model`.

---

## Deploy (share a public link — e.g. for your professor)

The easiest path for a **Streamlit** app is **[Streamlit Community Cloud](https://streamlit.io/cloud)** (free **public** app URL). Your professor opens the link in a browser; no install needed on their side.

### Steps

1. **Put the project on GitHub** (create a new repository, push this folder as the repo root).  
   - Do **not** commit `.venv/` (it should stay in `.gitignore`).

2. Go to **[share.streamlit.io](https://share.streamlit.io)** → sign in with GitHub → **New app**.

3. Pick your **repository** and **branch** (usually `main`).

4. Under **Advanced settings** → **Main file path**, set:

   `frontend/streamlit_app.py`

   (Streamlit Cloud defaults to a root `streamlit_app.py`; this project keeps the UI under `frontend/`, so this path is required.)

5. **Deploy.** The first build can take **several minutes** (installs PyTorch, `sentence-transformers`, then downloads the model on first use). If the app **sleeps** after inactivity, the next visitor may wait again while the app wakes.

6. **Optional — Hugging Face rate limits:** if model download fails on Cloud, add a secret **`HF_TOKEN`** in the app’s Streamlit **Secrets** (a read token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)). Do not commit tokens into the repo.

### If Community Cloud fails (memory / build size)

PyTorch + embeddings are heavy. If the free tier **runs out of memory** or the build fails, alternatives:

- **[Hugging Face Spaces](https://huggingface.co/spaces)** — create a **Streamlit** Space and copy the same `frontend/` + `backend/` + `requirements.txt` layout (adjust the Space “App file” to your entry script), or  
- Share a **screen recording** plus the **GitHub link** so the professor can run locally with `README` instructions.

---

## Professor email (short template)

> Live demo: **&lt;paste your streamlit.app URL&gt;**  
> Source code: **&lt;paste your GitHub repo URL&gt;**  
> The first load may take 1–2 minutes while the model downloads. This tool reports **semantic similarity** only, not legal plagiarism.
