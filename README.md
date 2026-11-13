# LeafLens AI

LeafLens identifies leaf images with the Hugging Face Vision Transformer (ViT)
classifier and displays its top five predictions. The existing upload and
results workflow is backed by an independent Flask inference service. A
source-linked plant report is retrieved separately; report availability does
not block image prediction.

## Architecture

- **Frontend:** TanStack Start, React, TypeScript, and Vite.
- **Main Flask API (`8000`):** retains `POST /predict`, forwards the uploaded
  image to the ViT service, and provides `GET /plant-info`.
- **ViT service (`5001`):** loads `dhivyasri-d/vit-plant-classifier`, from the
  `vit-plant-push` subfolder, using its own `AutoImageProcessor`. It runs CPU
  inference and returns the model's 80-class output.
- **Plant reports:** GBIF supplies taxonomy and common names; Europe PMC
  supplies literature search results. Edibility and safety are reported as
  unknown unless supported by verified information. A prediction or its
  confidence is not evidence of identity or safety.

The existing MobileNetV2 checkpoint files are legacy assets and are not used
for active predictions. They are already tracked in this repository and are
retained as requested; the ignore rules prevent adding additional weight files.
The active ViT weights are loaded from the Hugging Face cache, not copied into
the repository. The separate evaluation pipeline and its dataset requirements
are documented in `ML/medicinal_vit/evaluation/README.md`.

## Prerequisites

Windows PowerShell instructions are shown first.

- Git
- Python 3.11 (64-bit; the backend environments have been tested with 3.11)
- Node.js 22.12+ (or 20.19+); the frontend is tested with Node 24 and npm 11
- Internet access for package installation and the first ViT model download
- Enough disk space for PyTorch and the model cache

The main Flask API and ViT inference service use separate Python environments.
Do not install the ViT dependencies into the existing TensorFlow environment.

## Clone the repository

```powershell
git clone https://github.com/VJKhanai/Leaflens.git
Set-Location Leaflens
```

## Set up the Python environments

Create and install the main Flask API environment:

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

In a second PowerShell terminal, create and install the separate ViT
environment:

```powershell
Set-Location Leaflens
py -3.11 -m venv vit-venv
.\vit-venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r vit-requirements.txt
```

The ViT requirements include PyTorch, Transformers, Pillow, and
Hugging Face Hub. The requirements files contain runtime dependencies only;
the Python tests use the standard library's `unittest`.

## Hugging Face model setup

The configured model is currently public and does not require authentication.
The first ViT service startup calls `from_pretrained` and automatically
downloads the processor and checkpoint into the Hugging Face cache (normally
under your user profile, outside this repository). Do not download or copy
model weights into the project directory.

If Hugging Face reports that access is restricted, first request/accept access
on the model page, then authenticate inside the ViT environment:

```powershell
hf auth login
```

Enter the token at the CLI prompt; do not put it in source control or
`.env.example`. The Hugging Face client uses the saved login or `HF_TOKEN`
provided by the process environment. To place the cache somewhere else, set
`HF_HOME` before starting the ViT service.

## Environment configuration

`.env.example` documents optional service URL overrides. Defaults work for a
local development setup. Vite reads a local `frontend/.env` file if you need
the `VITE_*` overrides; copy the example without committing the local file:

```powershell
Copy-Item .env.example frontend\.env
```

The Flask process reads `MEDICINAL_VIT_PREDICT_URL` from its process
environment; Python does not automatically load `.env` files. For a temporary
PowerShell override:

```powershell
$env:MEDICINAL_VIT_PREDICT_URL = "http://127.0.0.1:5001/medicinal/predict"
```

No API key is required for GBIF or Europe PMC. Keep any future credentials out
of `.env.example` and out of Git.

## Start the services

Run each service in its own PowerShell terminal from the repository root.

**Terminal 1 — ViT inference service:**

```powershell
.\vit-venv\Scripts\Activate.ps1
python -m backend.medicinal_vit.app
```

**Terminal 2 — main Flask API:**

```powershell
.\venv\Scripts\Activate.ps1
python -m backend.app
```

**Terminal 3 — frontend:**

```powershell
Set-Location frontend
npm install
npm run dev
```

Open the Vite URL printed in the terminal (default `http://localhost:8080`).
The frontend proxies `/predict` and `/plant-info` to the main API.

| Service | Default address | Endpoints |
| --- | --- | --- |
| Frontend | `http://localhost:8080` | Existing upload and results UI |
| Main Flask API | `http://127.0.0.1:8000` | `POST /predict`, `GET /plant-info` |
| ViT Flask service | `http://127.0.0.1:5001` | `POST /medicinal/predict`, `GET /medicinal/health` |

Quick health checks:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:5001/medicinal/health
```

The first ViT startup can take longer while the model is downloaded and loaded.

## Tests and build

Run the main API/report tests in the main environment:

```powershell
.\venv\Scripts\Activate.ps1
python -m unittest backend.test_plant_reports backend.medicinal_vit.test_integration
```

Run the ViT service and evaluation tests in the ViT environment:

```powershell
.\vit-venv\Scripts\Activate.ps1
python -m unittest backend.medicinal_vit.test_service ML.medicinal_vit.evaluation.test_evaluate_vit
```

Run frontend checks:

```powershell
Set-Location frontend
npm install
npm test
npx tsc --noEmit
npm run lint
npm run build
```

## Common setup issues

- **PowerShell blocks activation:** use
  `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal,
  or invoke the environment's `python.exe` directly. This setting lasts only
  for the current process.
- **`ModuleNotFoundError`:** check that the correct environment is activated,
  then install its corresponding requirements file.
- **Model download/load error:** check internet access and free disk space. If
  the model is no longer public, accept access and run `hf auth login` in the
  ViT environment. The startup error names the model and cache/auth checks.
- **Port already in use:** close the previous service terminal, or identify
  the process listening on `8000`, `5001`, or `8080` before restarting.
- **Frontend requests fail:** start both Flask services before using the
  frontend and confirm their health endpoints respond.
- **Lint reports `Delete ␍` errors:** the existing Prettier rule expects LF
  endings while this Windows checkout uses CRLF. This does not prevent tests,
  type checking, or the production build; avoid reformatting unrelated files
  solely to silence the repository-wide line-ending warnings.
- **Missing model files:** the service downloads from Hugging Face on startup
  when the model is available. The downloaded files belong in the Hugging Face
  cache, not in this repository.
