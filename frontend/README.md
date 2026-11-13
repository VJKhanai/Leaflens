# Welcome to your Lovable project

This project was built with [Lovable](https://lovable.dev).

## Build with Lovable

Open your project in the [Lovable editor](https://lovable.dev) and keep building.

- **Ship faster**: describe what you want to build and Lovable handles the code.
- **Stay in sync**: connect the project to GitHub and every change made in Lovable is committed straight to your repository.
- **Full ownership**: this code is yours. Push to your repository and your changes sync back into Lovable, ready for your next prompt.

## Development

Prefer working locally? You need Node.js and npm — [install with nvm](https://github.com/nvm-sh/nvm#installing-and-updating).

```sh
git clone <this-repository-url>
cd <repository-name>
npm i
npm run dev
```

## Plant identification services

The existing `POST /predict` endpoint forwards images to the independent ViT
service at `POST /medicinal/predict`; the frontend continues to use the same
upload workflow. The ViT process runs in `vit-venv` on port 5001, and the
existing Flask API runs on port 8000. During local development Vite proxies
`/predict` and `/plant-info` to port 8000.

Start the services in separate PowerShell terminals from the repository root:

```powershell
# Existing Flask environment
python -m backend.app

# ViT environment
.\vit-venv\Scripts\Activate.ps1
python -m backend.medicinal_vit.app

# Frontend terminal
Set-Location frontend
npm run dev
```

The Flask API also exposes `GET /plant-info?name=<plant-name>`. Structured
reports use GBIF for high-confidence taxonomy, vernacular names, and attributed
descriptions. Common-name queries resolve only when GBIF has an exact vernacular
match to one accepted species; ambiguous names remain unresolved. Europe PMC
provides linked literature search results. Literature
matches are not treated as proof of clinical benefit. Edibility, toxicity,
medicinal-use, and safety fields remain explicitly unknown unless supported by
verified sources; the ViT prediction itself never establishes identity or
safety. References include direct links and access dates where available.

## Built with

- TanStack Start
- TypeScript
- React
- Tailwind CSS
