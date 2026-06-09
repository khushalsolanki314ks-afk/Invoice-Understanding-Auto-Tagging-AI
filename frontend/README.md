Frontend (BillMart)
====================

This is the single-file frontend UI for the BillMart Invoice AI demo.

Quick start (assumes backend running at http://localhost:8000):

1. Serve static files (option A: Python simple server)

```powershell
cd invoice-ai/frontend
python -m http.server 8080
# Open http://localhost:8080 in your browser
```

2. Or open the file directly in your browser: open `invoice-ai/frontend/index.html`.

Features
- Upload invoices (Drag & Drop or click).
- Select OCR language (`eng`, `hin`, `fra`, `spa`).
- Toggle `Real OCR` to request the backend use Tesseract instead of mock pipeline.
- Demo scenarios (clean, duplicate, high_amount, date_issue, unknown_buyer).

Notes
- For `Real OCR` to work the backend must have Tesseract installed and language packs present.
- The UI will fall back to mock data when the backend is not reachable.

Run the full stack locally
1. Start backend (from `invoice-ai/backend`):
```powershell
cd invoice-ai/backend
.\env\Scripts\Activate.ps1
uvicorn main:app --reload --port 8000
```
2. Serve frontend (see step 1 above) and open the page.

If you want, I can add a small Express or Python wrapper to serve both frontend and backend together.
