# Document uploads for ChristmanVideoEngine

ChristmanVideoEngine extracts text from files dropped in the **Prompt → Source document** field.

## Supported formats

- **PDF** — text layer via `pypdf`; scanned pages via `christman_ocr_shared.py` (PaddleOCR)
- **HTML / HTM** — body text extracted for editing in the prompt box
- **TXT / MD** — plain text
- **PNG / JPG / …** — OCR via Christman OCR

## Editable workflow

1. Drop document in `/video` Prompt UI
2. Text loads into the **Prompt** textarea — edit freely
3. Render uses **your edited prompt** (document body is not double-injected)

## OCR module

`christman_ocr_shared.py` lives at the Christman-Sound repo root.

Optional heavy deps (scanned PDFs / photos only):

```bash
pip install paddleocr paddlepaddle pymupdf pillow numpy
```