# models/documents

This folder holds no model and no document. It holds this note.

## What this note used to say

It described a document-upload screen for ChristmanVideoEngine: a `/video`
Prompt UI, a "Source document" field, PDF text through `pypdf`. None of that is
in this repository. ChristmanVideoEngine is a separate project, and its upload
screen lives there. (Corrected 2026-10-28.)

## What this repository does have for documents

`christman_ocr_shared.py`, at the repository root. It is the one document
reader here:

- `ChristmanOCR(being_name=...).read_document(path)` reads an image
  (`.jpg .jpeg .png .bmp .tiff .tif .webp`) or a PDF. PDF pages are rendered at
  300 dpi with PyMuPDF and read with PaddleOCR. There is no `pypdf` text-layer
  path; every page goes through OCR.
- `read_screen()` reads the screen once; `watch_screen()` keeps reading it.
- Lines below confidence 0.50 are dropped.
- Text found is sent to Derek, a Christman being, over a local WebSocket to be
  read aloud. If Derek does not answer, the read-aloud raises
  `SynthesisUnavailable` with the reason. There is no fallback voice.

What it needs beyond the base install:

```bash
pip install paddleocr paddlepaddle pymupdf pillow
```

Not offline yet: PaddleOCR fetches its model files from the network the first
time it is constructed. Everett's rule is offline, no downloads; this is an
open item (tests/test_offline.py lists it).
