# RSL → Text

Russian Sign Language (RSL) recognition in the browser. It runs the pre-trained
[Slovo](https://github.com/ai-forever/slovo) MViTv2 models (1,000 signs plus the
fingerspelling alphabet) on your webcam or a video file. Video never leaves your device.

## Run it locally (recommended)

Requires Python 3.8+ and no other dependencies.

```bash
python serve.py
```

This downloads the two models (~140 MB each, once) into `web/models/`, starts a
server and opens http://localhost:8000. Then:

1. Click **Load model**.
2. Click **Sample video** to check it works (it should recognize «Привет!»), or
   **Start camera** and sign.

Options: `--models mvit16-4` (download only the faster model), `--port 8080`, `--no-open`.

## Host it on GitHub Pages

1. In the repo: **Settings → Pages → Build and deployment → Source: GitHub Actions**.
2. Re-run the "Deploy web app to GitHub Pages" workflow (Actions tab), or push a change under `web/`.

The workflow downloads the models at deploy time, so they never go into git.
The site is then at `https://<user>.github.io/<repo>/`.

## How it works

```
camera ──► sample every 2nd/4th frame ──► letterbox 224×224, normalize
       ──► window of 32/16 frames ──► MViTv2 (ONNX Runtime Web, WASM, multi-threaded, in a Web Worker)
       ──► 1001 class probabilities ──► threshold + dedupe ──► transcript ──► speech (ru-RU)
```

| Model      | Window             | Slovo top-1 | Speed (4-core CPU) |
|------------|--------------------|-------------|--------------------|
| `mvit16-4` | 16 frames, every 4 | 58%         | ~3 s / prediction  |
| `mvit32-2` | 32 frames, every 2 | 64%         | ~7 s / prediction  |

- Preprocessing matches Slovo's `demo.py` (RGB, letterbox with gray 114 padding, ImageNet mean/std).
- Multi-threading needs cross-origin isolation. `serve.py` sends the COOP/COEP headers; on
  static hosts `web/sw.js` (a service worker) adds them.
- The WebGPU backend isn't used: ONNX Runtime Web 1.30 can't run these models' grouped 3D
  convolutions on WebGPU.

## Limitations

- Isolated signs only: sign one word, pause, sign the next. No continuous sentences yet.
- The model was trained on Slovo signers (RSL as used in Russia). Signs used in Uzbekistan
  may differ; fine-tuning on local signers will be needed.
- CPU inference is a few seconds per prediction, so the transcript lags behind the signing.

## Files

- `web/index.html`, `web/style.css`, `web/app.js`: UI, camera capture, preprocessing, transcript
- `web/worker.js`: ONNX Runtime inference in a Web Worker
- `web/classes.js`: the 1001 Slovo labels
- `web/sw.js`: service worker for cross-origin isolation on static hosts
- `web/sample.mp4`: sample clip from the Slovo repo
- `serve.py`: local server and model downloader

## Credits and license

Models, labels and the sample video come from
[Slovo](https://github.com/ai-forever/slovo) by ai-forever (Kapitanov et al., 2023),
licensed under a variant of CC BY-SA 4.0. See their
[license](https://github.com/hukenovs/slovo/blob/master/license/en_us.pdf).
