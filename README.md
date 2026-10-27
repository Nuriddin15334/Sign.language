# RSL → Text

Russian Sign Language (RSL) recognition in the browser. It runs the pre-trained
[Slovo](https://github.com/ai-forever/slovo) MViTv2 models (1,000 signs plus the
fingerspelling alphabet) on your webcam or a video file. Video never leaves your device.

## Run it locally (recommended)

Requires Python 3.8+. `pip install onnx` is optional and enables the GPU.

```bash
pip install onnx   # optional: lets serve.py build the GPU versions of the models
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

The workflow downloads the models and builds their GPU versions at deploy time, so they never go into git.
The site is then at `https://<user>.github.io/<repo>/`.

## How it works

```
camera ──► sample every 2nd/4th frame ──► letterbox 224×224, normalize
       ──► window of 32/16 frames ──► MViTv2 (ONNX Runtime Web in a Web Worker: WebGPU, or multi-threaded WASM)
       ──► 1001 class probabilities ──► threshold + dedupe ──► transcript ──► speech (ru-RU)
```

| Model      | Window             | Slovo top-1 | CPU (4 cores)      |
|------------|--------------------|-------------|--------------------|
| `mvit16-4` | 16 frames, every 4 | 58%         | ~3 s / prediction  |
| `mvit32-2` | 32 frames, every 2 | 64%         | ~7 s / prediction  |

On a GPU, a prediction should take a fraction of a second; the page shows the measured
time under **Inference**.

- Preprocessing matches Slovo's `demo.py` (RGB, letterbox with gray 114 padding, ImageNet mean/std).
- **GPU:** ONNX Runtime Web's WebGPU backend can't run grouped (depthwise) 3D convolutions,
  which MViTv2 uses for attention pooling. `tools/convert_webgpu.py` rewrites each one as an
  exact equivalent (a sum of depthwise 2D convolutions over time-shifted slices), giving the
  `*-webgpu.onnx` files. Outputs match the originals to ~1e-7. With **Run on: Auto**, the page
  uses the GPU when the browser has WebGPU (Chrome/Edge) and otherwise falls back to the CPU.
- **CPU:** multi-threading needs cross-origin isolation. `serve.py` sends the COOP/COEP
  headers; on static hosts `web/sw.js` (a service worker) adds them.

## Limitations

- Isolated signs only: sign one word, pause, sign the next. No continuous sentences yet.
- The model was trained on Slovo signers (RSL as used in Russia). Signs used in Uzbekistan
  may differ; fine-tuning on local signers will be needed.
- Without a GPU, inference is a few seconds per prediction, so the transcript lags behind the signing.

## Files

- `web/index.html`, `web/style.css`, `web/app.js`: UI, camera capture, preprocessing, transcript
- `web/worker.js`: ONNX Runtime inference in a Web Worker
- `web/classes.js`: the 1001 Slovo labels
- `web/sw.js`: service worker for cross-origin isolation on static hosts
- `web/sample.mp4`: sample clip from the Slovo repo
- `serve.py`: local server and model downloader
- `tools/convert_webgpu.py`: makes the models runnable on the GPU (WebGPU)

## Credits and license

Models, labels and the sample video come from
[Slovo](https://github.com/ai-forever/slovo) by ai-forever (Kapitanov et al., 2023),
licensed under a variant of CC BY-SA 4.0. See their
[license](https://github.com/hukenovs/slovo/blob/master/license/en_us.pdf).
