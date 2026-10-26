#!/usr/bin/env python3
"""Local server for the RSL -> Text web app.

Downloads the Slovo ONNX models (once) into web/models/ and serves web/ on
http://localhost:8000 with the headers multi-threaded ONNX Runtime Web needs.

Usage:
    python serve.py                  # download both models, serve on port 8000
    python serve.py --models mvit16-4 --port 8080
    python serve.py --no-open        # don't open a browser tab

Standard library only, no pip install needed.
"""

import argparse
import functools
import http.server
import shutil
import sys
import threading
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"
MODELS_DIR = WEB_DIR / "models"
BASE_URL = "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/slovo/models/mvit/onnx/"
MODELS = {
    "mvit16-4": 140510654,
    "mvit32-2": 140785735,
}


def download(name: str, size: int) -> None:
    target = MODELS_DIR / f"{name}.onnx"
    if target.exists() and target.stat().st_size == size:
        print(f"[ok] {name} already downloaded")
        return
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".onnx.part")
    url = BASE_URL + f"{name}.onnx"
    print(f"[..] downloading {name} ({size / 1048576:.0f} MB) from {url}")
    with urllib.request.urlopen(url) as response, open(partial, "wb") as out:
        total = int(response.headers.get("Content-Length") or size)
        done = 0
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            sys.stdout.write(f"\r     {done / 1048576:6.1f} / {total / 1048576:.1f} MB")
            sys.stdout.flush()
    print()
    if partial.stat().st_size != size:
        partial.unlink()
        raise RuntimeError(f"{name}: download incomplete, please run again")
    shutil.move(partial, target)
    print(f"[ok] {name} saved to {target.relative_to(ROOT)}")


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".js": "text/javascript",
        ".mjs": "text/javascript",
        ".wasm": "application/wasm",
        ".onnx": "application/octet-stream",
    }

    def end_headers(self):
        # Cross-origin isolation enables SharedArrayBuffer -> multi-threaded WASM.
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        if not self.path.startswith("/models/"):
            self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def log_message(self, format, *args):
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to allow other devices (camera then needs https)")
    parser.add_argument("--models", default=",".join(MODELS), help="comma-separated: " + ", ".join(MODELS))
    parser.add_argument("--no-download", action="store_true", help="skip model download")
    parser.add_argument("--no-open", action="store_true", help="don't open the browser")
    args = parser.parse_args()

    if not args.no_download:
        for name in filter(None, (m.strip() for m in args.models.split(","))):
            if name not in MODELS:
                parser.error(f"unknown model {name!r}, choose from {', '.join(MODELS)}")
            download(name, MODELS[name])

    handler = functools.partial(Handler, directory=str(WEB_DIR))
    server = http.server.ThreadingHTTPServer((args.host, args.port), handler)
    url = f"http://localhost:{args.port}/" if args.host in ("127.0.0.1", "0.0.0.0") else f"http://{args.host}:{args.port}/"
    print(f"\nRSL -> Text is running at {url}  (Ctrl+C to stop)")
    if not args.no_open:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
