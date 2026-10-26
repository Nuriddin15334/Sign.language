// Runs ONNX Runtime off the main thread so camera capture and the UI stay smooth
// while the model works (a CPU inference takes a few seconds).
//
// The CPU (WASM) backend is used on purpose: the Slovo MViTv2 models use grouped
// 3D convolutions that ONNX Runtime's WebGPU backend can't run (as of 1.30).
const ORT_VERSION = '1.30.0';
const ORT_CDN = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;

importScripts(ORT_CDN + 'ort.wasm.min.js');

let session = null;
let inputName = null;
let outputName = null;

function readInputFrames(s) {
  // Input shape is [1, 1, 3, T, 224, 224]; T is the window length.
  const meta = s.inputMetadata && s.inputMetadata[0];
  const shape = meta && meta.shape;
  if (Array.isArray(shape) && typeof shape[3] === 'number') return shape[3];
  return null;
}

self.onmessage = async (e) => {
  const msg = e.data;
  try {
    if (msg.type === 'load') {
      if (session) {
        await session.release();
        session = null;
      }
      ort.env.wasm.wasmPaths = ORT_CDN;
      ort.env.wasm.numThreads = msg.threads;
      ort.env.logLevel = 'error';
      session = await ort.InferenceSession.create(new Uint8Array(msg.buffer), {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
      });
      inputName = session.inputNames[0];
      outputName = session.outputNames[0];
      self.postMessage({ type: 'loaded', threads: ort.env.wasm.numThreads, frames: readInputFrames(session) });
    } else if (msg.type === 'run') {
      const t0 = performance.now();
      const input = new ort.Tensor('float32', msg.data, [1, 1, 3, msg.frames, 224, 224]);
      const out = await session.run({ [inputName]: input });
      const probs = Float32Array.from(out[outputName].data);
      self.postMessage({ type: 'result', id: msg.id, probs, ms: performance.now() - t0 }, [probs.buffer]);
    }
  } catch (err) {
    self.postMessage({ type: 'error', id: msg.id, stage: msg.type, message: String((err && err.message) || err) });
  }
};
