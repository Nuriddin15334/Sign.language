// Runs ONNX Runtime off the main thread so camera capture and the UI stay smooth
// while the model works. The backend comes from the URL (worker.js?backend=...):
//   webgpu - GPU via ONNX Runtime's native WebGPU backend (needs the -webgpu model)
//   wasm   - multi-threaded CPU
const ORT_VERSION = '1.30.0';
const ORT_CDN = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
const BACKEND = new URLSearchParams(self.location.search).get('backend') === 'webgpu' ? 'webgpu' : 'wasm';

importScripts(ORT_CDN + (BACKEND === 'webgpu' ? 'ort.webgpu.min.js' : 'ort.wasm.min.js'));

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

async function run(data, frames) {
  const input = new ort.Tensor('float32', data, [1, 1, 3, frames, 224, 224]);
  const out = await session.run({ [inputName]: input });
  return Float32Array.from(out[outputName].data);
}

self.onmessage = async (e) => {
  const msg = e.data;
  try {
    if (msg.type === 'load') {
      if (BACKEND === 'webgpu' && !(self.navigator.gpu && await self.navigator.gpu.requestAdapter())) {
        throw new Error('WebGPU is not available in this browser');
      }
      ort.env.wasm.wasmPaths = ORT_CDN;
      ort.env.wasm.numThreads = msg.threads;
      ort.env.logLevel = 'error';
      session = await ort.InferenceSession.create(new Uint8Array(msg.buffer), {
        executionProviders: [BACKEND],
        graphOptimizationLevel: 'all',
      });
      inputName = session.inputNames[0];
      outputName = session.outputNames[0];
      const frames = readInputFrames(session);
      let warmupMs = null;
      if (BACKEND === 'webgpu') {
        // Compiles the GPU shaders now rather than on the first real prediction,
        // and surfaces unsupported operators while the CPU fallback is still possible.
        const t0 = performance.now();
        await run(new Float32Array(3 * (frames || 32) * 224 * 224), frames || 32);
        warmupMs = performance.now() - t0;
      }
      self.postMessage({ type: 'loaded', id: msg.id, threads: ort.env.wasm.numThreads, frames, warmupMs });
    } else if (msg.type === 'run') {
      const t0 = performance.now();
      const probs = await run(msg.data, msg.frames);
      self.postMessage({ type: 'result', id: msg.id, probs, ms: performance.now() - t0 }, [probs.buffer]);
    }
  } catch (err) {
    self.postMessage({ type: 'error', id: msg.id, message: String((err && err.message) || err) });
  }
};
