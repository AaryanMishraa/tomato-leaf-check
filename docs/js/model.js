// On-phone inference: the same steps as the Streamlit app, in the browser.
// 1. Resize the photo to 224 x 224 with TensorFlow's bilinear rule (half-pixel centres), round to whole values.
// 2. MobileNetV2 (TF.js graph model) gives a 7 x 7 x 1280 feature map.
// 3. Average pooling + the logistic-regression weights give 7 probabilities.
// 4. Grad-CAM: for a linear head, the gradient of a class score with respect to the
//    feature map is that class's weight vector, so the heatmap is relu(features . W[:, c]).
/* global tf */

const SIZE = 224;
let graph = null, meta = null, W = null, B = null, Wt = null;
export let backendName = '';

export async function loadModel(base = '') {
  if (tf.wasm && tf.wasm.setWasmPaths) tf.wasm.setWasmPaths(base + 'vendor/');
  for (const b of ['wasm', 'webgl', 'cpu']) {
    try {
      if (await tf.setBackend(b)) { await tf.ready(); backendName = b; break; }
    } catch (e) { /* try the next backend */ }
  }
  meta = await (await fetch(base + 'data/model.json')).json();
  graph = await tf.loadGraphModel(base + 'model/model.json');
  W = tf.tensor2d(meta.W);                     // 1280 x 7
  B = tf.tensor1d(meta.b);
  // column-major copy of W for fast heatmaps: Wt[c][k]
  Wt = meta.classes.map((_, c) => Float32Array.from(meta.W, row => row[c]));
  tf.tidy(() => graph.execute(tf.zeros([1, SIZE, SIZE, 3])));   // warm up
  return meta;
}

export function getMeta() { return meta; }

// Round half to even, as numpy.round does.
function roundEven(v) {
  const r = Math.round(v);
  return (Math.abs(v % 1) === 0.5 && r % 2 !== 0) ? r - 1 : r;
}

// TensorFlow's bilinear resize (half_pixel_centers=True) for one axis.
function axisWeights(inSize, outSize) {
  // float32 arithmetic, as in TensorFlow's CPU kernel, so rounding matches exactly
  const F = Math.fround, scale = F(inSize / outSize), lo = new Int32Array(outSize), hi = new Int32Array(outSize), t = new Float32Array(outSize);
  for (let i = 0; i < outSize; i++) {
    const x = F(F(F(i + 0.5) * scale) - 0.5), f = Math.floor(x);
    lo[i] = Math.max(f, 0); hi[i] = Math.min(Math.ceil(x), inSize - 1); t[i] = F(x - f);
  }
  return { lo, hi, t };
}

// Draws an image (img, bitmap or canvas) and returns the 224 x 224 x 3 model input (values 0-255).
export function prepare(source) {
  let w = source.naturalWidth || source.width, h = source.naturalHeight || source.height;
  const maxArea = 16e6;                          // stay under phone canvas limits
  if (w * h > maxArea) { const k = Math.sqrt(maxArea / (w * h)); w = Math.floor(w * k); h = Math.floor(h * k); }
  const cv = document.createElement('canvas'); cv.width = w; cv.height = h;
  const ctx = cv.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(source, 0, 0, w, h);
  const ax = axisWeights(w, SIZE), ay = axisWeights(h, SIZE);
  const rows = new Map();
  const row = y => { if (!rows.has(y)) rows.set(y, ctx.getImageData(0, y, w, 1).data); return rows.get(y); };
  const out = new Float32Array(SIZE * SIZE * 3), F = Math.fround;
  for (let oy = 0; oy < SIZE; oy++) {
    const top = row(ay.lo[oy]), bot = row(ay.hi[oy]), ty = ay.t[oy];
    for (let ox = 0; ox < SIZE; ox++) {
      const l = ax.lo[ox] * 4, r = ax.hi[ox] * 4, tx = ax.t[ox], o = (oy * SIZE + ox) * 3;
      for (let c = 0; c < 3; c++) {
        const tp = F(top[l + c] + F(F(top[r + c] - top[l + c]) * tx));
        const bt = F(bot[l + c] + F(F(bot[r + c] - bot[l + c]) * tx));
        out[o + c] = Math.min(255, Math.max(0, roundEven(F(tp + F(F(bt - tp) * ty)))));
      }
    }
  }
  return out;
}

// Runs the model. Returns probabilities, the raw feature map and the time taken.
export async function diagnose(pixels) {
  const t0 = performance.now();
  const x = tf.tensor4d(pixels, [1, SIZE, SIZE, 3]);
  const feat = graph.execute(x);
  const probsT = tf.tidy(() => tf.softmax(tf.add(tf.matMul(tf.mean(feat, [1, 2]), W), B)));
  const [probs, features] = await Promise.all([probsT.data(), feat.data()]);
  const ms = performance.now() - t0;
  x.dispose(); feat.dispose(); probsT.dispose();
  const order = Array.from(probs.keys()).sort((a, b) => probs[b] - probs[a]);
  return { probs: Array.from(probs), order, features, ms };
}

// 7 x 7 Grad-CAM map for class c, scaled to 0..1.
export function camFor(features, c) {
  const w = Wt[c], n = w.length, cam = new Float32Array(49);
  let mx = 0;
  for (let i = 0; i < 49; i++) {
    let s = 0;
    for (let k = 0, o = i * n; k < n; k++) s += features[o + k] * w[k];
    cam[i] = Math.max(0, s); mx = Math.max(mx, cam[i]);
  }
  for (let i = 0; i < 49; i++) cam[i] /= (mx + 1e-8);
  return cam;
}

const INFERNO = '000004,040312,0b0724,150b37,230c4c,310a5c,3e0966,4c0c6b,5a116e,67166e,741a6e,801f6c,8f2469,9b2964,a82e5f,b43359,c13a50,cc4248,d74b3f,e05536,e9612b,ef6e21,f57b17,f8890c,fb9906,fca80d,fbb81d,f9c72f,f5d949,f2e865,f3f586,fcffa4'
  .split(',').map(h => [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)]);

function inferno(v) {
  const x = Math.min(1, Math.max(0, v)) * (INFERNO.length - 1), i = Math.floor(x), t = x - i;
  const a = INFERNO[i], b = INFERNO[Math.min(i + 1, INFERNO.length - 1)];
  return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
}

// Paints the heatmap, upsampled with the same bilinear rule, onto a canvas of any size.
export function paintHeat(canvas, cam) {
  const w = canvas.width, h = canvas.height, ctx = canvas.getContext('2d');
  const img = ctx.createImageData(w, h), ax = axisWeights(7, w), ay = axisWeights(7, h);
  for (let y = 0; y < h; y++) {
    const r0 = ay.lo[y] * 7, r1 = ay.hi[y] * 7, ty = ay.t[y];
    for (let x = 0; x < w; x++) {
      const a = cam[r0 + ax.lo[x]], b = cam[r0 + ax.hi[x]], c = cam[r1 + ax.lo[x]], d = cam[r1 + ax.hi[x]];
      const top = a + (b - a) * ax.t[x], bot = c + (d - c) * ax.t[x];
      const [R, G, Bc] = inferno(top + (bot - top) * ty), o = (y * w + x) * 4;
      img.data[o] = R; img.data[o + 1] = G; img.data[o + 2] = Bc; img.data[o + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

// Median time of n runs on the given input (for the speed test).
export async function benchmark(pixels, n = 10) {
  const times = [];
  for (let i = 0; i < n; i++) times.push((await diagnose(pixels)).ms);
  times.sort((a, b) => a - b);
  return times[Math.floor(n / 2)];
}
