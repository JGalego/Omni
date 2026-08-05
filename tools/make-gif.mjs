// PNG frames → one animated GIF.
//
//   node tools/make-gif.mjs <framedir> <out.gif> [--fps 10]
//
// Written from the two specs rather than pulled from npm, for the same reason the
// reference implementation has no dependencies: this repository should be
// buildable from a checkout and a toolchain. PNG (RFC 2083) needs only `zlib`,
// which Node ships; GIF89a needs an LZW encoder, which is fifty lines.
//
// Three things keep the file small enough to sit in a README:
//
//   * a 256-colour palette chosen by frequency across every frame, so the flat
//     UI colours are exact and only antialiasing blends get approximated;
//   * identical consecutive frames merged into one with a longer delay, so
//     holding still on a caption costs nothing;
//   * each frame stored as the bounding box of what changed, with unchanged
//     pixels left transparent — a click that repaints one pane does not re-encode
//     the other two.

import { readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { inflateSync } from 'node:zlib';
import { join } from 'node:path';

// ------------------------------------------------------------------- PNG in --

/** Decodes a non-interlaced 8-bit RGB/RGBA PNG to {w, h, rgb: Uint8Array}. */
function decodePng(buf) {
  if (buf.readUInt32BE(0) !== 0x89504e47) throw new Error('not a PNG');
  let p = 8, w = 0, h = 0, colour = 0, depth = 0, idat = [];
  while (p < buf.length) {
    const len = buf.readUInt32BE(p);
    const type = buf.toString('ascii', p + 4, p + 8);
    const data = buf.subarray(p + 8, p + 8 + len);
    if (type === 'IHDR') {
      w = data.readUInt32BE(0); h = data.readUInt32BE(4);
      depth = data[8]; colour = data[9];
      if (depth !== 8 || (colour !== 2 && colour !== 6)) {
        throw new Error(`unsupported PNG: depth ${depth}, colour type ${colour}`);
      }
      if (data[12] !== 0) throw new Error('interlaced PNG');
    } else if (type === 'IDAT') idat.push(data);
    else if (type === 'IEND') break;
    p += 12 + len;
  }
  const bpp = colour === 6 ? 4 : 3;
  const raw = inflateSync(Buffer.concat(idat));
  const rgb = new Uint8Array(w * h * 3);
  const stride = w * bpp;
  const prev = new Uint8Array(stride);
  const line = new Uint8Array(stride);
  let at = 0;
  for (let y = 0; y < h; y++) {
    const filter = raw[at++];
    line.set(raw.subarray(at, at + stride)); at += stride;
    // RFC 2083 §6: undo the per-scanline filter.
    for (let i = 0; i < stride; i++) {
      const a = i >= bpp ? line[i - bpp] : 0, b = prev[i];
      const c = i >= bpp ? prev[i - bpp] : 0;
      let v = line[i];
      if (filter === 1) v += a;
      else if (filter === 2) v += b;
      else if (filter === 3) v += (a + b) >> 1;
      else if (filter === 4) {
        const pa = Math.abs(b - c), pb = Math.abs(a - c), pc = Math.abs(a + b - 2 * c);
        v += pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
      }
      line[i] = v & 0xff;
    }
    for (let x = 0; x < w; x++) {
      const s = x * bpp, d = (y * w + x) * 3;
      rgb[d] = line[s]; rgb[d + 1] = line[s + 1]; rgb[d + 2] = line[s + 2];
    }
    prev.set(line);
  }
  return { w, h, rgb };
}

// --------------------------------------------------------------- palette ----

/** The 255 most common colours across every frame, plus one transparent slot. */
function palette(frames) {
  const count = new Map();
  for (const f of frames) {
    for (let i = 0; i < f.rgb.length; i += 3) {
      const k = (f.rgb[i] << 16) | (f.rgb[i + 1] << 8) | f.rgb[i + 2];
      count.set(k, (count.get(k) || 0) + 1);
    }
  }
  const top = [...count.entries()].sort((a, b) => b[1] - a[1]).slice(0, 255);
  const total = [...count.values()].reduce((a, b) => a + b, 0);
  const covered = top.reduce((a, [, n]) => a + n, 0);
  return {
    colours: top.map(([k]) => k),
    distinct: count.size,
    coverage: covered / total,
  };
}

/** Maps a colour to its palette index, nearest by squared distance, memoised. */
function mapper(colours) {
  const exact = new Map();
  colours.forEach((c, i) => exact.set(c, i));
  const cache = new Map();
  return (k) => {
    let i = exact.get(k);
    if (i !== undefined) return i;
    i = cache.get(k);
    if (i !== undefined) return i;
    const r = k >> 16, g = (k >> 8) & 255, b = k & 255;
    let best = 0, bd = Infinity;
    for (let j = 0; j < colours.length; j++) {
      const c = colours[j];
      const dr = r - (c >> 16), dg = g - ((c >> 8) & 255), db = b - (c & 255);
      const d = dr * dr + dg * dg + db * db;
      if (d < bd) { bd = d; best = j; }
    }
    cache.set(k, best);
    return best;
  };
}

// ------------------------------------------------------------------ LZW -----

function lzw(indices, minCodeSize) {
  const clear = 1 << minCodeSize, eoi = clear + 1;
  let codeSize = minCodeSize + 1, next = eoi + 1;
  let dict = new Map();
  const out = [];
  let cur = 0, bits = 0;
  const emit = (code) => {
    cur |= code << bits; bits += codeSize;
    while (bits >= 8) { out.push(cur & 0xff); cur >>= 8; bits -= 8; }
  };
  emit(clear);
  let prefix = indices[0];
  for (let i = 1; i < indices.length; i++) {
    const k = indices[i];
    const key = prefix * 4096 + k;
    const found = dict.get(key);
    if (found !== undefined) { prefix = found; continue; }
    emit(prefix);
    if (next < 4096) {
      dict.set(key, next++);
      // The code width grows when the dictionary outgrows it, and the whole
      // dictionary is dropped at 4096 — the decoder does the same, which is what
      // makes the stream self-describing.
      if (next - 1 === (1 << codeSize) && codeSize < 12) codeSize++;
    } else {
      emit(clear);
      dict = new Map(); next = eoi + 1; codeSize = minCodeSize + 1;
    }
    prefix = k;
  }
  emit(prefix);
  emit(eoi);
  if (bits > 0) out.push(cur & 0xff);
  return out;
}

// ------------------------------------------------------------------ GIF out -

function gif(frames, delays, pal, w, h) {
  const bytes = [];
  const push = (...b) => bytes.push(...b);
  const u16 = (v) => push(v & 0xff, (v >> 8) & 0xff);
  const str = (s) => push(...[...s].map((c) => c.charCodeAt(0)));

  str('GIF89a');
  u16(w); u16(h);
  // Global colour table of 256 entries, no sort, no background colour that
  // matters (every frame paints its own pixels).
  push(0xf7, 0, 0);
  for (let i = 0; i < 256; i++) {
    const c = pal.colours[i];
    if (c === undefined) push(0, 0, 0);
    else push(c >> 16, (c >> 8) & 255, c & 255);
  }
  // Netscape looping extension: the one thing every viewer agrees on.
  push(0x21, 0xff, 0x0b); str('NETSCAPE2.0'); push(3, 1, 0, 0, 0);

  const TRANS = 255; // the slot left free by taking only 255 real colours
  for (let f = 0; f < frames.length; f++) {
    const { x0, y0, fw, fh, idx, transparent } = frames[f];
    push(0x21, 0xf9, 4);
    // Disposal 1 = leave the frame in place, which is what makes a transparent
    // pixel mean "unchanged" rather than "background".
    push(transparent ? 0x05 : 0x04);
    u16(Math.max(2, Math.round(delays[f] / 10)));
    push(transparent ? TRANS : 0, 0);
    push(0x2c); u16(x0); u16(y0); u16(fw); u16(fh); push(0);
    const min = 8;
    push(min);
    const data = lzw(idx, min);
    for (let i = 0; i < data.length; i += 255) {
      const chunk = data.slice(i, i + 255);
      push(chunk.length, ...chunk);
    }
    push(0);
  }
  push(0x3b);
  return Buffer.from(bytes);
}

// -------------------------------------------------------------------- main --

const dir = process.argv[2];
const out = process.argv[3];
const fpsArg = process.argv.indexOf('--fps');
const fps = fpsArg > 0 ? Number(process.argv[fpsArg + 1]) : 10;
if (!dir || !out) {
  console.error('usage: node tools/make-gif.mjs <framedir> <out.gif> [--fps N]');
  process.exit(2);
}

const files = readdirSync(dir).filter((f) => f.endsWith('.png')).sort();
if (!files.length) { console.error(`no PNGs in ${dir}`); process.exit(2); }
const decoded = files.map((f) => decodePng(readFileSync(join(dir, f))));
const { w, h } = decoded[0];
for (const d of decoded) {
  if (d.w !== w || d.h !== h) throw new Error('frames differ in size');
}

const pal = palette(decoded);
const toIndex = mapper(pal.colours);

// Every frame as palette indices.
const indexed = decoded.map((d) => {
  const a = new Uint8Array(w * h);
  for (let i = 0, p = 0; i < d.rgb.length; i += 3, p++) {
    a[p] = toIndex((d.rgb[i] << 16) | (d.rgb[i + 1] << 8) | d.rgb[i + 2]);
  }
  return a;
});

// Merge runs of identical frames into one with a longer delay.
const merged = [];
const delays = [];
for (let i = 0; i < indexed.length; i++) {
  const same = merged.length
    && indexed[i].every((v, k) => v === indexed[merged[merged.length - 1]][k]);
  if (same) delays[delays.length - 1] += 1000 / fps;
  else { merged.push(i); delays.push(1000 / fps); }
}

// Each frame as the bounding box of what changed, unchanged pixels transparent.
const TRANS = 255;
const out_frames = [];
let prev = null;
for (const i of merged) {
  const cur = indexed[i];
  let x0 = 0, y0 = 0, fw = w, fh = h, transparent = false;
  if (prev) {
    let minx = w, miny = h, maxx = -1, maxy = -1;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        if (cur[y * w + x] !== prev[y * w + x]) {
          if (x < minx) minx = x; if (x > maxx) maxx = x;
          if (y < miny) miny = y; if (y > maxy) maxy = y;
        }
      }
    }
    if (maxx < 0) continue; // nothing changed at all
    x0 = minx; y0 = miny; fw = maxx - minx + 1; fh = maxy - miny + 1;
    transparent = true;
  }
  const idx = new Uint8Array(fw * fh);
  for (let y = 0; y < fh; y++) {
    for (let x = 0; x < fw; x++) {
      const s = (y0 + y) * w + x0 + x;
      idx[y * fw + x] = transparent && cur[s] === prev[s] ? TRANS : cur[s];
    }
  }
  out_frames.push({ x0, y0, fw, fh, idx, transparent });
  prev = cur;
}

const buf = gif(out_frames, delays, pal, w, h);
writeFileSync(out, buf);
console.log(
  `${out}: ${w}×${h}, ${files.length} captured → ${out_frames.length} stored, `
  + `${(buf.length / 1024 / 1024).toFixed(2)} MiB\n`
  + `  palette: 255 of ${pal.distinct} distinct colours, `
  + `covering ${(pal.coverage * 100).toFixed(2)} % of pixels exactly`,
);
