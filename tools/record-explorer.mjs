// Drives docs/explorer.html in headless Chromium and captures the frames of the
// tour that becomes assets/explorer.gif.
//
// The interaction is scripted rather than hand-performed, and the pointer is a
// div the script moves — a headless browser has no cursor to film. Everything it
// clicks is a real element and everything that appears is the page reacting to it;
// nothing is staged with a fake screenshot.
//
//   node tools/record-explorer.mjs [outdir]
//
// Frames land as f0000.png … in `outdir` (default /tmp/omni-frames), one per
// 100 ms of the finished animation. Identical consecutive frames are merged by
// tools/make-gif.mjs, so holding still is cheap.
//
// This is the one thing in the repository with a dependency: it needs Playwright
// and a Chromium. Neither is a build dependency — the recording is committed, and
// CI checks the committed GIF rather than making one — so this runs when the
// explorer changes and not otherwise. If Playwright is installed somewhere other
// than the resolver's path, say where:
//
//   PLAYWRIGHT_MODULE=/usr/lib/node_modules/playwright/index.js \
//   CHROMIUM=/path/to/chrome node tools/record-explorer.mjs

import { mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

// ESM has no NODE_PATH, so an install outside the resolver's reach is named by
// path. A directory needs its entry point spelled out; both forms work.
// Named by path it resolves as CommonJS, whose exports arrive under `default`;
// named by package it resolves through the package's own exports map. Accept both
// rather than making the caller know which.
const spec = process.env.PLAYWRIGHT_MODULE || 'playwright';
const mod = await import(spec.endsWith('/') ? spec + 'index.js' : spec);
const chromium = mod.chromium || mod.default?.chromium;
if (!chromium) throw new Error(`no chromium export in ${spec}`);

const OUT = process.argv[2] || '/tmp/omni-frames';
const PAGE = 'file://' + join(process.cwd(), 'docs/explorer.html');
const W = 1160, H = 660;
const FPS = 10;

rmSync(OUT, { recursive: true, force: true });
mkdirSync(OUT, { recursive: true });

const browser = await chromium.launch({
  // Playwright's own browser unless one is named: the point is a deterministic
  // renderer, not a particular install.
  ...(process.env.CHROMIUM ? { executablePath: process.env.CHROMIUM } : {}),
  args: ['--force-color-profile=srgb', '--font-render-hinting=none'],
});
const page = await browser.newPage({ viewport: { width: W, height: H } });
await page.goto(PAGE);
await page.waitForTimeout(300);

let n = 0;
/** Captures one frame. */
async function frame(count = 1) {
  for (let i = 0; i < count; i++) {
    await page.screenshot({
      path: join(OUT, 'f' + String(n).padStart(4, '0') + '.png'),
      animations: 'disabled',
    });
    n++;
  }
}

/** Moves the scripted pointer to an element's centre, in a few steps. */
async function moveTo(selectorOrHandle, steps = 5) {
  const h = typeof selectorOrHandle === 'string'
    ? await page.waitForSelector(selectorOrHandle) : selectorOrHandle;
  if (!h) throw new Error(`nothing to move to: ${selectorOrHandle}`);
  await h.scrollIntoViewIfNeeded().catch(() => {});
  const box = await h.boundingBox();
  if (!box) {
    const html = await h.evaluate((e) => e.outerHTML.slice(0, 120));
    throw new Error(`no box for ${html} — is it hidden?`);
  }
  const to = { x: box.x + Math.min(box.width / 2, 120), y: box.y + box.height / 2 };
  const from = await page.evaluate(() => window.__cur || { x: 500, y: 320 });
  for (let i = 1; i <= steps; i++) {
    const t = i / steps;
    // Ease-out, so the pointer arrives rather than stops dead.
    const e = 1 - Math.pow(1 - t, 2);
    await page.evaluate(
      ([x, y]) => window.__moveCursor(x, y),
      [from.x + (to.x - from.x) * e, from.y + (to.y - from.y) * e],
    );
    await frame();
  }
  return h;
}

/** Moves to an element, pulses the pointer, and really clicks it. */
async function click(selectorOrHandle, hold = 3) {
  const h = await moveTo(selectorOrHandle);
  await page.evaluate(() => window.__downCursor(true));
  await frame();
  await h.click({ force: true });
  await page.evaluate(() => window.__downCursor(false));
  await frame(hold);
  return h;
}

/** Hovers an element so the page's own mouseenter handlers fire. */
async function hover(h, hold = 2) {
  await moveTo(h, 3);
  await h.hover({ force: true });
  await frame(hold);
}

// The scripted pointer, and a caption line the tour writes into.
await page.evaluate(() => {
  const c = document.getElementById('cursor');
  window.__cur = { x: 500, y: 320 };
  window.__moveCursor = (x, y) => {
    window.__cur = { x, y };
    c.style.opacity = 1; c.style.left = x + 'px'; c.style.top = y + 'px';
  };
  window.__downCursor = (d) => c.classList.toggle('down', d);
  const cap = document.createElement('div');
  cap.id = 'caption';
  cap.style.cssText =
    'position:fixed;left:0;right:0;bottom:0;padding:7px 16px;background:#161a24;'
    + 'border-top:1px solid #232833;color:#8fa3c0;font-size:12px;z-index:98;'
    + 'letter-spacing:.01em;';
  document.body.appendChild(cap);
  document.querySelector('main').style.paddingBottom = '44px';
  window.__say = (t) => { cap.textContent = t; };
});
const say = (t) => page.evaluate((t) => window.__say(t), t);

// ---------------------------------------------------------------- the tour --

await say('examples/toy.omni — a real container: two transformer layers, 49 objects');
await frame(14);

// 1. The header: 128 fixed bytes, and what each field says.
await say('It opens at the 128-byte file header. Every field, in the bytes themselves.');
await frame(6);
const hdrRows = await page.$$('#infobody table.kv tr');
for (const name of ['magic', 'hash_algo', 'root_digest', 'file_size']) {
  for (const r of hdrRows) {
    if (((await r.textContent()) || '').trim().startsWith(name)) {
      await say(`header · ${name} — hovering a field highlights the bytes it is`);
      await hover(r, 4);
      break;
    }
  }
}

// 2. The file, to scale. The weights dominate; that is the honest picture.
await say('The bar above is the file to scale. The weights are almost all of it.');
await frame(10);
await click('#map div.k-OBJ', 5);
await say('OBJ — every structure object: 6 KiB that answers every question about a 111 KiB file');
await frame(10);

// 3. Down the graph, from the root.
await say('The graph is built from the refs the objects actually carry.');
await frame(6);
const nodes = await page.$$('.node');
/** The first *visible* node whose label contains `needle`. */
const byText = async (needle) => {
  for (const r of await page.$$('.node')) {
    if (!((await r.textContent()) || '').includes(needle)) continue;
    if (await r.isVisible()) return r;
  }
  throw new Error(`no visible node matching ${JSON.stringify(needle)}`);
};
await click(await byText('Manifest'), 6);
await say('Manifest — the root object. Its CBOR is on the right, its bytes in the middle.');
await frame(8);
await click(await byText('Metadata'), 5);
await say('Metadata — the model card: architecture, parameters, licence.');
await frame(8);
await click(await byText('Table'), 5);
await say('TensorTable — names to descriptors. This is the only place a tensor has a name.');
await frame(8);

// 4. One tensor, all the way to its bytes.
const desc = await byText('L0 attn.q_proj');
await click(desc, 5);
await say('A tensor descriptor: shape [64,64], bf16, and a value that is an expression.');
await frame(12);
await click(await desc.$('.fold'), 3);
await say('Its value is `literal` over a ChunkList — the weights are addressed, not embedded.');
await frame(8);
const chunks = await click(await byText('Chunks'), 5);
await say('ChunkList — the chunks that make up the tensor, each one addressed by its digest.');
await frame(10);
await click(await chunks.$('.fold'), 3);
await click(await byText('Blob'), 5);
await say('And there are the weights: 4096 bytes of bf16, aligned, addressed by their digest.');
await frame(14);

// 5. Where the weights live, and what alignment costs.
await click('#map div.k-BLOB', 4);
await say('BLOB — the weights. Every data object starts on a 4096-byte boundary.');
await frame(12);
await click('#map div.k-PAD', 4);
await say('PAD — what that alignment costs. It must be zero, and the verifier checks it.');
await frame(12);
await click('#map div.k-INDEX', 4);
await say('INDEX — the one structure that is not CBOR: a sorted array, usable straight from an mmap.');
await frame(12);

// 6. The ladder. The lines are the ones `omni verify --level 6` printed.
await say('And then it proves itself: V0 framing through V6 recomputed derived objects.');
await frame(6);
await page.evaluate(() => window.runLadder(260));
await frame(26);
await say('`omni verify toy.omni --level 6` → valid. Nothing here was drawn by hand.');
await frame(22);

writeFileSync(join(OUT, 'meta.json'), JSON.stringify({ frames: n, fps: FPS, w: W, h: H }));
console.log(`captured ${n} frames at ${W}×${H} (${(n / FPS).toFixed(1)}s at ${FPS} fps)`);
await browser.close();
