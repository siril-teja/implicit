// Frame time of a scene with some groups shown and hidden: the viewer's own readout
// ("sharp in N ms": the full-resolution image), median of several redraws, in headless
// Chromium (SwiftShader by default: a CPU renderer, so compare ratios, not times).
//
//   node ci/frame_time.mjs <site dir> <scene key> "<group>" ["<group>" ...]
//
// Prints the median with all groups shown, with the named groups hidden, and the ratio.
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";

const { chromium } = createRequire(import.meta.url)("playwright");
const [siteDir, key, ...groups] = process.argv.slice(2);
if (!siteDir || !key || !groups.length) {
  console.error('usage: node ci/frame_time.mjs <site dir> <scene key> "<group>" ...');
  process.exit(2);
}
const types = { ".html": "text/html", ".js": "text/javascript", ".json": "application/json" };
const server = http.createServer((req, res) => {
  const rel = decodeURIComponent(new URL(req.url, "http://x").pathname).replace(/^\/+/, "") || "index.html";
  const root = path.resolve(siteDir), file = path.resolve(root, rel);
  if (!file.startsWith(root + path.sep)) { res.writeHead(403); res.end(); return; }
  fs.readFile(file, (err, data) => {
    if (err) { res.writeHead(404); res.end(); return; }
    res.writeHead(200, { "Content-Type": types[path.extname(file)] || "application/octet-stream" });
    res.end(data);
  });
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
setTimeout(() => { console.log("frame_time: timed out"); process.exit(1); }, 20 * 60 * 1000).unref();
const browser = await chromium.launch({ args: ["--use-angle=swiftshader", "--use-gl=angle", "--enable-unsafe-swiftshader"],
  channel: "chromium", headless: true });
const page = await browser.newPage({ viewport: { width: 800, height: 600 } });
await page.goto(`http://127.0.0.1:${server.address().port}/index.html?scene=${key}`);
const compiled = () => page.waitForFunction(() => /compiled/i.test(document.querySelector("#status").textContent),
  null, { timeout: 10 * 60 * 1000, polling: 500 });

async function sharp(n = 5) {
  const times = [];
  for (let i = 0; i < n; i++) {
    await page.evaluate(() => { document.querySelector("#fps").textContent = ""; window.dispatchEvent(new Event("resize")); });
    await page.waitForFunction(() => /sharp in/.test(document.querySelector("#fps").textContent), null,
      { timeout: 5 * 60 * 1000, polling: 100 });
    times.push(+(await page.textContent("#fps")).match(/sharp in (\d+) ms/)[1]);
  }
  times.sort((a, b) => a - b);
  return times[Math.floor(n / 2)];
}

await compiled();
const shown = await sharp();
for (const g of groups) {
  const box = page.locator("#bodies label.body", { hasText: g }).locator("input");
  if (await box.count() !== 1) { console.log(`frame_time: no group "${g}"`); process.exit(1); }
  await box.uncheck();
}
await page.waitForTimeout(500);
await compiled();
const hidden = await sharp();
console.log(`frame_time ${key}: ${shown} ms with all groups, ${hidden} ms without ${groups.join(", ")}: `
  + `x${(shown / hidden).toFixed(2)}`);
await browser.close();
server.close();
