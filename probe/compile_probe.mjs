// Throwaway: D3D compile time of the viewer's full shader (all groups shown) per scene.
import http from "node:http"; import fs from "node:fs"; import path from "node:path"; import { createRequire } from "node:module";
const { chromium } = createRequire(import.meta.url)("playwright");
const [siteDir, backend, ...keys] = process.argv.slice(2);
const types = { ".html": "text/html", ".js": "text/javascript", ".json": "application/json" };
const server = http.createServer((req, res) => { const f = path.resolve(siteDir, decodeURIComponent(new URL(req.url, "http://x").pathname).replace(/^\/+/, "") || "index.html");
  fs.readFile(f, (e, d) => { if (e) { res.writeHead(404); res.end(); return; } res.writeHead(200, { "Content-Type": types[path.extname(f)] || "application/octet-stream" }); res.end(d); }); });
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const args = [`--use-angle=${backend}`, "--use-gl=angle", "--ignore-gpu-blocklist", "--disable-software-rasterizer", "--disable-gpu-sandbox"];
const browser = await chromium.launch({ args, channel: "chromium", headless: true, timeout: 60000 });
for (const key of keys) {
  const v = await browser.newPage({ viewport: { width: 160, height: 120 } });
  const t0 = Date.now();
  try {
    await v.goto(`http://127.0.0.1:${server.address().port}/index.html?scene=${key}`, { timeout: 120000 });
    await v.waitForFunction(() => /compiled|error|could not/i.test(document.querySelector("#status").textContent), null, { timeout: 20 * 60 * 1000, polling: 1000 });
    const first = (await v.textContent("#status")).split("\n")[0];
    const t1 = Date.now();
    await v.evaluate(() => { state.vis = state.vis.map(() => true); useScene(state.scene, true); });
    const all = (await v.textContent("#status")).split("\n")[0];
    console.log(`PROBE ${key}: load ${((t1 - t0) / 1000).toFixed(0)} s (${first}); all groups: ${((Date.now() - t1) / 1000).toFixed(0)} s (${all})`);
  } catch (e) { console.log(`PROBE ${key}: FAIL ${e.message.split("\n")[0]}`); }
  await v.close();
}
await browser.close(); server.close();
