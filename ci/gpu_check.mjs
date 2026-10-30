// Run the viewer's GPU check (site/diagnostics.html) in headless Chromium on a
// chosen ANGLE graphics backend and fail unless every check matches NumPy.
//
//   node ci/gpu_check.mjs <site dir> <backend> [<renderer substring>]
//   backend: swiftshader (Linux), d3d11-warp (Windows, Direct3D 11 via WARP),
//            metal (macOS), or anything Chromium's --use-angle accepts.
// The optional renderer substring makes the run fail if Chromium silently fell
// back to another backend (so each CI leg really tests what it claims).
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";

const { chromium } = createRequire(import.meta.url)("playwright");

const [siteDir, backend, expectRenderer] = process.argv.slice(2);
if (!siteDir || !backend) {
  console.error("usage: node ci/gpu_check.mjs <site dir> <backend> [<renderer substring>]");
  process.exit(2);
}

const types = { ".html": "text/html", ".js": "text/javascript", ".json": "application/json", ".gz": "application/octet-stream" };
const server = http.createServer((req, res) => {
  const rel = decodeURIComponent(new URL(req.url, "http://x").pathname).replace(/^\/+/, "") || "index.html";
  const root = path.resolve(siteDir);
  const file = path.resolve(root, rel);
  if (!file.startsWith(root + path.sep)) { res.writeHead(403); res.end(); return; }
  fs.readFile(file, (err, data) => {
    if (err) { res.writeHead(404); res.end(); return; }
    res.writeHead(200, { "Content-Type": types[path.extname(file)] || "application/octet-stream" });
    res.end(data);
  });
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const url = `http://127.0.0.1:${server.address().port}/diagnostics.html`;

const args = [`--use-angle=${backend}`, "--use-gl=angle", "--ignore-gpu-blocklist"];
if (backend === "swiftshader") args.push("--enable-unsafe-swiftshader");
else args.push("--disable-software-rasterizer");  // fail rather than fall back to SwiftShader
if (backend.startsWith("d3d")) args.push("--disable-gpu-sandbox");  // D3D in a service session
const COMPILE_LIMIT_MS = 15000;
// Never hang a CI runner: hard stop after 12 minutes whatever happens.
setTimeout(() => { console.log("FAIL: GPU check timed out"); process.exit(1); }, 12 * 60 * 1000).unref();
// channel "chromium": the full browser (Playwright's default headless shell only
// has SwiftShader). GPU_CHECK_HEADED=1 runs it headed, for runners where headless
// Chrome would not use the hardware/D3D backend.
const headless = process.env.GPU_CHECK_HEADED !== "1";
const browser = await chromium.launch({ args, channel: "chromium", headless, timeout: 60000 });
const page = await browser.newPage();
page.on("pageerror", (e) => console.log("pageerror:", e.message));
const t0 = Date.now();  // progress in the CI log, so a timeout shows where it stopped
page.on("console", (m) => { const s = m.text().match(/^diag: \s*(== .*)/); if (s) console.log(`[${Math.round((Date.now() - t0) / 1000)} s] ${s[1]}`); });
let code = 1;
try {
  await page.goto(url, { timeout: 60000 });
  await page.waitForFunction(() => window.DIAG && window.DIAG.done, null, { timeout: 10 * 60 * 1000, polling: 2000 });
  const report = await page.inputValue("#report");
  const diag = await page.evaluate(() => window.DIAG);
  // The viewer's own render shader (picking, highlighting) must compile here too.
  const viewerErrors = [];
  const keys = JSON.parse(fs.readFileSync(path.join(siteDir, "scenes", "index.json"), "utf8"));
  for (const key of keys) {
    const v = await browser.newPage();
    const errs = [];
    v.on("pageerror", (e) => errs.push(e.message));
    await v.goto(url.replace("diagnostics.html", `index.html?scene=${key}`), { timeout: 60000 });
    await v.waitForFunction(() => /compiled|error|could not/i.test(document.querySelector("#status").textContent),
      null, { timeout: 5 * 60 * 1000, polling: 1000 });
    const status = await v.textContent("#status");
    console.log(`[${Math.round((Date.now() - t0) / 1000)} s] viewer ${key} done`);
    console.log(`viewer ${key}: ${status.split("\n")[0]}`);
    if (!/compiled/.test(status) || errs.length) viewerErrors.push(`${key}: ${status} ${errs.join("; ")}`);
    // A browser gives up on a GPU task that stalls for long (Chrome: about 10 s), so a
    // shader that compiles slower than this on a CI machine fails here, not for users.
    const ms = +(status.match(/compiled in (\d+) ms/) || [0, 0])[1];
    if (ms > COMPILE_LIMIT_MS) viewerErrors.push(`${key}: shader compile ${ms} ms > ${COMPILE_LIMIT_MS} ms`);
    await v.close();
  }
  console.log(report);
  if (expectRenderer && !String(diag.gpu || "").toLowerCase().includes(expectRenderer.toLowerCase())) {
    console.log(`\nFAIL: expected a renderer containing "${expectRenderer}", got "${diag.gpu}"`);
  } else if (viewerErrors.length) {
    console.log(`\nFAIL: viewer shader: ${viewerErrors.join(" | ")}`);
  } else if (diag.pass) {
    code = 0;
  }
} catch (e) {
  console.log("FAIL:", e.message);
} finally {
  await browser.close();
  server.close();
}
process.exit(code);
