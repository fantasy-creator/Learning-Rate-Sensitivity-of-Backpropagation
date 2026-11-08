// Record the demo's guided tour as a 1920x1080 MP4: headless Chrome renders every frame of the tour
// (docs/app.js, window.__tour.render(t) in ?record=1 mode) and ffmpeg encodes the frames.
// Usage: node src/record_tour.mjs [--url http://127.0.0.1:8765/?record=1] [--out tour.mp4] [--fps 30]
//        node src/record_tour.mjs --at 12,40 --out preview   (only those seconds, as preview_12.png, ...)
// Needs Node 22+ (built-in fetch and WebSocket), Chrome (CHROME, else the default path) and ffmpeg
// (FFMPEG, else on PATH). Serve docs/ first, e.g. python -m http.server 8765 --directory docs.
import { spawn } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

const opt = {};
process.argv.slice(2).forEach((v, i, a) => { if (v.startsWith("--")) opt[v.slice(2)] = a[i + 1]; });
const PAGE = opt.url || "http://127.0.0.1:8765/?record=1";
const OUT = path.resolve(opt.out || "tour.mp4");
const FPS = Number(opt.fps || 30);
const W = 1440, H = 810, DPR = 4 / 3; // CSS viewport 1440x810 -> 1920x1080 frames
const CHROME = process.env.CHROME ||
  (process.platform === "win32" ? "C:/Program Files/Google/Chrome/Application/chrome.exe" : "google-chrome");
const FFMPEG = process.env.FFMPEG || "ffmpeg";
const PORT = 9300 + Math.floor(Math.random() * 600);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
async function closeChrome() { // wait for Chrome to exit before deleting its profile; leftovers are harmless
  chrome.kill();
  await Promise.race([new Promise((r) => chrome.once("exit", r)), sleep(5000)]);
  try { fs.rmSync(profile, { recursive: true, force: true, maxRetries: 5, retryDelay: 300 }); } catch {}
}

const profile = fs.mkdtempSync(path.join(os.tmpdir(), "tour-chrome-"));
const chrome = spawn(CHROME, ["--headless=new", `--remote-debugging-port=${PORT}`, `--user-data-dir=${profile}`,
  "--hide-scrollbars", "--mute-audio", "--no-first-run", "--no-default-browser-check", "--no-sandbox",
  `--window-size=${W},${H}`, "about:blank"], { stdio: "ignore" });

async function getJSON(url) {
  for (let k = 0; k < 100; k++) {
    try { return await (await fetch(url)).json(); } catch { await sleep(200); }
  }
  throw new Error("Chrome did not start: " + CHROME);
}
const target = (await getJSON(`http://127.0.0.1:${PORT}/json/list`)).find((t) => t.type === "page");
const ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; });
let seq = 0;
const pending = new Map();
ws.onmessage = (ev) => {
  const msg = JSON.parse(ev.data);
  const p = msg.id && pending.get(msg.id);
  if (!p) return;
  pending.delete(msg.id);
  if (msg.error) p.rej(new Error(msg.error.message)); else p.res(msg.result);
};
const send = (method, params = {}) => new Promise((res, rej) => {
  const id = ++seq;
  pending.set(id, { res, rej });
  ws.send(JSON.stringify({ id, method, params }));
});
const evaluate = async (expression) =>
  (await send("Runtime.evaluate", { expression, returnByValue: true, awaitPromise: true })).result.value;

await send("Page.enable");
await send("Emulation.setDeviceMetricsOverride", { width: W, height: H, deviceScaleFactor: DPR, mobile: false });
await send("Page.navigate", { url: PAGE });
for (let k = 0; ; k++) { // page, images, fonts and the tour hook ready
  const ready = await evaluate("document.readyState === 'complete' && !!window.__tour && " +
    "[...document.images].every((i) => i.complete) && document.fonts.status === 'loaded'");
  if (ready) break;
  if (k > 600) throw new Error("the page did not become ready: " + PAGE);
  await sleep(100);
}
if (opt.at) { // preview: a few frames as PNG files, no video
  for (const t of opt.at.split(",").map(Number)) {
    await evaluate(`window.__tour.render(${t})`);
    const shot = await send("Page.captureScreenshot", { format: "png" });
    fs.writeFileSync(`${OUT.replace(/\.mp4$/, "")}_${t}.png`, Buffer.from(shot.data, "base64"));
  }
  ws.close();
  await closeChrome();
  console.log("wrote previews for", opt.at);
  process.exit(0);
}
const duration = await evaluate("window.__tour.duration");
const frames = Math.round(duration * FPS);
const dir = fs.mkdtempSync(path.join(os.tmpdir(), "tour-frames-"));
console.log(`recording ${frames} frames (${duration} s at ${FPS} fps) from ${PAGE}`);
for (let f = 0; f < frames; f++) {
  await evaluate(`window.__tour.render(${(f / FPS).toFixed(5)})`);
  const shot = await send("Page.captureScreenshot", { format: "png" });
  fs.writeFileSync(path.join(dir, `f${String(f).padStart(5, "0")}.png`), Buffer.from(shot.data, "base64"));
  if (f % 300 === 0) console.log(`  frame ${f}/${frames}`);
}
ws.close();
await closeChrome();

await new Promise((res, rej) => {
  const p = spawn(FFMPEG, ["-y", "-loglevel", "error", "-framerate", String(FPS), "-i", path.join(dir, "f%05d.png"),
    "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
    "-threads", process.env.FFMPEG_THREADS || "0", OUT], // FFMPEG_THREADS limits the encoder's CPU use
  { stdio: "inherit" });
  p.on("exit", (code) => (code === 0 ? res() : rej(new Error("ffmpeg exited with " + code))));
});
fs.rmSync(dir, { recursive: true, force: true });
console.log("wrote", OUT);
