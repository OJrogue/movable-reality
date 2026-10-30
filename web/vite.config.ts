import {
  createReadStream,
  existsSync,
  mkdirSync,
  readdirSync,
  statSync,
  writeFileSync,
} from "node:fs";
import type { IncomingMessage } from "node:http";
import { extname, join, normalize, relative, sep } from "node:path";
import { fileURLToPath } from "node:url";
import basicSsl from "@vitejs/plugin-basic-ssl";
import { defineConfig, type Plugin } from "vite";
import { logStem } from "./src/perf/logName";

// Dev only: serve ../data/scenes at /scenes so the viewer reads pipeline output directly.
// For hosting, copy precomputed scenes into web/public/scenes or pass ?base=<url>.
const SCENES_DIR = fileURLToPath(new URL("../data/scenes", import.meta.url));
const PUBLIC_SCENES = fileURLToPath(new URL("./public/scenes", import.meta.url));
const REPO_ROOT = fileURLToPath(new URL("..", import.meta.url));
const BENCH_DIR = fileURLToPath(new URL("../benchmarks", import.meta.url));
const TYPES: Record<string, string> = {
  ".json": "application/json",
  ".ply": "application/octet-stream",
  ".spz": "application/octet-stream",
  ".rad": "application/octet-stream",
  ".jpg": "image/jpeg",
  ".png": "image/png",
};

function serveLocalScenes(): Plugin {
  return {
    name: "serve-local-scenes",
    configureServer(server) {
      server.middlewares.use("/scenes", (req, res, next) => {
        const rel = decodeURIComponent((req.url ?? "/").split("?")[0]);
        for (const dir of [SCENES_DIR, PUBLIC_SCENES]) {
          const file = normalize(join(dir, rel));
          if (file.startsWith(dir + sep) && existsSync(file) && statSync(file).isFile()) {
            res.setHeader("Content-Type", TYPES[extname(file)] ?? "application/octet-stream");
            res.setHeader("Cache-Control", "no-store");
            res.setHeader("Accept-Ranges", "bytes");
            // Spark's paged .rad loader reads header and chunks with Range requests.
            const size = statSync(file).size;
            const range = /^bytes=(\d*)-(\d*)$/.exec(req.headers.range ?? "");
            if (range && (range[1] || range[2])) {
              const start = range[1] ? Number(range[1]) : Math.max(0, size - Number(range[2]));
              const end = range[1] && range[2] ? Math.min(Number(range[2]), size - 1) : size - 1;
              if (start > end || start >= size) {
                res.statusCode = 416;
                res.setHeader("Content-Range", `bytes */${size}`);
                res.end();
                return;
              }
              res.statusCode = 206;
              res.setHeader("Content-Range", `bytes ${start}-${end}/${size}`);
              res.setHeader("Content-Length", String(end - start + 1));
              createReadStream(file, { start, end }).pipe(res);
              return;
            }
            res.setHeader("Content-Length", String(size));
            createReadStream(file).pipe(res);
            return;
          }
        }
        if (req.method === "GET") {
          res.statusCode = 404;
          res.end("not found");
          return;
        }
        next();
      });
    },
  };
}

const MAX_LOG_BYTES = 16 * 1024 * 1024; // a 10 min log at 90 Hz is about 5 MB

function readBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let size = 0;
    req.on("data", (c: Buffer) => {
      size += c.length;
      if (size > MAX_LOG_BYTES) {
        reject(new Error("log too large"));
        req.destroy();
      } else chunks.push(c);
    });
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
    req.on("error", reject);
  });
}

interface PostedLog {
  kind?: unknown;
  created?: unknown;
  scene?: unknown;
  mode?: unknown;
  budget?: unknown;
  headless?: unknown;
  device?: { label?: unknown };
}

const jsonFiles = (dir: string) =>
  existsSync(dir) ? readdirSync(dir).filter((f) => f.endsWith(".json")).sort() : [];

/**
 * Dev only: POST /__bench with a frame log (web/src/perf/frameLog.ts) writes
 * benchmarks/raw/<timestamp>_<device>_<scene>_<mode>.json. Logs from headless browsers go
 * to benchmarks/headless/ instead, so they are never mistaken for device numbers.
 * GET /__bench lists the log files, for the run menu (?menu=r0).
 */
function benchEndpoint(): Plugin {
  return {
    name: "bench-endpoint",
    configureServer(server) {
      server.middlewares.use("/__bench", async (req, res) => {
        const reply = (status: number, body: object) => {
          res.statusCode = status;
          res.setHeader("Content-Type", "application/json");
          res.end(JSON.stringify(body));
        };
        if (req.method === "GET") {
          return reply(200, {
            raw: jsonFiles(join(BENCH_DIR, "raw")),
            headless: jsonFiles(join(BENCH_DIR, "headless")),
          });
        }
        if (req.method !== "POST") return reply(405, { error: "POST a frame log" });
        try {
          const text = await readBody(req);
          const log = JSON.parse(text) as PostedLog;
          if (log.kind !== "mr-frame-log") return reply(400, { error: "not a frame log" });
          const stem = logStem({
            created: String(log.created ?? new Date().toISOString()),
            device: { label: String(log.device?.label ?? "unknown") },
            scene: String(log.scene ?? "unknown"),
            mode: String(log.mode ?? "unknown"),
            budget: log.budget === true,
          });
          const dir = join(BENCH_DIR, log.headless === true ? "headless" : "raw");
          mkdirSync(dir, { recursive: true });
          let file = join(dir, `${stem}.json`);
          for (let i = 2; existsSync(file); i++) file = join(dir, `${stem}_${i}.json`);
          writeFileSync(file, text);
          const path = relative(REPO_ROOT, file).split(sep).join("/");
          server.config.logger.info(`[bench] wrote ${path} (${(text.length / 1024).toFixed(0)} KB)`);
          reply(200, { path });
        } catch (e) {
          reply(400, { error: String(e) });
        }
      });
    },
  };
}

// Dev only: /r0 serves the R0 run menu (src/bench/menu.ts; Vite compiles the module).
const MENU_HTML = `<!doctype html>
<html lang="en"><head><meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>R0 runs</title><style>html,body{margin:0;background:#111}</style></head>
<body><script type="module" src="/src/bench/menu.ts"></script></body></html>`;

function runMenu(): Plugin {
  return {
    name: "run-menu",
    configureServer(server) {
      server.middlewares.use("/r0", (req, res, next) => {
        if ((req.url ?? "/").split("?")[0] !== "/") return next();
        res.setHeader("Content-Type", "text/html; charset=utf-8");
        res.setHeader("Cache-Control", "no-store");
        res.end(MENU_HTML);
      });
    },
  };
}

// HTTPS for headsets on the LAN (WebXR needs a secure context): MR_HTTPS=1 npm run dev.
// @vitejs/plugin-basic-ssl makes a self-signed certificate; the headset browser asks once
// to accept it. Over USB, `adb reverse` + http://localhost needs no certificate.
const https = process.env.MR_HTTPS === "1";

export default defineConfig({
  plugins: [serveLocalScenes(), benchEndpoint(), runMenu(), ...(https ? [basicSsl()] : [])],
  server: {
    port: 5173,
    host: https ? true : undefined,
    // Core lane: the scene server (uv run python -m movable_reality.server, port 8000)
    // holds the agent and the edit API; the viewer works without it (local edits only).
    proxy: {
      "/api": "http://127.0.0.1:8000",
      "/ws": { target: "ws://127.0.0.1:8000", ws: true },
    },
  },
  build: { chunkSizeWarningLimit: 4000 }, // Spark bundles its WASM sorter (~3 MB)
});
