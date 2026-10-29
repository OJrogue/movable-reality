import { createReadStream, existsSync, statSync } from "node:fs";
import { extname, join, normalize, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig, type Plugin } from "vite";

// Dev only: serve ../data/scenes at /scenes so the viewer reads pipeline output directly.
// For hosting, copy precomputed scenes into web/public/scenes or pass ?base=<url>.
const SCENES_DIR = fileURLToPath(new URL("../data/scenes", import.meta.url));
const PUBLIC_SCENES = fileURLToPath(new URL("./public/scenes", import.meta.url));
const TYPES: Record<string, string> = {
  ".json": "application/json",
  ".ply": "application/octet-stream",
  ".spz": "application/octet-stream",
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

export default defineConfig({
  plugins: [serveLocalScenes()],
  server: { port: 5173 },
  build: { chunkSizeWarningLimit: 4000 }, // Spark bundles its WASM sorter (~3 MB)
});
