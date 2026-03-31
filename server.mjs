// server.mjs
import express from "express";
import { listen } from "@pinggy/pinggy";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PORT = process.env.PORT || 3000;

const app = express();

// Strict COOP/COEP to enable SharedArrayBuffer (required for WebGPU)
app.use((req, res, next) => {
  res.setHeader("Cross-Origin-Embedder-Policy", "require-corp");
  res.setHeader("Cross-Origin-Opener-Policy", "same-origin");
  next();
});

// Serve frontend (public) and static models
app.use(express.static(path.join(__dirname, "public")));
app.use("/static/models", express.static(path.join(__dirname, "static/models")));

// Health-check endpoint
app.get("/health", (_req, res) => res.json({ status: "ok", port: PORT }));

// Start Express + Pinggy tunnel; log public URL as soon as it's ready
listen(app, {
  forwarding: `localhost:${PORT}`,
}).then((server) => {
  console.log("Local server port:", server.address().port);
  const urls = server.tunnel?.urls();
  if (urls && urls.length > 0) {
    console.log("Public Pinggy URL:", urls[0]);
  } else {
    console.warn("Pinggy tunnel URLs not found; ensure network/tunnel is up.");
  }
}).catch((err) => {
  console.error("Failed to start server or tunnel:", err);
  process.exit(1);
});
