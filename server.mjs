// server.mjs
import express from "express";
import { listen } from "@pinggy/pinggy";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PORT = process.env.PORT || 3000;
const app = express();

app.use((req, res, next) => {
  res.setHeader("Cross-Origin-Embedder-Policy", "require-corp");
  res.setHeader("Cross-Origin-Opener-Policy", "same-origin");
  next();
});

app.use(express.static(path.join(__dirname, "public")));
app.use("/static/models", express.static(path.join(__dirname, "static/models")));

app.get("/health", (_req, res) => res.json({ status: "ok", port: PORT }));

listen(app, { forwarding: `localhost:${PORT}` }).then((server) => {
  console.log("Local server port:", server.address().port);
  const urls = server.tunnel?.urls();
  if (urls?.length) console.log("Public Pinggy URL:", urls[0]);
}).catch(err => {
  console.error("Server failed:", err);
  process.exit(1);
});
