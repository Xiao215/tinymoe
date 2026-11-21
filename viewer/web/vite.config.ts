import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `npm run dev` proxies the API to a running `tinymoe viewer` (override with VIEWER_API).
const api = process.env.VIEWER_API ?? "http://127.0.0.1:8765";

export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/api": { target: api, changeOrigin: true } } },
  build: { outDir: "dist", emptyOutDir: true, chunkSizeWarningLimit: 1500 },
});
