import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

export default defineConfig({
  plugins: [react()],
  base: "/dashboard/",
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    port: 5173,
    proxy: {
      "/dashboards": "http://localhost:8000",
      "/admin": "http://localhost:8000",
      "/query": "http://localhost:8000",
    },
  },
});
