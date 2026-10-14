import type { NextConfig } from "next";

/**
 * The Python API owns every RAG step and runs on 127.0.0.1:8775. Proxying /api
 * through Next keeps the browser on one origin, so there is no CORS surface.
 */
const API_TARGET = process.env.RAG_STUDIO_API_URL ?? "http://127.0.0.1:8775";

const nextConfig: NextConfig = {
  /**
   * Next's dev server returns 403 for its own /_next chunks when the page's
   * Origin is a host it does not recognise, and it does not treat 127.0.0.1 as
   * localhost. The HTML renders, hydration never happens, and the console says
   * nothing (learned in the trading app). Both spellings are listed.
   */
  allowedDevOrigins: ["127.0.0.1", "localhost", "127.0.0.1:3100", "localhost:3100"],

  async rewrites() {
    return [{ source: "/api/:path*", destination: `${API_TARGET}/api/:path*` }];
  },

  experimental: {
    /**
     * The rewrite proxy kills a socket after 30 s without traffic
     * (`server/lib/router-utils/proxy-request.js`: `proxyTimeout || 30000`,
     * checked in 16.3.0). A CPU answer from qwen3:8b took 141-246 s with no
     * bytes sent while the prompt was read. Streams send a heartbeat comment
     * every 15 s, so this is the bound of last resort, not the real one.
     */
    proxyTimeout: 330_000,
  },
  reactStrictMode: true,
  // The default bottom-left spot covers the sidebar's status rows.
  devIndicators: { position: "bottom-right" },
};

export default nextConfig;
