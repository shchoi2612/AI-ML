import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "var(--background)",
        foreground: "var(--foreground)",
        "dos-green": "#33ff66",
        "dos-green-dim": "#1a6633",
        "dos-green-bright": "#99ffbb",
        "dos-amber": "#ffaa00",
        "dos-red": "#ff4444",
        "dos-bg": "#0d0d0d",
        "dos-blue": "#6699ff",
        "dos-purple": "#cc66ff",
      },
      fontFamily: {
        mono: ["var(--font-geist-mono)", "Courier New", "monospace"],
      },
    },
  },
  plugins: [],
};
export default config;
