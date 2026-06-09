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
        // 상황실 팔레트
        "sr-bg": "var(--sr-bg)",
        "sr-bg2": "var(--sr-bg2)",
        "sr-panel": "var(--sr-panel)",
        "sr-panel2": "var(--sr-panel2)",
        "sr-border": "var(--sr-border)",
        "sr-ink": "var(--sr-ink)",
        "sr-mut": "var(--sr-mut)",
        "sr-dim": "var(--sr-dim)",
        crisis: "var(--crisis)",
        gold: "var(--gold)",
        stable: "var(--stable)",
        // 레거시 호환
        "dos-green": "var(--stable)",
        "dos-green-dim": "var(--sr-dim)",
        "dos-amber": "var(--gold)",
        "dos-red": "var(--crisis)",
        "dos-bg": "var(--sr-bg)",
        "dos-blue": "#5b9bd5",
        "dos-purple": "#cc66ff",
      },
      fontFamily: {
        mono: ["var(--font-geist-mono)", "Courier New", "monospace"],
        display: ["var(--font-display)"],
      },
    },
  },
  plugins: [],
};
export default config;
