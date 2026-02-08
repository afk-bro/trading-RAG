import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        background: "#0d1117",
        "bg-secondary": "#161b22",
        "bg-tertiary": "#21262d",
        border: "#30363d",
        "border-subtle": "#21262d",
        foreground: "#c9d1d9",
        "text-muted": "#8b949e",
        "text-emphasis": "#f0f6fc",
        accent: "#58a6ff",
        success: "#3fb950",
        warning: "#d29922",
        danger: "#f85149",
        purple: "#a371f7",
        orange: "#f7a141",
      },
      fontSize: {
        xs: "11px",
        sm: "12px",
        base: "14px",
        lg: "16px",
        xl: "18px",
        "2xl": "24px",
      },
      fontFamily: {
        sans: [
          "-apple-system",
          "BlinkMacSystemFont",
          "Segoe UI",
          "Roboto",
          "sans-serif",
        ],
        mono: [
          "SF Mono",
          "Consolas",
          "Liberation Mono",
          "Menlo",
          "monospace",
        ],
      },
      borderRadius: {
        sm: "4px",
        md: "6px",
        lg: "8px",
      },
      spacing: {
        "1": "4px",
        "2": "8px",
        "3": "12px",
        "4": "16px",
        "5": "20px",
        "6": "24px",
        "8": "32px",
        "10": "40px",
        "12": "48px",
      },
    },
  },
  plugins: [],
};

export default config;
