const rawUrl = import.meta.env.VITE_BACKEND_URL || "http://localhost:3001";
export const BACKEND_URL = rawUrl.replace(/\/$/, "");
