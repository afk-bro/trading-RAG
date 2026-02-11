import { getActiveWorkspaceId } from "@/context/workspace";
import { supabase } from "@/lib/supabase";

const BASE_URL =
  import.meta.env.VITE_API_BASE_URL ?? window.location.origin;

async function getAuthToken(): Promise<string | null> {
  // Prefer Supabase session token
  const { data } = await supabase.auth.getSession();
  if (data.session?.access_token) {
    return data.session.access_token;
  }
  // Fallback: legacy admin token
  return localStorage.getItem("admin_token");
}

function handle401() {
  // Redirect to login unless already there
  if (!window.location.pathname.startsWith("/dashboard/login")) {
    window.location.href = "/dashboard/login";
  }
}

async function request<T>(
  method: "GET" | "POST" | "PATCH",
  path: string,
  body?: unknown,
  params?: Record<string, string | number | boolean>,
  signal?: AbortSignal,
): Promise<T> {
  const url = new URL(path, BASE_URL);
  if (params) {
    for (const [k, v] of Object.entries(params)) {
      if (v !== undefined && v !== null && v !== "") {
        url.searchParams.set(k, String(v));
      }
    }
  }

  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };

  const token = await getAuthToken();
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  const wsId = getActiveWorkspaceId();
  if (wsId) {
    headers["X-Workspace-Id"] = wsId;
  }

  const res = await fetch(url.toString(), {
    method,
    headers,
    body: body ? JSON.stringify(body) : undefined,
    signal,
  });

  if (res.status === 401) {
    handle401();
    throw new Error("401: Unauthorized");
  }

  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`${res.status}: ${text}`);
  }

  return res.json() as Promise<T>;
}

export function apiGet<T>(
  path: string,
  params?: Record<string, string | number | boolean>,
  signal?: AbortSignal,
): Promise<T> {
  return request<T>("GET", path, undefined, params, signal);
}

export function apiPost<T>(path: string, body: unknown, signal?: AbortSignal): Promise<T> {
  return request<T>("POST", path, body, undefined, signal);
}

export function apiPatch<T>(path: string, body: unknown, signal?: AbortSignal): Promise<T> {
  return request<T>("PATCH", path, body, undefined, signal);
}

export async function downloadFile(path: string, filename: string) {
  const url = new URL(path, BASE_URL);
  const headers: Record<string, string> = {};
  const token = await getAuthToken();
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  const res = await fetch(url.toString(), { headers });

  if (res.status === 401) {
    handle401();
    throw new Error("401: Unauthorized");
  }

  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`${res.status}: ${text}`);
  }

  const blob = await res.blob();
  const blobUrl = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = blobUrl;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(blobUrl);
}
