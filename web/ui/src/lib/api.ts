/**
 * API client for the RAG Studio engine.
 *
 * Every number in this app comes from these calls. Nothing here parses a PDF,
 * ranks a passage or scores an answer - that all happens in Python and arrives
 * finished. The UI's job is to show it clearly.
 */

export const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "/api";

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    cache: "no-store",
  });
  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      detail = body?.detail ?? detail;
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(detail, res.status);
  }
  return res.json() as Promise<T>;
}

export const fetcher = <T,>(path: string) => request<T>(path);

export const api = {
  status: () => request<AppStatus>("/status"),
  health: () => request<HealthReport>("/health"),
  documents: () => request<{ documents: DocumentFile[] }>("/documents"),
};

// ---------------------------------------------------------------------------
// Types. They mirror the engine's output field for field (rag_studio/health.py).
// ---------------------------------------------------------------------------

/** One status vocabulary for the whole app (docs/product-language.md). */
export type Status = "healthy" | "running" | "warning" | "needs_attention" | "unavailable" | "info";

export interface AppStatus {
  ai: {
    up: boolean;
    backend: "local" | "remote";
    answer_model: string | null;
    answer_model_installed: boolean | null;
    busy_with: string[];
    error: string | null;
  };
  documents: number;
  indexed: boolean;
  version: string;
}

export interface HealthCheck {
  id: string;
  status: Status;
  label: string;
  message: string;
  fix: string | null;
  technical: Record<string, string | number>;
}

export interface HealthSection {
  id: string;
  name: string;
  checks: HealthCheck[];
}

export interface InstalledModel {
  name: string;
  size: number;
  family: string | null;
  parameter_size: string | null;
  quantization: string | null;
  context_length: number | null;
  capabilities: string[];
  modified_at: string | null;
}

export interface ResidentModel {
  name: string;
  size: number;
  size_vram: number;
  expires_at: string | null;
  context_length: number | null;
}

export interface RoleRow {
  role: string;
  label: string;
  purpose: string;
  model: string | null;
  runs_on: "ollama" | "python";
  arrives: string | null;
  installed: boolean | null;
  loaded: boolean | null;
}

export interface DocumentFile {
  doc_id: string;
  file: string;
  bytes: number;
}

export interface HealthReport {
  checked_at: string;
  overall: Status;
  counts: Record<Status, number>;
  sections: HealthSection[];
  ollama: {
    up: boolean;
    host: string;
    version: string | null;
    installed: InstalledModel[];
    resident: ResidentModel[];
    error: string | null;
  };
  roles: RoleRow[];
  documents: DocumentFile[];
  app: { version: string; python: string; platform: string; ollama_url: string };
}
