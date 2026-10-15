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
  startRun: (body: {
    sets: string[];
    mode: RunMode;
    model?: string | null;
    k?: number;
    label?: string | null;
    profile?: string;
    prompt?: string;
  }) => request<{ id: string }>("/eval/runs", { method: "POST", body: JSON.stringify(body) }),
  cancelRun: (id: string) =>
    request<{ id: string; cancelling: boolean }>(`/eval/runs/${id}/cancel`, { method: "POST" }),

  /**
   * One question, answered as server-sent events. `fetch` + a stream reader
   * rather than `EventSource`, which can only GET: the question needs a body.
   * `signal` aborts it - a new question or leaving the page must stop the
   * server generating for nobody (it closes the Ollama connection).
   */
  askStream: async (
    body: { question: string; k?: number; profile?: string; prompt?: string },
    onEvent: (event: AskEvent) => void,
    signal?: AbortSignal,
  ): Promise<void> => {
    const res = await fetch(`${API_BASE}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      cache: "no-store",
      signal,
    });
    if (!res.ok || !res.body) {
      let detail = `Request failed (${res.status})`;
      try {
        const err = await res.json();
        detail = typeof err?.detail === "string" ? err.detail : detail;
      } catch {
        /* non-JSON error body */
      }
      throw new ApiError(detail, res.status);
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      // Frames end with a blank line. A partial frame waits for the rest; a
      // `: ping` heartbeat has no event name and is skipped.
      let split = buffer.indexOf("\n\n");
      while (split !== -1) {
        const frame = buffer.slice(0, split);
        buffer = buffer.slice(split + 2);
        const name = /^event:\s*(.+)$/m.exec(frame)?.[1]?.trim();
        const data = /^data:\s*([\s\S]+)$/m.exec(frame)?.[1];
        if (name && data) {
          try {
            onEvent({ event: name, data: JSON.parse(data) } as AskEvent);
          } catch {
            /* an unparseable frame is dropped; the rest of the stream still counts */
          }
        }
        split = buffer.indexOf("\n\n");
      }
    }
  },
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
  index: { profile: string; label: string; rows: number; built_at: string } | null;
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
  digest: string | null;
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
  /** From the PDF's own metadata; null when it has none. */
  title: string | null;
  /** null until an index includes the document. */
  pages: number | null;
  passages: number | null;
  near_empty_pages: number[] | null;
  indexed: boolean;
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

// ---------------------------------------------------------------------------
// Asking. Mirrors rag_studio/answer.py and trace.py.
// ---------------------------------------------------------------------------

export interface TraceStep {
  id: "embed" | "search" | "guard" | "prompt" | "generate" | string;
  label: string;
  ms: number;
  detail: Record<string, string | number | boolean | null>;
}

/** An instruction-like phrase the guard found (rag_studio/guard.py). */
export interface GuardFlag {
  kind: string;
  label: string;
  match: string;
  start: number;
  end: number;
}

export interface Source {
  /** The number the model sees and cites: [n]. */
  n: number;
  rank: number;
  chunk_id: string;
  doc_id: string;
  page: number;
  /** The number printed on the page, when the page prints one (M2 indexes). */
  page_label: string | null;
  section_path: string | null;
  type: "text" | "table" | "figure";
  /** body, a contents page (toc) or a reference list (references). */
  kind: "body" | "toc" | "references";
  /** Cosine similarity. Rank matters more than this number. */
  score: number;
  text: string;
  char_start: number;
  char_end: number;
  /** Instruction-like phrases in the passage. Shown, never hidden. */
  flags: GuardFlag[];
}

export interface IndexInfo {
  profile: string;
  label: string;
  table: string;
  built_at: string;
  rows: number;
  config: Record<string, unknown>;
  embedding: {
    model: string;
    digest: string;
    quantization: string | null;
    dimensions: number;
    doc_prefix: string;
    query_prefix: string;
    backend: string;
    ollama_version: string;
  };
}

export interface AskReady {
  question: string;
  sources: Source[];
  prompt: {
    version: string;
    sha256: string;
    messages: { role: string; content: string }[];
    estimated_tokens: number;
    chars: number;
    num_ctx: number;
    passages: number;
  };
  model: { name: string; backend: "local" | "remote"; host: string; think: boolean; temperature: number; num_ctx: number };
  index: IndexInfo;
}

export interface Usage {
  prompt_tokens: number | null;
  answer_tokens: number | null;
  first_token_ms: number | null;
  load_ms: number | null;
  prompt_ms: number | null;
  answer_ms: number | null;
  done_reason: string | null;
  /** True when the model read far fewer tokens than the prompt holds. */
  truncated: boolean | null;
}

export interface Citations {
  numbers: number[];
  valid: number[];
  invalid: number[];
  map: { n: number; chunk_id: string; doc_id: string; page: number }[];
}

export interface AskDone extends AskReady {
  answer: string;
  refused: boolean;
  citations: Citations;
  usage: Usage;
  trace: TraceStep[];
  total_ms: number;
}

export interface AskError {
  stage: string;
  error: string;
  trace?: TraceStep[];
}

export type AskEvent =
  | { event: "stage"; data: TraceStep }
  | { event: "ready"; data: AskReady }
  | { event: "delta"; data: { text: string } }
  | { event: "done"; data: AskDone }
  | { event: "error"; data: AskError };

export interface PassageBoxes {
  width: number;
  height: number;
  /** In 0..1 page coordinates, one per line of the passage. */
  boxes: { x0: number; y0: number; x1: number; y1: number }[];
  /** Baseline: whether the words lined up. Cleaned indexes: the share of the passage's words found in place (0..1). */
  aligned: boolean | number;
}

/** The query string for a passage's outline on its page. */
export function boxesPath(source: Source, profile: string): string {
  const q = new URLSearchParams({
    start: String(source.char_start),
    end: String(source.char_end),
    profile,
    chunk: source.chunk_id,
  });
  return `/documents/${encodeURIComponent(source.doc_id)}/pages/${source.page}/boxes?${q}`;
}

// ---------------------------------------------------------------------------
// Index profiles and prompts. Mirrors rag_studio/profiles.py and prompt.py.
// ---------------------------------------------------------------------------

export interface IndexProfileRow {
  id: string;
  label: string;
  description: string;
  /** The active build's manifest; null until the profile is built. */
  active: { table: string; built_at: string; rows: number; config: Record<string, unknown>; documents: unknown[] } | null;
  builds: number;
}

export interface PromptVersion {
  version: string;
  sha256: string;
  text: string;
}

// ---------------------------------------------------------------------------
// The page inspector (M2). Mirrors rag_studio/library.py inspect_page().
// ---------------------------------------------------------------------------

/** A cleaning rule's change in the clean text: where, which rule, and the extractor's original. */
export interface TextMark {
  start: number;
  end: number;
  rule: string;
  before: string;
}

/** Where a rule acted in the raw text (furniture, hidden text, a split word...). */
export interface RawMark {
  start: number;
  end: number;
  rule: string;
}

export type PageBox = [number, number, number, number];

export interface HiddenSpan {
  reason: "covered" | "invisible" | "same_colour" | "clipped" | "off_page" | "tiny";
  label: string;
  text: string;
  box: PageBox;
}

export interface ParserView {
  text_raw: string;
  raw_marks: RawMark[];
  text: string;
  marks: TextMark[];
  label: string | null;
  furniture: { text: string; box: PageBox }[];
  hidden: HiddenSpan[];
  headings: { start: number; text: string; size: number }[];
  sections: { start: number; path: string[] }[];
  kinds: { start: number; kind: "body" | "toc" | "references" }[];
  /** Docling only: each layout item with its box and where its text sits in `text`. */
  items: { label: string; box: PageBox; start: number | null; end: number | null; hidden?: boolean }[];
  near_empty: boolean;
  flags: GuardFlag[];
  /** How many times each rule acted on this page. */
  counts: Record<string, number>;
}

export interface CleaningRule {
  id: string;
  defect: number | null;
  label: string;
  explain: string;
}

export interface PageInspection {
  doc_id: string;
  page: number;
  page_count: number;
  width: number;
  height: number;
  label: string | null;
  rules: CleaningRule[];
  pymupdf: ParserView;
  /** null until Docling has converted the document (./rag ingest --profile docling-clean). */
  docling: ParserView | null;
}

export interface CleaningSummary {
  doc_id: string;
  pymupdf: ParserSummary | null;
  docling: ParserSummary | null;
}

export interface ParserSummary {
  counts: Record<string, number>;
  info: Record<string, unknown>;
  /** Pages where some rule acted, with per-rule counts. */
  pages: { page: number; label: string | null; counts: Record<string, number> }[];
}

// ---------------------------------------------------------------------------
// Evaluation. Mirrors rag_studio/eval.py.
// ---------------------------------------------------------------------------

export type Outcome = "PASS" | "REFUSED" | "WRONG";
export type RunMode = "answer" | "retrieval";
export type RunStatus = "running" | "done" | "failed" | "cancelled" | "interrupted";

/** A fact: a substring, any of several, or a pattern with a readable label. */
export type Fact = string | string[] | { pattern: string; label: string };

export interface GoldenQuestion {
  id: string;
  question: string;
  category?: string;
  doc_id?: string | null;
  answer_pages?: number[];
  answers?: { doc_id: string; pages: number[] }[];
  expect: Fact[] | "REFUSE" | null;
  note?: string;
}

export interface GoldenSet {
  id: string;
  name: string;
  description: string;
  source?: string;
  scored: boolean;
  questions: GoldenQuestion[];
}

export interface Target {
  doc_id: string;
  pages: number[];
}

export interface RunHit {
  rank: number;
  chunk_id: string;
  doc_id: string;
  page: number;
  score: number;
}

export interface RunResult {
  set: string;
  id: string;
  category: string | null;
  question: string;
  targets: Target[];
  expect: Fact[] | "REFUSE" | null;
  expect_labels: string[] | "REFUSE" | null;
  hits: RunHit[];
  /** Rank of the first answer page within the top k; null is a Miss. */
  hit_rank: number | null;
  /** Where the first answer page ranked within the top 50; null if not there. */
  best_rank: number | null;
  answer: string | null;
  outcome: Outcome | null;
  found: string[];
  missing: string[];
  refused: boolean | null;
  citations: Citations | null;
  usage: Usage | null;
  error: string | null;
  timings: { embed_ms: number; search_ms: number; generate_ms: number | null };
}

export interface RunSummary {
  recall: Record<string, { hits: number; total: number }>;
  outcomes: Record<Outcome, number> | null;
  scored: number;
  questions: number;
  errors: number;
  answer_seconds: number;
}

export interface RunBrief {
  id: string;
  status: RunStatus;
  started_at: string;
  finished_at: string | null;
  label: string | null;
  ladder: { row: number; label: string } | null;
  spec: { sets: string[]; mode: RunMode; k: number; model: string | null; prompt_version: string; profile: string };
  sets: Record<string, { name: string; scored: boolean }>;
  progress: { done: number; total: number };
  summary: RunSummary | null;
  seconds: number | null;
  error: string | null;
  config: {
    mode: RunMode;
    k: number;
    profile: string;
    index_label: string;
    table: string;
    /** pymupdf or docling; null on runs saved before M2. */
    parser?: string | null;
    cleaner?: string | null;
    embedding_model: string;
    retrieval: string;
    answer_model: string | null;
    prompt_version: string | null;
  };
}

export interface RunConfig {
  mode: RunMode;
  k: number;
  deep_k: number;
  index: IndexInfo;
  retrieval: { method: string; metric: string; exact: boolean };
  answer: {
    model: string;
    think: boolean;
    temperature: number;
    num_ctx: number;
    digest: string | null;
    quantization: string | null;
    parameter_size: string | null;
    prompt_version: string;
    prompt_sha256: string;
  } | null;
  app_version: string;
  ollama_version: string;
}

export interface Run extends Omit<RunBrief, "config"> {
  config: RunConfig;
  results: RunResult[];
}

export interface AnswerModel {
  name: string;
  parameter_size: string | null;
  quantization: string | null;
  note: string | null;
}

// ---------------------------------------------------------------------------
// Comparing two runs (spec §8). Mirrors rag_studio/eval.py compare().
// ---------------------------------------------------------------------------

export interface CompareSide {
  outcome: Outcome | null;
  hit_rank: number | null;
  best_rank: number | null;
  answer: string | null;
  found: string[];
  missing: string[];
  refused: boolean | null;
  error: string | null;
}

export interface CompareRow {
  set: string;
  id: string;
  question: string;
  category: string | null;
  a: CompareSide;
  b: CompareSide;
  outcome_changed: boolean;
  retrieval_changed: boolean;
  rank_moved: boolean;
}

export interface RunComparison {
  a: Pick<RunBrief, "id" | "label" | "ladder" | "started_at" | "status" | "config">;
  b: Pick<RunBrief, "id" | "label" | "ladder" | "started_at" | "status" | "config">;
  /** Setup keys whose values differ (flattened, e.g. "index.config.parser"). */
  config: { key: string; a: unknown; b: unknown }[];
  rows: CompareRow[];
  summary: { a: RunSummary; b: RunSummary };
  /** Questions whose outcome or retrieval changed. */
  changed: number;
}
