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
    search?: SearchMethod;
    section_cap?: number | null;
    rerank?: string;
    rewrite?: string;
    agent_rounds?: number;
    as_typed?: boolean;
  }) => request<{ id: string }>("/eval/runs", { method: "POST", body: JSON.stringify(body) }),
  /** Measure approximate indexes against exact search on one build (5-60 s). */
  annCheck: (profile: string) =>
    request<AnnCheck>(`/learn/store/ann-check?profile=${encodeURIComponent(profile)}`, { method: "POST" }),
  /** Steps 10 and 12: an edited answer read by the same citation, refusal and scoring code. Changes nothing. */
  answerCheck: (body: { run: string; question: string; text: string }) =>
    request<AnswerCheck>("/learn/answer-check", { method: "POST", body: JSON.stringify(body) }),
  /** Step 13: the crafted page answered now by the Answer role's model (10-60 s). */
  safetyLive: (version: "v2" | "v3") =>
    request<SafetyLive>(`/learn/safety/live?version=${version}`, { method: "POST" }),
  // Conversations (M5)
  chats: () => request<{ chats: ChatBrief[] }>("/chats"),
  chat: (id: string) => request<ChatFull>(`/chats/${encodeURIComponent(id)}`),
  deleteChat: (id: string) => request<{ id: string; deleted: boolean }>(`/chats/${encodeURIComponent(id)}`, { method: "DELETE" }),
  cancelRun: (id: string) =>
    request<{ id: string; cancelling: boolean }>(`/eval/runs/${id}/cancel`, { method: "POST" }),

  // Indexes (M4)
  indexes: () => request<IndexesResponse>("/indexes"),
  buildIndex: (profile: string, on: "local" | "colab", force = false) =>
    request<{ profile: string; job: BuildJob | null }>(`/indexes/${encodeURIComponent(profile)}/build`, {
      method: "POST", body: JSON.stringify({ on, force }),
    }),
  cancelBuild: (profile: string) =>
    request<{ profile: string; cancelling: boolean }>(`/indexes/${encodeURIComponent(profile)}/cancel`, { method: "POST" }),
  /** Context lines (M5): write the missing ones on Remote AI in the background, or stop. */
  writeContext: (profile: string) =>
    request<{ profile: string; job: ContextJobFile | null }>(`/indexes/${encodeURIComponent(profile)}/context`, { method: "POST" }),
  cancelContext: (profile: string) =>
    request<{ profile: string; cancelling: boolean }>(`/indexes/${encodeURIComponent(profile)}/context/cancel`, { method: "POST" }),
  /** The vector match check: up to two minutes on this CPU. */
  matchCheck: (profile: string) =>
    request<MatchCheck>(`/indexes/${encodeURIComponent(profile)}/check`, { method: "POST" }),

  // Remote AI (M4)
  llmStatus: (probe = false) => request<LlmSnapshot>(`/llm/status${probe ? "?probe=true" : ""}`),
  /** The setup cell. `models` is space-separated Ollama tags; `worker` the GPU worker's jobs, comma-separated. */
  llmColab: (job?: { models?: string; worker?: string }) =>
    request<ColabCell>(`/llm/colab${job ? `?${new URLSearchParams(Object.entries(job).filter(([, v]) => v) as [string, string][])}` : ""}`),
  llmConnect: (body: { code?: string; url?: string; token?: string }) =>
    request<ConnectResult>("/llm/connect", { method: "POST", body: JSON.stringify(body) }),
  llmTest: () => request<SmokeTest & LlmSnapshot>("/llm/test", { method: "POST" }),
  llmDisconnect: (forgetToken = false) =>
    request<{ ok: boolean } & LlmSnapshot>("/llm/disconnect", {
      method: "POST", body: JSON.stringify({ forget_token: forgetToken }),
    }),

  /**
   * One question, answered as server-sent events. `fetch` + a stream reader
   * rather than `EventSource`, which can only GET: the question needs a body.
   * `signal` aborts it - a new question or leaving the page must stop the
   * server generating for nobody (it closes the Ollama connection).
   */
  askStream: async (
    body: {
      question: string;
      k?: number;
      profile?: string;
      prompt?: string;
      search?: SearchMethod;
      section_cap?: number | null;
      rerank?: string;
      rewrite?: string;
      agent_rounds?: number;
      chat_id?: string;
      as_typed?: boolean;
    },
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
  /** The Colab runtime as saved at connect time; no request crosses the tunnel. */
  remote?: { connected: boolean; host: string | null; last_error: string | null };
  ai: {
    up: boolean;
    backend: "local" | "remote";
    answer_model: string | null;
    answer_model_installed: boolean | null;
    busy_with: string[];
    error: string | null;
  };
  documents: number;
  default_profile: string;
  build_command: string;
  /** What the first question loads in this app, and what is free now (GB). */
  memory: { needed_gb: number | null; free_gb: number | null; short: boolean };
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
  /** local: this computer (Ollama or this app); remote: the Colab runtime only, with no local fallback. */
  where: "local" | "remote";
  /** Remote roles: whether a runtime is connected. */
  connected?: boolean;
  arrives: string | null;
  /** Ollama: pulled; this app: downloaded at its pinned revision; Remote AI: served by the runtime. null = unknown. */
  installed: boolean | null;
  loaded: boolean | null;
}

/** One technique default, read from the engine (health.technique_defaults). */
export interface TechniqueDefault {
  key: string;
  label: string;
  value: string;
  id?: string;
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
  /** What Ask and a run use unless told otherwise. */
  defaults?: TechniqueDefault[];
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
  /** Cosine similarity (dense), BM25 score, or the RRF score (hybrid). Rank matters more than this number. */
  score: number;
  text: string;
  char_start: number;
  char_end: number;
  /** The section label put in front of the passage ("Document: ...\nSection: ..."); null on indexes without one. */
  label?: string | null;
  /** Where the passage ranked in each list before fusion or the section limit. */
  ranks?: SearchRanks;
  /** The reranker's score for it (higher is better); null when it was not reranked. */
  rerank_score?: number | null;
  /** The line that places the passage in its document (contextual retrieval). Embedded with the passage, never shown to the answer model. */
  context?: string | null;
  /** Moved behind the others by "Limit per section". */
  capped?: boolean;
  /** Instruction-like phrases in the passage. Shown, never hidden. */
  flags: GuardFlag[];
}

export type SearchMethod = "dense" | "bm25" | "hybrid";

/** A passage's rank in each list it was searched in; null when it was not in that list. */
export type SearchRanks = Partial<Record<string, number | null>>;

/** How a question was searched, as the engine records it (retrieve.py `SearchSpec.describe()`). */
export interface SearchDescribe {
  method: SearchMethod;
  metric?: string;
  exact?: boolean;
  fusion?: string;
  rrf_k?: number;
  depth?: number;
  section_cap?: number;
  word_index?: { name: string; column: string; settings: Record<string, unknown> } | null;
  /** A reranker id (or, in a recorded run, its full description) and how many passages it reads. */
  rerank?: string | Record<string, unknown>;
  rerank_depth?: number;
  /** decompose, multi or hyde (or, in a recorded run, its full description). */
  rewrite?: string | Record<string, unknown>;
  agent_rounds?: number;
  agent?: Record<string, unknown>;
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
  /** The question as typed. */
  question: string;
  /** What was searched: a follow-up made whole by Remote AI (the same text when it was not rewritten). */
  searched_for?: string;
  standalone_question?: string | null;
  /** true: rewritten; false: asked as typed on purpose; null: not a follow-up. */
  rewritten?: boolean | null;
  chat_id?: string;
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
  /** Missing on answers made before M3: those searched by meaning only. */
  search?: SearchDescribe;
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
  /** Set when the answer could not be saved to its conversation (the answer itself is fine). */
  chat_warning?: string;
}

export interface AskError {
  stage: string;
  error: string;
  trace?: TraceStep[];
  chat_id?: string;
  /** A follow-up that Remote AI could not rewrite can be sent as typed. */
  can_ask_as_typed?: boolean;
}

/** A saved conversation (rag_studio/chat.py). Sources are references, never copies of passages. */
export interface ChatTurn {
  question: string;
  standalone_question: string | null;
  rewritten: boolean | null;
  answer: string;
  refused: boolean;
  citations: { n: number; chunk_id: string; doc_id: string; page: number }[];
  sources: { n: number; chunk_id: string; doc_id: string; page: number }[];
  setup: Record<string, unknown>;
  trace: { id: string; ms: number }[];
  total_ms: number | null;
  at: string;
}

export interface ChatBrief {
  id: string;
  title: string | null;
  turns: number;
  created_at: string | null;
  updated_at: string | null;
}

export interface ChatFull extends Omit<ChatBrief, "turns"> {
  turns: ChatTurn[];
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

/** Where a profile embeds: this computer, the Colab runtime, or either. */
export type Where = "local" | "colab" | "either";

/** A yes/no the engine decided, with the reason (and fix) when it is no. */
export interface Can {
  ok: boolean;
  reason: string | null;
  fix?: string | null;
}

export interface BuildManifest {
  table: string;
  built_at: string;
  rows: number;
  config: Record<string, unknown>;
  documents: unknown[];
  embedding: {
    model: string;
    digest: string;
    dimensions: number;
    built_on?: "local" | "colab";
    backend?: string;
    ollama_version?: string;
    gpu?: string | null;
    quantization?: string | null;
    revision?: string;
    [key: string]: unknown;
  };
  embedded?: number;
  reused?: number;
  from_cache?: number;
  timings?: Record<string, number>;
}

/** A build's progress record (rag_studio/ingest.py `_Job`). */
export interface BuildJob {
  profile: string;
  label: string;
  on: "local" | "colab";
  status: "running" | "done" | "failed" | "cancelled" | "interrupted";
  stage: string;
  started_at: string;
  finished_at?: string;
  done: number;
  total: number | null;
  passages?: number;
  reused?: number;
  cached?: number;
  error: string | null;
  up_to_date?: boolean;
  rows?: number;
  embedded?: number;
  seconds?: number;
}

export interface MatchPair {
  a: "local" | "colab";
  b: "local" | "colab" | "stored";
  what: "passages" | "questions";
  /** noise: one machine against itself in another order; index: against the stored vectors; cross: the spec's check. */
  kind: "noise" | "index" | "cross";
  n: number;
  min: number;
  median: number;
  mean: number;
}

/** The vector match check (rag_studio/embed.py `match_check`). */
export interface MatchCheck {
  profile: string;
  table: string;
  model: string;
  built_on: "local" | "colab";
  passages: number;
  questions: number;
  threshold: number;
  sides: Record<"local" | "colab", { available: boolean; reason?: string; seconds?: number; ollama_version?: string; gpu?: string | null }>;
  pairs: MatchPair[];
  cross_machine: boolean;
  /** null with one machine: nothing to compare across. */
  passed: boolean | null;
  noise_floor: number | null;
  noise_passages: number;
  checked_at: string;
}

export interface IndexProfileRow {
  id: string;
  label: string;
  description: string;
  parser: string;
  /** fixed, structure or semantic. */
  chunker: string;
  /** Whether its passages carry section labels (prompt v4 expects them). */
  labels: boolean;
  /** Whether a model-written context line is embedded with each passage (M5, ladder row 7). */
  contextual?: boolean;
  /** Whether the build has a word index, so BM25 and hybrid search can run. */
  word_index: boolean;
  /** The active build's manifest; null until the profile is built. */
  active: BuildManifest | null;
  builds: number;
  history?: { table: string; built_at: string; rows: number; built_on: "local" | "colab"; seconds: number | null }[];
  embedding?: {
    model: string;
    name: string;
    dimensions: number;
    backend: "ollama" | "sentence-transformers";
    build_on: Where;
    query_on: "local" | "colab";
    doc_prefix: string;
    query_prefix: string;
    options: Record<string, unknown> | null;
    revision: string | null;
  };
  can?: { build_local: Can; build_colab: Can; query: Can };
  build_ram_gb?: number | null;
  job?: BuildJob | null;
  check?: MatchCheck | null;
}

export interface SearchMethodInfo {
  id: SearchMethod;
  label: string;
  explain: string;
}

/** Rerankers (rag_studio/rerank.py `availability`): where each runs and whether it can run now. */
export interface Reranker {
  id: string;
  label: string;
  where: "local" | "colab";
  model: string;
  can: Can;
}

export interface RerankersResponse {
  depth: number;
  rerankers: Reranker[];
  /** Whether Remote AI can rewrite questions now (connected, serving the rewrite model), and why not. */
  rewrite?: Can;
}

export interface IndexesResponse {
  profiles: IndexProfileRow[];
  search_methods: SearchMethodInfo[];
  /** The index the app uses unless told otherwise (rag_studio/profiles.py DEFAULT). */
  default_profile: string;
  remote?: { connected: boolean; host: string | null; models: string[] | null; connected_at: string | null };
  free_ram_gb?: number | null;
  active_build?: BuildJob | null;
}

/** How it works, step 5 (rag_studio/learn.py `embed_lab`). */
export interface EmbedHit {
  rank: number;
  id: string;
  doc: string;
  page: number;
  score: number;
  xy: [number, number];
  text: string;
}

export interface EmbedLab {
  profile: string;
  label: string;
  model: string;
  model_name: string;
  dimensions: number;
  doc_prefix: string;
  query_prefix: string;
  embedded_on: "local" | "remote";
  rows: number;
  /** Share of the vectors' variance the two PCA axes keep (0-1). */
  explained: number;
  documents: { doc_id: string; title: string }[];
  points: { id: string; doc: string; page: number; xy: [number, number] }[];
  question: {
    text: string;
    sent: string;
    embed_ms: number;
    xy: [number, number];
    nearest: EmbedHit[];
    matryoshka: { dimensions: number; ids: string[]; kept: number; bytes: number }[];
  } | null;
  mix?:
    | { profile: string; model_name: string; possible: true; nearest: EmbedHit[]; kept: number }
    | { profile: string; model_name: string; possible: false; reason: string };
}

// ---------------------------------------------------------------------------
// Remote AI (M4). Mirrors rag_studio/remote.py. No field ever holds a token:
// a credential travels browser -> API, never back.
// ---------------------------------------------------------------------------

export interface RemoteConnection {
  connected: boolean;
  backend: "local" | "remote";
  token_remembered: boolean;
  state_path: string;
  code_prefix: string;
  needed_models: string[];
  fallback: "strict";
  url?: string;
  host?: string;
  authenticated?: boolean;
  models?: string[];
  gateway?: string | null;
  connected_at?: string | null;
  last_ok_at?: string | null;
  last_error?: string | null;
  last_error_at?: string | null;
  last_error_kind?: string | null;
}

export interface RemoteProbe {
  reachable: boolean;
  error: string | null;
  error_kind: string | null;
  latency_ms?: number;
  ollama_version?: string;
  models?: { name: string; digest: string | null; size: number; quantization: string | null; capabilities: string[] }[];
  loaded?: { name: string; size: number; size_vram: number; on_gpu: boolean; fully_on_gpu: boolean }[];
  gpu?: { available: boolean; reason?: string; gpus?: { name: string; memory_total_mb: number; memory_used_mb: number; utilization_pct: number }[] };
  missing?: string[];
  /** The GPU worker (tools/colab/worker.py): the rerankers it serves, the one loaded, and whether it writes context lines. */
  worker?: { loaded: string | null; served: string[]; context?: boolean; vram?: { peak_mb?: number | null } | null } | null;
}

export interface LlmSnapshot {
  connection: RemoteConnection;
  remote: RemoteProbe | null;
}

export interface SmokeTest {
  ok: boolean;
  stage: string;
  error: string | null;
  error_kind: string | null;
  seconds: number | null;
  embedded: { model: string; digest: string | null; dimensions: number; seconds: number }[];
  /** Chat models that answered a one-token chat. */
  chatted?: { model: string; seconds: number }[];
  worker?: { loaded: string | null; served: string[] } | null;
}

export interface ConnectResult extends SmokeTest {
  missing: string[];
  remote: RemoteProbe | null;
  connection: RemoteConnection;
}

export interface ColabCell {
  cell: string;
  models: string[];
  /** The GPU worker's jobs: rerankers by id, and "context" for context lines. */
  worker?: string[];
  ollama_version: string | null;
  files: string[];
  tokenless: boolean;
  cli_equivalent: string;
  kilobytes: number;
  lines: number;
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
  /** Set E: its follow-ups are rewritten by Remote AI unless asked as typed. */
  remote_only?: boolean;
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
  /** M3 runs: the passage's rank in each list before fusion or the section limit. */
  ranks?: SearchRanks;
  capped?: boolean;
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
  /** embed_ms is null for a BM25-only search (no question vector). */
  timings: { embed_ms: number | null; search_ms: number; generate_ms: number | null; rerank_ms?: number };
  // M5 (rag_studio/eval.py `_search_one`): each is null when its switch was off.
  /** Where the first answer page ranked before and after the reranker, and whether its scores were remembered. */
  rerank?: { before_rank: number | null; after_rank: number | null; cache: "hit" | "miss" | null; compute_ms: number | null } | null;
  /** Each Remote AI output this question used, in order: the follow-up rewrite, the rewrite, each round's decision. */
  rewrites?: RewriteRecord[] | null;
  /** Multi-step search, round by round (round 1 is the normal search). */
  rounds?: RoundRecord[] | null;
  /** How the multi-step search ended. */
  agent?: { stop: string; stop_text: string; tokens: number; seconds: number; rounds_used: number } | null;
  /** What crossed the tunnel, per remote step (followup, rewrite, rerank). */
  network?: Record<string, NetRecord> | null;
  /** Set E: the follow-up as Remote AI made it whole; null when asked as typed. */
  standalone_question?: string | null;
  /** "hit" (remembered, no call) or "miss" (asked) per remote output. */
  cache?: Record<string, string> | null;
  /** Set E: the recorded earlier turn(s). */
  history?: { question: string; answer: string }[];
  /** Set E: the rewrite held every expected word and no forbidden one; null when not rewritten. */
  rewrite_ok?: boolean | null;
}

export interface RewriteRecord {
  technique: "standalone" | "decompose" | "multi" | "hyde" | "agent" | string;
  value: {
    standalone?: string;
    questions?: string[];
    queries?: string[];
    passage?: string;
    action?: string;
    query?: string | null;
    doc?: string | null;
    reason?: string;
  };
  cache: string;
  seq: number | null;
  ms: number | null;
  tokens: { prompt?: number | null; answer?: number | null };
  network: NetRecord | null;
}

export interface RoundRecord {
  round: number;
  /** search, answer (enough evidence) or stop. */
  action: string;
  query?: string | null;
  doc?: string | null;
  reason?: string | null;
  cache?: string | null;
  /** New passages this round put in the top 10. */
  new?: number;
  found?: number;
}

export interface NetRecord {
  calls?: number;
  bytes_out_raw?: number;
  bytes_out_wire?: number;
  bytes_in_raw?: number;
  bytes_in_wire?: number;
  wall_ms?: number;
  server_ms?: number;
  connections?: number;
}

/** MRR and recall@20 from the deep search (eval.py `ranking_stats`); null when no question counted. */
export interface RankingStats {
  n: number;
  mrr: number | null;
  recall20: number | null;
}

export interface RunSummary {
  recall: Record<string, { hits: number; total: number }>;
  outcomes: Record<Outcome, number> | null;
  scored: number;
  questions: number;
  errors: number;
  answer_seconds: number;
  /** Every run has it: older ones are filled in from their recorded ranks. */
  ranking?: RankingStats & { single: RankingStats; multi: RankingStats };
  /** Follow-ups rewritten right, of those rewritten; null when none were. */
  rewrite?: { ok: number; total: number } | null;
  /** Set E's chain: rewritten right, then found, then answered right; null when no follow-up was asked. */
  followups?: { n: number; rewritten: number; rewrite_ok: number; hits: number; scored: number; passes: number } | null;
}

export interface RunBrief {
  id: string;
  status: RunStatus;
  started_at: string;
  finished_at: string | null;
  label: string | null;
  ladder: { row: number; label: string } | null;
  spec: {
    sets: string[];
    mode: RunMode;
    k: number;
    model: string | null;
    prompt_version: string;
    profile: string;
    /** Missing on runs made before M3 (all dense, no limit). M5 keys appear only when on. */
    search?: {
      method: SearchMethod;
      section_cap: number | null;
      rerank?: string;
      rerank_depth?: number;
      rewrite?: string;
      agent_rounds?: number;
    };
    as_typed?: boolean;
  };
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
    /** dense, bm25 or hybrid. */
    retrieval: string;
    /** "Limit per section"; null or missing when off. */
    section_cap?: number | null;
    /** M5 (eval.py `_brief`): each null when off. `rerank` is "qwen3-4b (top 20)"; `rerank_name` is the picker's name. */
    rerank?: string | null;
    rerank_name?: string | null;
    rerank_depth?: number | null;
    rewrite?: string | null;
    agent_rounds?: number | null;
    /** Set E's follow-ups: "rewritten" or "as typed"; null when the run asked none. */
    followup?: "rewritten" | "as typed" | null;
    /** The index embeds a context line with each passage. */
    contextual?: boolean;
    answer_model: string | null;
    prompt_version: string | null;
  };
}

export interface RunConfig {
  mode: RunMode;
  k: number;
  deep_k: number;
  index: IndexInfo;
  retrieval: SearchDescribe;
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
  /** Set E: how follow-ups were handled (the standalone prompt's description, or as typed). */
  followup?: { as_typed?: boolean; prompt_version?: string; prompt_sha256?: string; model?: string } | null;
  /** Facts that vary between runs of one setup (where the reranker ran, the runtime's GPU). Never grouped on. */
  runtime?: Record<string, Record<string, unknown>> | null;
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

// ---------------------------------------------------------------------------
// How it works (M3). Mirrors rag_studio/learn.py.
// ---------------------------------------------------------------------------

export type ChunkMethod = "fixed" | "recursive" | "sentence" | "structure" | "semantic";

export interface ChunkLab {
  doc_id: string;
  page: number;
  page_count: number;
  parser: "pymupdf" | "docling";
  method: ChunkMethod;
  size: number;
  overlap: number;
  percentile: number;
  /** The page's clean text; every span below indexes it. */
  text: string;
  near_empty: boolean;
  ms: number;
  /** The embedding model semantic chunking compared sentences with; null otherwise. */
  model: string | null;
  passages: { start: number; end: number; chars: number; starts_mid_word: boolean }[];
  /** Fixed only: the M1 chunker's spans, which can start mid-word. */
  baseline: { start: number; end: number; starts_mid_word: boolean }[] | null;
  /** Structure-aware only: each passage's section and kind, and the section label an index puts in front of it. */
  sections:
    | { start: number; end: number; path: string[]; kind: "body" | "toc" | "references"; role: string; label: string }[]
    | null;
  /** Semantic only: the similarity of each pair of neighbouring sentences, and whether a passage ends there. */
  curve: { after: number; next: number; similarity: number; cut: boolean }[] | null;
  /** Semantic only: the similarity below which a passage ends. */
  threshold: number | null;
  sentences: { start: number; end: number }[] | null;
  headings: { start: number; text: string; size: number }[];
  page_sections: { start: number; path: string[] }[];
  kinds: { start: number; kind: string }[];
}

export interface LabHit {
  rank: number;
  chunk_id: string;
  doc_id: string;
  page: number;
  page_label: string | null;
  section_path: string | null;
  kind: "body" | "toc" | "references";
  type: string | null;
  score: number;
  text: string;
  char_start: number;
  char_end: number;
  /** On one of the test question's answer pages. */
  answer_page: boolean;
  ranks: SearchRanks | null;
  capped: boolean;
}

export interface LabList {
  hits: LabHit[];
  /** Where the first answer page ranked, searching 50 deep; null when not found or not a test question. */
  first_answer_rank: number | null;
}

export interface SearchLab {
  question: string;
  profile: string;
  label: string;
  rows: number;
  k: number;
  section_cap: number | null;
  /** The test question with this exact wording, e.g. "B4"; null otherwise. */
  test_question: string | null;
  targets: Target[];
  word_index: boolean;
  /** `reranked` appears only when the lab was asked to rerank (it is slow). */
  lists: Record<SearchMethod, LabList | null> & { reranked?: RerankedList };
  timings: Record<string, number>;
  rrf_k: number;
}

/** The dense top 20 re-read with the question by a reranker (learn.search_lab with `rerank`). */
export interface RerankedList {
  hits: (LabHit & { rank_before: number | null; rerank_score: number | null })[];
  first_answer_rank: number | null;
  reranker: Record<string, unknown>;
  label: string;
  ms: number;
}

export interface AnnResult {
  index: string;
  label: string;
  /** Share of exact search's top 10 that the index also returned. */
  recall: number;
  median_ms: number;
  build_s: number;
}

export interface AnnCheck {
  table: string;
  queries: number;
  k: number;
  results: AnnResult[];
  measured_at: string;
  lancedb_version: string;
}

export interface StoreFacts {
  profile: string;
  label: string;
  table: string;
  rows: number;
  dimensions: number;
  bytes: number;
  columns: { name: string; type: string }[];
  sample: { chunk_id: string; text: string; label: string | null; vector_head: number[] };
  versions: { version: number; at: string; rows: number }[];
  indexes: { name: string; type: string; columns: string[]; rows: number | null; bytes: number | null }[];
  word_index: { name: string; column: string; settings: Record<string, unknown>; table_version?: number } | null;
  builds: { table: string; built_at: string; rows: number; active: boolean }[];
  ann: AnnCheck | null;
}

/** How it works, step 4 (rag_studio/learn.py `context_lab`). Read from the context-line cache: no model call. */
export interface ContextJob {
  /** running, done, failed, cancelled; "interrupted" when the job's process is gone. */
  status: string | null;
  done: number | null;
  total: number | null;
  started_at: string | null;
  finished_at: string | null;
}

export interface ContextPassage {
  chunk_id: string;
  doc_id: string;
  page: number;
  section_path: string | null;
  kind: string;
  label: string | null;
  text: string;
  /** null: no context line written for this passage yet. */
  context: string | null;
  /** The index's document prefix, which the embedding model reads first. */
  doc_prefix: string;
  /** Exactly what is embedded: prefix, section label, then the passage. */
  embedded_before: string;
  /** The same with the context line between the label and the passage; null without a line. */
  embedded_after: string | null;
  /** The part of the document the model read to write the line, shared by every passage in it. */
  window: { passages: number; position: number; chars: number; est_tokens: number; head: string };
  /** This passage's rank for the test question on each built index (null: not in the top 50). */
  ranks?: Partial<Record<"plain" | "ctx", number | null>>;
}

export interface ContextRankSide {
  profile: string;
  /** Where the first answer page ranked, searching 50 deep; null when not in the top 50. */
  first_answer_rank: number | null;
  top: { rank: number; chunk_id: string; doc_id: string; page: number; answer_page: boolean }[];
}

export interface ContextLab {
  profile: string;
  label: string;
  plain_profile: string;
  plain_label: string | null;
  coverage: {
    passages: number;
    written: number;
    /** Windows: section-contiguous blocks, each read once per passage in it. */
    blocks: number;
    digest: string | null;
    built: boolean;
    plain_built: boolean;
    rows: number | null;
    /** Whether data/context/<profile>.jsonl exists (the lines' backup). */
    export: boolean;
    job: ContextJob | null;
  };
  profiles: { id: string; label: string; parser: string; built: boolean; job: ContextJob | null }[];
  describe: { model: string; prompt_version: string; prompt_sha256: string; window_policy: string; num_ctx: number };
  prompt: { system: string; suffix: string };
  passage: ContextPassage | null;
  samples: { chunk_id: string; doc_id: string; page: number; section_path: string | null; context: string; text: string }[];
  /** null without a test question; a side is null when that index is not built. */
  ranking: {
    id: string;
    question: string;
    targets: Target[];
    plain: ContextRankSide | null;
    ctx: ContextRankSide | null;
  } | null;
  timings: Record<string, number>;
}

/** How it works, step 8 (rag_studio/learn.py `rewrite_lab`): one technique as a recorded run used it. */
export interface RewriteLabTechnique {
  /** asked, decompose, multi, hyde, agent; for a follow-up: followup, as_typed. */
  technique: string;
  /** null: no finished run on this index has used it for this question yet. */
  run_id: string | null;
  label?: string | null;
  ladder?: { row: number; label: string } | null;
  mode?: "answer" | "retrieval";
  search?: Record<string, unknown>;
  result?: RunResult;
}

export interface RewriteLab {
  question_id: string;
  question: string;
  history: { question: string; answer: string }[] | null;
  targets: Target[];
  profile: string;
  label: string;
  techniques: RewriteLabTechnique[];
}

/** Step 8, live (`rewrite_live`): one question searched now with one technique. */
export interface RewriteLive {
  question: string;
  test_question: string | null;
  targets: Target[];
  profile: string;
  technique: string;
  steps: TraceStep[];
  hits: LabHit[];
  first_answer_rank: number | null;
  cache: Record<string, string> | null;
  network: Record<string, NetRecord> | null;
  ms: number;
}

/** A context-line job (rag_studio/context.py `write_lines`), as its job file records it. */
export interface ContextJobFile {
  profile: string;
  status: "running" | "done" | "failed" | "cancelled" | "interrupted";
  stage: string;
  done: number;
  total: number | null;
  error: string | null;
  started_at: string;
  finished_at?: string;
  summary?: { lines: number; seconds: number; lines_per_s: number | null; timeouts?: number; rejected?: number } | null;
}

/** GET /api/indexes/{id}/context. */
export interface ContextStatus {
  profile: string;
  passages: number;
  written: number;
  blocks: number;
  digest: string | null;
  job: ContextJobFile | null;
  can: Can;
  model: string;
}

// ---------------------------------------------------------------------------
// How it works (M6): steps 9, 10, 12 and 13. Mirrors rag_studio/learn.py.
// ---------------------------------------------------------------------------

export type RunHead = Pick<RunBrief, "id" | "label" | "ladder" | "started_at" | "status" | "config">;

/** A run that answered one test question (`recorded_answers`), for the pickers. */
export interface RecordedChoice {
  run_id: string;
  label: string | null;
  ladder: { row: number; label: string } | null;
  started_at: string;
  index_label: string | null;
  answer_model: string | null;
  prompt_version: string | null;
  rerank: string | null;
  rewrite: string | null;
  agent_rounds: number | null;
  /** The app's defaults: default index, prompt and answer model, every switch off. */
  default_setup: boolean;
  outcome: Outcome | null;
  hit_rank: number | null;
}

export interface RecordedAnswers {
  question_id: string;
  set: string;
  question: string;
  history: { question: string; answer: string }[] | null;
  category: string | null;
  runs: RecordedChoice[];
}

/** A passage as the recorded answer's model was shown it, read back from the run's own index build. */
export interface RecordedSource {
  n: number;
  rank: number | null;
  chunk_id: string;
  doc_id: string;
  page: number;
  page_label: string | null;
  section_path: string | null;
  label: string | null;
  kind: string | null;
  type: string | null;
  text: string;
  flags: GuardFlag[];
  /** The recorded answer cited it. */
  cited: boolean;
}

/** The same question from the same passages, with one thing changed (prompt version or answer model). */
export interface SamePassages {
  value: string;
  run_id: string;
  label: string | null;
  ladder: { row: number; label: string } | null;
  this_run: boolean;
  outcome: Outcome | null;
  answer: string;
  refused: boolean | null;
  found: string[] | null;
  missing: string[] | null;
  generate_ms: number | null;
  answer_tokens: number | null;
}

export interface GenerationInfo {
  model: string | null;
  think: boolean | null;
  temperature: number | null;
  num_ctx: number | null;
  quantization: string | null;
  parameter_size: string | null;
  prompt_version: string | null;
}

interface RecordedCommon {
  run: RunHead;
  question_id: string;
  set: string;
  question: string;
  searched_for: string;
  history: { question: string; answer: string }[] | null;
  category: string | null;
  answer: string;
  outcome: Outcome | null;
  refused: boolean | null;
  expect_labels: string[] | "REFUSE" | null;
  trap: boolean;
  sources: RecordedSource[];
  model: GenerationInfo;
}

export interface PromptPart {
  part: "system" | "passage" | "question" | "frame";
  n?: number;
  chars: number;
  /** Ollama's own count split by characters ("measured"), or 4 characters per token ("estimated"). */
  tokens: number;
}

export interface PromptVersionView {
  version: string;
  sha256: string;
  system: string;
  /** The user message this version builds from the same passages. */
  user: string;
  against: string | null;
  /** Lines of `system` that the previous version does not have. */
  changed: number[];
  fenced: boolean;
  current: boolean;
  measured: SamePassages | null;
}

/** Step 9 (`prompt_lab`). */
export interface PromptLab extends RecordedCommon {
  version: string;
  prompt_sha256: string;
  /** identical: proven by the hash the run recorded; rebuilt: the run predates that hash; differs: something changed. */
  exact: "identical" | "rebuilt" | "differs";
  messages: { role: string; content: string }[];
  parts: PromptPart[];
  budget: {
    num_ctx: number | null;
    chars: number;
    estimated_tokens: number;
    prompt_tokens: number | null;
    answer_tokens: number | null;
    chars_per_token: number | null;
    truncation_chars_per_token: number;
    truncated: boolean | null;
    split: "measured" | "estimated";
  };
  markers: { fence_open: string; fence_close: string; reminder: string };
  versions: PromptVersionView[];
}

/** One `[n]` group in an answer (answer.py `citation_marks`). */
export interface CitationMark {
  start: number;
  end: number;
  numbers: number[];
  valid: number[];
  invalid: number[];
}

export interface FactCheck {
  label: string;
  found: boolean;
  spans: [number, number][];
}

/** Step 10 (`cite_lab`). */
export interface CiteLab extends RecordedCommon {
  citations: Citations;
  marks: CitationMark[];
  matches_recorded: boolean;
  usage: Usage | null;
  timings: RunResult["timings"] | null;
  found: string[] | null;
  missing: string[] | null;
  facts: FactCheck[];
  refusal: string;
  /** The answer key changed since this run scored it. */
  key_changed: boolean;
  /** The same passages answered by each answer model that was measured on this setup. */
  models: SamePassages[];
}

/** An edited answer, read by the engine (`answer_check`). */
export interface AnswerCheck {
  citations: Citations;
  marks: CitationMark[];
  k: number;
  refused: boolean;
  score: { outcome: Outcome | null; found: string[]; missing: string[]; refused: boolean | null };
  facts: FactCheck[];
}

export interface EvaluateQuestion {
  set: string;
  id: string;
  question: string;
  category: string | null;
  expect_labels: string[] | "REFUSE" | null;
  found: string[];
  missing: string[];
  outcome: Outcome | null;
  refused: boolean | null;
  hit_rank: number | null;
  best_rank: number | null;
  targets: Target[];
  answer: string | null;
  error: string | null;
  history?: { question: string; answer: string }[] | null;
  standalone_question?: string | null;
  rewrite_ok?: boolean | null;
  trap: boolean;
  scored: boolean;
  /** 1 / the rank of the first answer page in the top 50, 0 when it is not there; null without answer pages. */
  reciprocal: number | null;
  facts: FactCheck[];
}

export interface PassInterval {
  passes: number;
  n: number;
  rate: number | null;
  low: number | null;
  high: number | null;
}

/** Step 12 (`evaluate_lab`). */
export interface EvaluateLab {
  run: RunHead;
  k: number | null;
  deep_k: number | null;
  sets: Record<string, { name: string; scored: boolean }>;
  summary: RunSummary | null;
  /** hit, miss, no_pages (traps) × Pass, Refused, Wrong: the question ids in each cell. */
  grid: Record<"hit" | "miss" | "no_pages", Record<Outcome, string[]>>;
  questions: EvaluateQuestion[];
  interval: PassInterval;
  against: {
    run: RunHead;
    n: number;
    better: string[];
    worse: string[];
    other_changes: string[];
    /** The exact two-sided sign test on `better` vs `worse`. */
    p: number;
    interval: PassInterval;
    config: { key: string; a: unknown; b: unknown }[];
  } | null;
}

export interface SafetyPageChoice {
  source: string;
  page: number;
  title: string | null;
  hidden: number;
  reasons: string[];
}

/** Step 13 (`safety_lab`). Boxes are PDF points: [x0, y0, x1, y1] of a `width` × `height` page. */
export interface SafetyLab {
  source: string;
  page: number;
  width: number;
  height: number;
  hidden: HiddenSpan[];
  changed_pixels: number;
  dpi: number;
  threshold: number;
  /** The crafted page only: the spans the guard kept. */
  kept: { text: string; box: PageBox }[] | null;
  reasons: Record<string, string>;
  flag_kinds: Record<string, string>;
  /** Set D's T3 from the same passages with different prompt versions, per setup where several were measured. */
  trap: {
    index_label: string | null;
    model: string | null;
    versions: Record<string, { run_id: string; label: string | null; outcome: Outcome | null; answer: string }>;
  }[];
  text: { raw: string; clean: string } | null;
  flags: GuardFlag[];
  question: string | null;
  truth: { visible: string; injection: string; white_on_dark: string } | null;
  fence: { version: string; rule: string | null; user: string }[] | null;
}

export interface SafetyLive {
  version: string;
  answer: string;
  refused: boolean;
  says_visible: boolean;
  says_injected: boolean;
  /** Which figures the answer contains; "both" needs a reader (a quoted instruction or an obeyed one). */
  verdict: "kept" | "obeyed" | "both" | "refused" | "neither";
  model: { name: string };
  usage: Usage;
  trace: TraceStep[];
  citations: Citations;
}
