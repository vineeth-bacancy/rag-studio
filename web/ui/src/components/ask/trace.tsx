import { AlertTriangle, ChevronRight, Cpu } from "lucide-react";

import { TechnicalDetails } from "@/components/common/technical-details";
import type { AskDone, AskReady, TraceStep } from "@/lib/api";
import { NO_DATA, num } from "@/lib/format";
import { listName, shortId, StepDetail } from "@/components/ask/trace-steps";
import { searchName } from "@/lib/search";
import { cn } from "@/lib/utils";

/** "1.2 s", "340 ms". */
export function duration(ms: number | null | undefined): string {
  if (ms === null || ms === undefined || !Number.isFinite(ms)) return NO_DATA;
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(ms < 10_000 ? 1 : 0)} s`;
}

/**
 * "What happened": every step that ran for this answer, how long it took,
 * and what went into the prompt. Each row is a step that finished and was
 * timed by the engine - nothing here is estimated except where it says so.
 */
export function WhatHappened({ steps, ready, done }: { steps: TraceStep[]; ready?: AskReady; done?: AskDone }) {
  const total = done?.total_ms ?? steps.reduce((t, s) => t + s.ms, 0);
  return (
    <details className="group rounded-lg ring-1 ring-foreground/10">
      <summary className="flex cursor-pointer list-none items-center gap-2 rounded-lg px-3 py-2 text-sm select-none hover:bg-muted/50 focus-visible:ring-[3px] focus-visible:ring-ring/50 focus-visible:outline-none [&::-webkit-details-marker]:hidden">
        <ChevronRight className="size-4 text-muted-foreground transition-transform group-open:rotate-90" aria-hidden />
        <span className="font-medium">What happened</span>
        <span className="ml-auto text-xs text-muted-foreground tabular-nums">
          {steps.length} steps · {duration(total)}
        </span>
      </summary>
      <div className="space-y-4 border-t px-3 py-3">
        <ol className="space-y-2">
          {steps.map((s, i) => (
            <StepRow key={`${s.id}-${i}`} step={s} index={i + 1} total={total} />
          ))}
        </ol>
        {ready && <PromptBudget ready={ready} done={done} />}
        {ready && <ModelLine ready={ready} done={done} />}
      </div>
    </details>
  );
}

function StepRow({ step, index, total }: { step: TraceStep; index: number; total: number }) {
  const share = total > 0 ? Math.max(1, (step.ms / total) * 100) : 0;
  return (
    <li className="grid grid-cols-[1.5rem_minmax(0,1fr)_4.5rem] items-center gap-x-2 text-sm">
      <span className="text-xs text-muted-foreground tabular-nums">{index}.</span>
      <div className="min-w-0 space-y-1">
        <p className="truncate">{step.label}</p>
        <div className="h-1 rounded-full bg-muted" aria-hidden>
          <div className="h-1 rounded-full bg-chart-1/70" style={{ width: `${share}%` }} />
        </div>
      </div>
      <span className="text-right text-xs text-muted-foreground tabular-nums">{duration(step.ms)}</span>
      {step.id === "search" && <SearchLists detail={step.detail} />}
      <StepDetail step={step} />
    </li>
  );
}

type ListEntry = { rank: number; chunk_id: string };

/**
 * The lists a search fused or limited, top first: for hybrid, what dense
 * search and BM25 each found before RRF merged them. "p19 c2" is page 19's
 * passage c2 (passages on a page are numbered from c0).
 */
function SearchLists({ detail }: { detail: TraceStep["detail"] }) {
  const lists = detail.lists as unknown as Record<string, ListEntry[]> | undefined;
  const method = detail.method as string | undefined;
  const capped = Number(detail.capped ?? 0);
  if (!lists || (Object.keys(lists).length < 2 && !capped && !detail.section_cap)) return null;
  return (
    <div className="col-start-2 col-end-4 space-y-1 pt-1 text-[11px] text-muted-foreground">
      <p>
        {searchName(method, detail.section_cap as number | null)}
        {method === "hybrid" && ` · fused by rank (RRF, k = ${detail.rrf_k ?? 60}) from ${detail.depth ?? 50} per list`}
        {capped > 0 && ` · ${capped} passage${capped === 1 ? "" : "s"} moved down by the section limit`}
      </p>
      {Object.entries(lists).map(([name, hits]) => (
        <p key={name} className="truncate font-mono" title={hits.map((h) => `${h.rank}. ${h.chunk_id}`).join("\n")}>
          <span className="font-sans">{listName(name)}:</span>{" "}
          {hits.slice(0, 5).map((h) => shortId(h.chunk_id)).join(" · ")}
        </p>
      ))}
    </div>
  );
}

/**
 * How full the model's context window was. Before the answer the number is
 * an estimate from the character count; afterwards it is the model's own
 * count. Ollama does not fail a prompt that overflows `num_ctx` - it drops the
 * START of the prompt and answers anyway - so an overflow is called out.
 */
function PromptBudget({ ready, done }: { ready: AskReady; done?: AskDone }) {
  const counted = done?.usage.prompt_tokens ?? null;
  const tokens = counted ?? ready.prompt.estimated_tokens;
  const pct = Math.min(100, (tokens / ready.prompt.num_ctx) * 100);
  const truncated = done?.usage.truncated === true;
  return (
    <div className="space-y-1.5">
      <div className="flex flex-wrap items-baseline justify-between gap-x-3 text-xs">
        <p className="font-medium">Context window</p>
        <p className="text-muted-foreground tabular-nums">
          {counted === null ? "about " : ""}
          {num(tokens)} of {num(ready.prompt.num_ctx)} tokens
          {counted === null ? " (estimated)" : ""} · {ready.prompt.passages} passages
        </p>
      </div>
      <div className="h-2 rounded-full bg-muted" role="img" aria-label={`${Math.round(pct)}% of the context window used`}>
        <div className={cn("h-2 rounded-full", truncated ? "bg-red-500" : "bg-chart-1")} style={{ width: `${Math.max(pct, 1)}%` }} />
      </div>
      {truncated && (
        <p className="flex items-start gap-1.5 text-xs text-red-700 dark:text-red-400">
          <AlertTriangle className="mt-0.5 size-3.5 shrink-0" aria-hidden />
          The model read far fewer tokens than the prompt holds: the start of the prompt was likely dropped.
        </p>
      )}
      <TechnicalDetails label="The exact messages">
        <div className="space-y-2">
          {ready.prompt.messages.map((m, i) => (
            <div key={i} className="space-y-1">
              <p className="font-medium text-muted-foreground">{m.role}</p>
              <pre className="max-h-64 overflow-auto rounded bg-background/60 p-2 font-mono text-[11px] leading-relaxed whitespace-pre-wrap">
                {m.content}
              </pre>
            </div>
          ))}
          <p className="text-muted-foreground">
            Prompt {ready.prompt.version} · sha256 <span className="font-mono">{ready.prompt.sha256.slice(0, 12)}</span>
          </p>
        </div>
      </TechnicalDetails>
    </div>
  );
}

function ModelLine({ ready, done }: { ready: AskReady; done?: AskDone }) {
  const u = done?.usage;
  const speed = u?.answer_tokens && u.answer_ms ? `${(u.answer_tokens / (u.answer_ms / 1000)).toFixed(1)} tokens/s` : NO_DATA;
  return (
    <div className="space-y-2">
      <p className="flex items-center gap-2 text-xs">
        <Cpu className="size-3.5 text-muted-foreground" aria-hidden />
        <span className="font-medium">{ready.model.backend === "local" ? "Local AI" : "Remote AI"}</span>
        <span className="text-muted-foreground">
          · {ready.model.name} answered ·{" "}
          {ready.search?.method === "bm25"
            ? `BM25 searched ${num(ready.index.rows)} passages by words`
            : ready.search?.method === "hybrid"
              ? `${ready.index.embedding.model} and BM25 searched ${num(ready.index.rows)} passages`
              : `${ready.index.embedding.model} searched ${num(ready.index.rows)} passages`}
        </span>
      </p>
      <TechnicalDetails
        rows={{
          "Answer model": ready.model.name,
          Thinking: ready.model.think ? "on" : "off",
          Temperature: String(ready.model.temperature),
          "Context (num_ctx)": num(ready.model.num_ctx),
          "Prompt tokens": num(u?.prompt_tokens),
          "Answer tokens": num(u?.answer_tokens),
          "First token": duration(u?.first_token_ms),
          "Model load": duration(u?.load_ms),
          "Reading prompt": duration(u?.prompt_ms),
          Writing: `${duration(u?.answer_ms)} · ${speed}`,
          Index: `${ready.index.label} · ${ready.index.table}`,
          Search: ready.search
            ? [searchName(ready.search.method, ready.search.section_cap),
               ready.search.metric && `${ready.search.metric}${ready.search.exact ? ", exact" : ""}`]
                .filter(Boolean).join(" · ")
            : "Dense",
          Embedding: `${ready.index.embedding.model} · ${ready.index.embedding.dimensions} dims`,
          "Query prefix": JSON.stringify(ready.index.embedding.query_prefix),
          Endpoint: ready.model.host,
        }}
      />
    </div>
  );
}
