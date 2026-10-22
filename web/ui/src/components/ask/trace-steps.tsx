import type { ReactNode } from "react";

import type { TraceStep } from "@/lib/api";
import { duration } from "@/components/ask/trace";

/**
 * The steps M5 added to "What happened": a follow-up made whole, a rewrite,
 * each search, the fusion, the reranker's moves and the multi-step search's
 * rounds. Each shows only what the engine measured: where a step ran (this
 * computer or Remote AI), and, for Remote AI, how much of the time was the
 * network and how much the runtime (the cache says when no call was made).
 */
type Net = { calls?: number; network_ms?: number; server_ms?: number; bytes_out_raw?: number; bytes_out_wire?: number };
type Move = { chunk_id: string; from: number; to: number };

const d = (step: TraceStep) => step.detail as Record<string, unknown>;

/** "p19 c2" from "wb_gep_2026_06:p19:c2". */
export function shortId(chunkId: string): string {
  const [, page, c] = chunkId.split(":");
  return page && c ? `${page} ${c}` : chunkId;
}

/** Names the lists a fusion took, in plain words (docs/product-language.md). */
export function listName(name: string): string {
  if (name === "bm25") return "BM25";
  if (name === "dense") return "Dense";
  if (name === "original") return "The question";
  if (name === "hyde") return "Drafted passage";
  if (name === "reranked") return "Reranked";
  if (/^q\d+$/.test(name)) return `Phrasing ${name.slice(1)}`;
  if (/^r\d+$/.test(name)) return `Round ${name.slice(1)}`;
  return name;
}

function Where({ step }: { step: TraceStep }) {
  const x = d(step);
  const net = x.network as Net | null | undefined;
  const cache = x.cache as string | null | undefined;
  const remote = x.where === "remote" || x.where === "colab";
  if (!remote && !net) return null;
  return (
    <p>
      {remote ? "Remote AI" : "This computer"}
      {cache === "hit" && " · from memory, no call was made"}
      {cache !== "hit" && net && (net.network_ms !== undefined || net.server_ms !== undefined) && (
        <>
          {" "}· Network {duration(net.network_ms ?? 0)} · Remote AI {duration(net.server_ms ?? 0)}
          {net.bytes_out_raw ? ` · sent ${Math.round((net.bytes_out_raw ?? 0) / 1024)} KB as ${Math.round((net.bytes_out_wire ?? 0) / 1024)} KB` : ""}
        </>
      )}
    </p>
  );
}

function Box({ children }: { children: ReactNode }) {
  return <div className="col-start-2 col-end-4 space-y-1 pt-1 text-[11px] text-muted-foreground">{children}</div>;
}

export function StepDetail({ step }: { step: TraceStep }): ReactNode {
  const x = d(step);
  switch (step.id) {
    case "followup":
      return (
        <Box>
          <p>
            Asked: <span className="text-foreground">{String(x.asked)}</span>
          </p>
          <p>
            Searched for: <span className="text-foreground">{String(x.searched_for)}</span>
            {x.changed === false && " (already complete)"}
          </p>
          <Where step={step} />
        </Box>
      );
    case "rewrite": {
      const queries = (x.queries as string[] | undefined) ?? [];
      return (
        <Box>
          {queries.map((q, i) => (
            <p key={i} className="truncate" title={q}>
              {i + 1}. <span className="text-foreground">{q}</span>
            </p>
          ))}
          {typeof x.passage === "string" && x.passage && (
            <p title={x.passage}>
              Drafted: <span className="text-foreground">{x.passage.slice(0, 160)}{x.passage.length > 160 ? "…" : ""}</span>
            </p>
          )}
          <Where step={step} />
        </Box>
      );
    }
    case "fuse": {
      const lists = (x.lists as string[] | undefined) ?? [];
      return (
        <Box>
          <p>{lists.map(listName).join(" + ")}</p>
        </Box>
      );
    }
    case "rerank": {
      const moves = ((x.moves as Move[] | undefined) ?? []).slice(0, 5);
      return (
        <Box>
          {moves.map((m) => (
            <p key={m.chunk_id} className="font-mono">
              {shortId(m.chunk_id)}: <span className="font-sans">was {m.from}, now {m.to}</span>
            </p>
          ))}
          <Where step={step} />
        </Box>
      );
    }
    case "round":
      return (
        <Box>
          {typeof x.reason === "string" && <p>Why: {x.reason}</p>}
          {typeof x.stop === "string" && <p>Stopped: {x.stop}</p>}
          {typeof x.new === "number" && <p>{x.new} new passages in the top 10</p>}
          <Where step={step} />
        </Box>
      );
    default:
      return null;
  }
}
