import type { ReactNode } from "react";
import { ArrowDown, ArrowUp, Check, Minus, X } from "lucide-react";

import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { duration } from "@/components/ask/trace";
import type { DocumentFile, NetRecord, RunResult, RunSummary } from "@/lib/api";
import { docTitle } from "@/lib/documents";
import { NO_DATA, num } from "@/lib/format";
import { REWRITE_LABELS } from "@/lib/search";
import { cn } from "@/lib/utils";
import { HitLabel, OutcomeBadge } from "./badges";

/**
 * How one question was searched when the run did more than one plain search
 * (M5): the follow-up made whole, the rewrite, each round of a multi-step
 * search, and what the reranker did to the answer page. Everything shown is
 * what the run recorded (rag_studio/eval.py `_search_one`); a step that did not
 * run is not shown, and a value that was not recorded is "-".
 */
export function SearchSteps({ r, docs }: { r: RunResult; docs: DocumentFile[] | undefined }) {
  const plan = r.rewrites?.find((x) => x.technique === "decompose" || x.technique === "multi" || x.technique === "hyde");
  const steps: ReactNode[] = [];

  if (r.history?.length) {
    const last = r.history[r.history.length - 1];
    steps.push(
      <Step key="followup" title="Follow-up question" cache={r.cache?.followup} net={r.network?.followup}>
        <p className="text-pretty text-muted-foreground">
          Earlier turn: <span className="text-foreground">{last.question}</span>{" "}
          <span title={last.answer}>({clip(last.answer, 140)})</span>
        </p>
        {r.standalone_question ? (
          <p className="flex flex-wrap items-baseline gap-x-2 gap-y-1 text-pretty">
            <span>
              Searched for: <span className="text-foreground">{r.standalone_question}</span>
            </span>
            <RewriteVerdict ok={r.rewrite_ok ?? null} />
          </p>
        ) : (
          <p className="text-muted-foreground">Asked as typed: not rewritten.</p>
        )}
      </Step>,
    );
  }

  if (plan) {
    const queries = plan.value.questions ?? plan.value.queries ?? [];
    steps.push(
      <Step key="rewrite" title={REWRITE_LABELS[plan.technique] ?? plan.technique} cache={plan.cache} net={plan.network ?? r.network?.rewrite}>
        {queries.length > 0 && (
          <ol className="list-decimal space-y-0.5 pl-4">
            {queries.map((q, i) => (
              <li key={i} className="text-pretty text-foreground">
                {q}
              </li>
            ))}
          </ol>
        )}
        {plan.value.passage && (
          <p className="text-pretty">
            Drafted: <span className="text-foreground">{plan.value.passage}</span>
          </p>
        )}
        <p className="text-muted-foreground">The question as asked was searched too, and the lists were combined.</p>
      </Step>,
    );
  }

  if (r.rounds?.length) {
    steps.push(
      <Step key="rounds" title="Multi-step search" cache={r.cache?.agent}>
        <ol className="space-y-1">
          {r.rounds.map((x) => (
            <li key={x.round} className="text-pretty">
              <span className="font-medium tabular-nums">Round {x.round}</span>
              {" · "}
              {x.round === 1 ? (
                <span className="text-muted-foreground">the normal search</span>
              ) : x.action === "search" ? (
                <>
                  searched for <span className="text-foreground">&ldquo;{x.query}&rdquo;</span>
                  {x.doc ? ` in ${docTitle(x.doc, docs)}` : ""}
                  {typeof x.new === "number" ? ` · ${x.new} new in the top 10` : ""}
                </>
              ) : x.action === "answer" ? (
                <span>enough evidence</span>
              ) : (
                <span>stopped</span>
              )}
              {x.round > 1 && x.reason && <span className="block pl-4 text-muted-foreground">{x.reason}</span>}
            </li>
          ))}
        </ol>
        {r.agent && (
          <p className="text-muted-foreground tabular-nums">
            Stopped: {r.agent.stop_text} · {r.agent.rounds_used} rounds used · {num(r.agent.tokens)} tokens ·{" "}
            {duration(r.agent.seconds * 1000)}
          </p>
        )}
      </Step>,
    );
  }

  if (r.rerank) {
    const { before_rank: before, after_rank: after } = r.rerank;
    const up = before !== null && after !== null && after < before;
    const down = before !== null && after !== null && after > before;
    const Icon = up ? ArrowUp : down ? ArrowDown : Minus;
    steps.push(
      <Step key="rerank" title="Reranker" net={r.network?.rerank}>
        <p className="flex flex-wrap items-center gap-1 tabular-nums">
          <Icon
            className={cn("size-3", up && "text-emerald-700 dark:text-emerald-400", down && "text-red-700 dark:text-red-400")}
            aria-hidden
          />
          First answer page: {rankText(before)} before reranking, {rankText(after)} after
        </p>
        <p className="text-muted-foreground tabular-nums">
          {r.rerank.cache === "hit"
            ? `Scores remembered from an earlier run (the model took ${duration(r.rerank.compute_ms)}).`
            : `Scored now: ${duration(r.rerank.compute_ms)} of model time, ${duration(r.timings.rerank_ms)} in all.`}
        </p>
      </Step>,
    );
  }

  if (steps.length === 0) return null;
  return (
    <section className="space-y-3 border-t p-4" aria-label="How it was searched">
      <h4 className="text-[11px] font-medium tracking-wide text-muted-foreground uppercase">How it was searched</h4>
      <div className="grid gap-x-6 gap-y-3 text-xs @2xl/result:grid-cols-2">{steps}</div>
    </section>
  );
}

function Step({
  title,
  cache,
  net,
  children,
}: {
  title: string;
  cache?: string | null;
  net?: NetRecord | null;
  children: ReactNode;
}) {
  return (
    <div className="min-w-0 space-y-1">
      <p className="font-medium">{title}</p>
      {children}
      <Where cache={cache} net={net} />
    </div>
  );
}

/** "Remote AI · from memory, no call was made", or the network and runtime split of a call. */
function Where({ cache, net }: { cache?: string | null; net?: NetRecord | null }) {
  if (cache === "hit") return <p className="text-[11px] text-muted-foreground">Remembered from an earlier run: no call was made.</p>;
  if (!net || net.wall_ms === undefined) return null;
  const server = net.server_ms ?? 0;
  return (
    <p className="text-[11px] text-muted-foreground tabular-nums">
      Network {duration(Math.max(0, net.wall_ms - server))} · Remote AI {duration(server)}
      {net.bytes_out_raw ? ` · sent ${kb(net.bytes_out_raw)} as ${kb(net.bytes_out_wire)}` : ""}
    </p>
  );
}

function RewriteVerdict({ ok }: { ok: boolean | null }) {
  if (ok === null) return null;
  return (
    <span
      className={cn(
        "inline-flex items-center gap-0.5 rounded-md px-1.5 py-0.5 text-[11px] ring-1",
        ok ? "text-emerald-700 ring-emerald-500/40 dark:text-emerald-400" : "text-red-700 ring-red-500/40 dark:text-red-400",
      )}
    >
      {ok ? <Check className="size-3" aria-hidden /> : <X className="size-3" aria-hidden />}
      {ok ? "Rewrite right" : "Rewrite wrong"}
    </span>
  );
}

/**
 * Set E (follow-up questions): each follow-up judged in order, the rewrite
 * first, then retrieval, then the answer. Its own table, because what it
 * measures is the conversation's memory, not the search.
 */
export function FollowupTable({
  results,
  summary,
  asTyped,
}: {
  results: RunResult[];
  summary: NonNullable<RunSummary["followups"]>;
  asTyped: boolean;
}) {
  const rows = results.filter((r) => r.history?.length);
  const of = (n: number, total: number) => (total ? `${n} of ${total}` : NO_DATA);
  return (
    <section className="space-y-2" aria-label="Follow-up questions">
      <div className="space-y-0.5">
        <h3 className="text-sm font-medium">Follow-up questions</h3>
        <p className="text-xs text-pretty text-muted-foreground tabular-nums">
          {asTyped
            ? `Asked as typed, with no rewrite: found ${of(summary.hits, summary.n)}`
            : `Rewritten right ${of(summary.rewrite_ok, summary.rewritten)} · then found ${of(summary.hits, summary.n)}`}
          {summary.scored ? ` · then answered right ${of(summary.passes, summary.scored)}` : " · retrieval only"}
        </p>
      </div>
      <div className="@container -mx-4">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="w-10 pl-4">ID</TableHead>
              <TableHead>Follow-up</TableHead>
              <TableHead className="hidden @lg:table-cell">Rewrite</TableHead>
              <TableHead className="hidden @lg:table-cell">Retrieval</TableHead>
              <TableHead className="pr-4">Outcome</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.map((r) => (
              <TableRow key={`${r.set}:${r.id}`}>
                <TableCell className="pl-4 align-top text-xs text-muted-foreground tabular-nums">{r.id}</TableCell>
                <TableCell className="align-top text-sm whitespace-normal">
                  <p className="text-pretty">{r.question}</p>
                  <p className="text-[11px] text-pretty text-muted-foreground">
                    {r.standalone_question ? `Searched for: ${r.standalone_question}` : "Asked as typed"}
                  </p>
                  <div className="mt-0.5 flex flex-wrap items-center gap-2 text-[11px] @lg:hidden">
                    <RewriteVerdict ok={r.rewrite_ok ?? null} />
                    <HitLabel result={r} />
                  </div>
                </TableCell>
                <TableCell className="hidden align-top @lg:table-cell">
                  {r.rewrite_ok === null || r.rewrite_ok === undefined ? (
                    <span className="text-xs text-muted-foreground">{NO_DATA}</span>
                  ) : (
                    <RewriteVerdict ok={r.rewrite_ok} />
                  )}
                </TableCell>
                <TableCell className="hidden align-top text-sm @lg:table-cell">
                  <HitLabel result={r} />
                </TableCell>
                <TableCell className="pr-4 align-top">
                  <OutcomeBadge outcome={r.outcome} error={r.error} />
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </section>
  );
}

function rankText(rank: number | null): string {
  return rank === null ? "not in the top 50" : `rank ${rank}`;
}

function clip(text: string, n: number): string {
  const t = text.trim();
  return t.length > n ? `${t.slice(0, n).trimEnd()}…` : t;
}

function kb(bytes: number | undefined): string {
  if (bytes === undefined) return NO_DATA;
  return bytes < 1024 ? `${bytes} B` : `${(bytes / 1024).toFixed(1)} KB`;
}
