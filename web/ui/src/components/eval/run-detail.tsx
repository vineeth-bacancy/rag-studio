import { useState } from "react";
import { Check, ChevronDown, Square, X } from "lucide-react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { Card, CardAction, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Tile, TileGrid } from "@/components/common/tiles";
import { TechnicalDetails } from "@/components/common/technical-details";
import { duration } from "@/components/ask/trace";
import { api, type DocumentFile, type Run, type RunResult } from "@/lib/api";
import { docTitle } from "@/lib/documents";
import { NO_DATA, dateTimeLabel, num } from "@/lib/format";
import { REWRITE_LABELS, ranksText, searchName, setupExtras } from "@/lib/search";
import { cn } from "@/lib/utils";
import { HitLabel, OutcomeBadge, RunStatusBadge, recallText } from "./badges";
import { rankingText } from "./ladder";
import { FollowupTable, SearchSteps } from "./search-steps";

/**
 * One run, question by question: where the answer page ranked, what the model
 * said, which expected facts it contained. While a run is in progress this
 * refreshes as each question finishes.
 */
export function RunDetail({
  run,
  docs,
  onClose,
}: {
  run: Run;
  docs: DocumentFile[] | undefined;
  onClose: () => void;
}) {
  const s = run.summary;
  const pct = run.progress.total ? (run.progress.done / run.progress.total) * 100 : 0;
  const cfg = run.config;
  const recall = Object.entries(s?.recall ?? {}).filter(([, v]) => v.total > 0);
  const ranking = s?.ranking;
  const ret = cfg.retrieval;
  // A run records each M5 switch as its full description (eval.py `_config`).
  const rerank = ret.rerank && typeof ret.rerank === "object" ? (ret.rerank as Record<string, unknown>) : null;
  const rewrite = ret.rewrite && typeof ret.rewrite === "object" ? (ret.rewrite as Record<string, unknown>) : null;
  const agent = ret.agent ?? null;
  const extras = setupExtras({
    rerank: typeof ret.rerank === "string" ? ret.rerank : null,
    rerank_name: rerank ? rerankName(rerank) : null,
    rewrite: rewrite ? String(rewrite.technique) : typeof ret.rewrite === "string" ? ret.rewrite : null,
    agent_rounds: agent ? Number(agent.rounds) : (ret.agent_rounds ?? null),
  });
  const asTyped = cfg.followup?.as_typed === true;

  async function cancel() {
    try {
      await api.cancelRun(run.id);
      toast("Stopping after the current question.");
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Couldn't cancel the run.");
    }
  }

  return (
    <Card id="run-detail">
      <CardHeader>
        <CardTitle as="h2">{run.label ?? Object.values(run.sets).map((x) => x.name).join(", ")}</CardTitle>
        <CardDescription>
          {dateTimeLabel(run.started_at)} · {run.spec.mode === "answer" ? `answers by ${cfg.answer?.model}` : "retrieval only"} · top{" "}
          {cfg.k} passages · {cfg.index.label} index ·{" "}
          {[searchName(cfg.retrieval?.method, cfg.retrieval?.section_cap), ...extras].join(" · ")}
          {cfg.followup ? (asTyped ? " · follow-ups asked as typed" : " · follow-ups rewritten") : ""}
          {cfg.answer ? ` · prompt ${cfg.answer.prompt_version}` : ""}
          {run.seconds !== null ? ` · ${duration(run.seconds * 1000)}` : ""}
          {s?.scored ? ` · ${s.scored} answers scored` : ""}
        </CardDescription>
        <CardAction className="flex items-center gap-1">
          <RunStatusBadge status={run.status} />
          <Button variant="ghost" size="icon-sm" onClick={onClose} aria-label="Close this run">
            <X aria-hidden />
          </Button>
        </CardAction>
      </CardHeader>
      <CardContent className="space-y-5">
        {run.status === "running" && (
          <div className="flex flex-wrap items-center gap-3">
            <Progress value={pct} className="min-w-48 flex-1" aria-label="Run progress" />
            <span className="text-xs text-muted-foreground tabular-nums">
              {run.progress.done} of {run.progress.total} questions
            </span>
            <Button variant="outline" size="sm" onClick={cancel}>
              <Square aria-hidden />
              Cancel
            </Button>
          </div>
        )}
        {run.error && (
          <p className="text-sm text-red-700 dark:text-red-400">
            The run stopped: {run.error}
          </p>
        )}

        {/* Two tile rows, each exactly as wide as its tile count: a grid
            with an empty cell would show the separator colour as a block. */}
        <div className="grid gap-3 sm:grid-cols-2">
          {recall.length > 0 && (
            <TileGrid style={{ gridTemplateColumns: `repeat(${recall.length}, minmax(0, 1fr))` }}>
              {recall.map(([sid, v]) => (
                <Tile key={sid} label={`${run.sets[sid]?.name ?? sid} recall@${cfg.k}`} value={recallText(v)} />
              ))}
            </TileGrid>
          )}
          {s?.outcomes && (
            <TileGrid className="grid-cols-3">
              {(["PASS", "REFUSED", "WRONG"] as const).map((o) => (
                <Tile
                  key={o}
                  label={{ PASS: "Pass", REFUSED: "Refused", WRONG: "Wrong" }[o]}
                  value={s.outcomes![o]}
                  valueClass={cn(o === "PASS" && "text-emerald-700 dark:text-emerald-400", o === "WRONG" && s.outcomes![o] > 0 && "text-red-700 dark:text-red-400")}
                />
              ))}
            </TileGrid>
          )}
          {ranking && ranking.n > 0 && (
            <TileGrid className="grid-cols-2">
              <Tile
                label="MRR (50 deep)"
                value={ranking.mrr !== null ? ranking.mrr.toFixed(3) : NO_DATA}
                hint="1 when every first answer page is at rank 1"
              />
              <Tile
                label="Recall@20"
                value={ranking.recall20 !== null ? `${ranking.recall20}/${ranking.n}` : NO_DATA}
                hint="Answer page in the top 20"
              />
            </TileGrid>
          )}
        </div>
        {ranking && ranking.multi.n > 0 && (
          <p className="-mt-2 text-xs text-pretty text-muted-foreground tabular-nums">
            Questions about one document: {rankingText(ranking.single)}. Questions that need two documents:{" "}
            {rankingText(ranking.multi)}.
          </p>
        )}

        {s?.followups && <FollowupTable results={run.results} summary={s.followups} asTyped={asTyped} />}

        <div className="@container -mx-4">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-10 pl-4">ID</TableHead>
                <TableHead>Question</TableHead>
                <TableHead className="hidden @lg:table-cell">Retrieval</TableHead>
                <TableHead className="pr-4">Outcome</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {run.results.map((r) => (
                <ResultRow key={`${r.set}:${r.id}`} r={r} docs={docs} />
              ))}
            </TableBody>
          </Table>
        </div>

        <TechnicalDetails
          label="Technical details: this run's setup"
          rows={{
            Run: run.id,
            Index: `${cfg.index.label} · ${cfg.index.table}`,
            Built: cfg.index.built_at,
            Parser: [cfg.index.config.parser, cfg.index.config.parser_version].filter(Boolean).join(" ") || NO_DATA,
            Cleaner: typeof cfg.index.config.cleaner === "string" ? cfg.index.config.cleaner : NO_DATA,
            Embedding: `${cfg.index.embedding.model} · ${cfg.index.embedding.dimensions} dims · ${cfg.index.embedding.quantization ?? NO_DATA}`,
            "Embedding digest": cfg.index.embedding.digest.slice(0, 19),
            Search: [
              searchName(cfg.retrieval.method, cfg.retrieval.section_cap),
              cfg.retrieval.metric && `${cfg.retrieval.metric}, ${cfg.retrieval.exact ? "exact" : "approximate"}`,
              cfg.retrieval.fusion && `${cfg.retrieval.fusion.toUpperCase()} k = ${cfg.retrieval.rrf_k}, ${cfg.retrieval.depth} per list`,
              `top ${cfg.k}`,
            ].filter(Boolean).join(" · "),
            "Word index": cfg.retrieval.word_index
              ? `${cfg.retrieval.word_index.column} · ${Object.entries(cfg.retrieval.word_index.settings ?? {})
                  .map(([k, v]) => `${k} ${String(v)}`)
                  .join(", ")}`
              : NO_DATA,
            ...(rerank
              ? {
                  Reranker: `${rerank.model} @ ${String(rerank.revision).slice(0, 8)} · ${rerank.dtype}${rerank.quantization ? ` (${rerank.quantization})` : ""} · ${rerank.where === "colab" ? "Remote AI" : "this computer"} · top ${rerank.depth}`,
                  "Reranker instruction": `${rerank.instruction_version} · sha256 ${String(rerank.instruction_sha256).slice(0, 12)}`,
                }
              : {}),
            ...(rewrite
              ? {
                  "Query rewriting": `${REWRITE_LABELS[String(rewrite.technique)] ?? rewrite.technique} · ${rewrite.model} on Remote AI · ${rewrite.prompt_version} · sha256 ${String(rewrite.prompt_sha256).slice(0, 12)}${rewrite.embed_as ? ` · draft embedded as a ${rewrite.embed_as}` : ""}`,
                }
              : {}),
            ...(agent
              ? {
                  "Multi-step search": `up to ${agent.rounds} rounds · ${agent.budget_s} s · ${num(Number(agent.budget_tokens))} tokens · ${agent.model} · ${agent.prompt_version} · sha256 ${String(agent.prompt_sha256).slice(0, 12)}`,
                  "Document filter": String(agent.filter ?? NO_DATA),
                }
              : {}),
            ...(cfg.followup
              ? {
                  "Follow-ups": asTyped
                    ? "asked as typed, not rewritten"
                    : `${cfg.followup.model} on Remote AI · ${cfg.followup.prompt_version} · sha256 ${String(cfg.followup.prompt_sha256).slice(0, 12)}`,
                }
              : {}),
            ...(cfg.runtime
              ? {
                  "Where it ran": Object.entries(cfg.runtime)
                    .map(([part, facts]) => `${part}: ${Object.values(facts ?? {}).map(String).join(", ")}`)
                    .join(" · "),
                }
              : {}),
            "Answer model": cfg.answer ? `${cfg.answer.model} · ${cfg.answer.parameter_size ?? NO_DATA} · ${cfg.answer.quantization ?? NO_DATA}` : NO_DATA,
            "Model digest": cfg.answer?.digest?.slice(0, 19) ?? NO_DATA,
            Options: cfg.answer ? `thinking ${cfg.answer.think ? "on" : "off"}, temperature ${cfg.answer.temperature}, num_ctx ${cfg.answer.num_ctx}` : NO_DATA,
            Prompt: cfg.answer ? `${cfg.answer.prompt_version} · sha256 ${cfg.answer.prompt_sha256.slice(0, 12)}` : NO_DATA,
            Ollama: cfg.ollama_version,
            App: cfg.app_version,
          }}
        />
      </CardContent>
    </Card>
  );
}

function ResultRow({ r, docs }: { r: RunResult; docs: DocumentFile[] | undefined }) {
  const [open, setOpen] = useState(false);
  const facts = Array.isArray(r.expect_labels) ? r.expect_labels : null;
  return (
    <>
      {/* Open: the question and its panel share one tinted band. The tint is
          set with the same variant the table row uses (has-aria-expanded), so
          tailwind-merge replaces it instead of both applying. */}
      <TableRow className={cn("has-aria-expanded:bg-muted/30", open && "border-b-0")}>
        <TableCell className="pl-4 align-top text-xs text-muted-foreground tabular-nums">{r.id}</TableCell>
        <TableCell className="align-top whitespace-normal">
          <button
            type="button"
            onClick={() => setOpen((v) => !v)}
            aria-expanded={open}
            className="group flex w-full items-start gap-1.5 rounded-sm text-left text-sm focus-visible:ring-[3px] focus-visible:ring-ring/50 focus-visible:outline-none"
          >
            <ChevronDown className={cn("mt-0.5 size-3.5 shrink-0 text-muted-foreground transition-transform", open && "rotate-180")} aria-hidden />
            <span className="text-pretty group-hover:underline group-hover:underline-offset-2">{r.question}</span>
          </button>
          <p className="mt-0.5 pl-5 text-[11px] text-muted-foreground">
            {r.category ? `${r.category} · ` : ""}
            {r.expect === "REFUSE" ? "Trap: must refuse" : facts ? `Expects ${facts.join(", ")}` : "Retrieval only"}
          </p>
          <p className="pl-5 text-[11px] @lg:hidden">
            <HitLabel result={r} />
          </p>
        </TableCell>
        <TableCell className="hidden align-top text-sm @lg:table-cell">
          <HitLabel result={r} />
        </TableCell>
        <TableCell className="pr-4 align-top">
          <OutcomeBadge outcome={r.outcome} error={r.error} />
        </TableCell>
      </TableRow>
      {open && (
        <TableRow className="bg-muted/30 hover:bg-muted/30">
          {/* The panel starts under the question, not under the ID. */}
          <TableCell aria-hidden className="p-0" />
          <TableCell colSpan={3} className="pt-2 pr-4 pb-5 whitespace-normal">
            <ResultPanel r={r} facts={facts} docs={docs} />
          </TableCell>
        </TableRow>
      )}
    </>
  );
}

/**
 * One question's evidence: what the model answered and which expected facts
 * it holds (left), and the passages search returned, answer pages marked
 * (right). Side by side when the card is wide, stacked when it is narrow.
 */
function ResultPanel({ r, facts, docs }: { r: RunResult; facts: string[] | null; docs: DocumentFile[] | undefined }) {
  const trap = r.expect === "REFUSE";
  return (
    <div className="@container/result overflow-hidden rounded-lg bg-card shadow-xs ring-1 ring-foreground/10">
      <div className="grid divide-y @2xl/result:grid-cols-[minmax(0,3fr)_minmax(0,2fr)] @2xl/result:divide-x @2xl/result:divide-y-0">
        <section className="min-w-0 space-y-3 p-4" aria-label="Answer">
          <h4 className="text-[11px] font-medium tracking-wide text-muted-foreground uppercase">Answer</h4>
          {r.answer !== null ? (
            <p className="max-w-prose text-sm leading-relaxed text-pretty whitespace-pre-wrap">{r.answer.trim()}</p>
          ) : (
            <p className="text-sm text-muted-foreground">
              {r.expect === null ? "Retrieval only: this question is not answered or scored." : "No answer was recorded."}
            </p>
          )}
          {r.error && <p className="text-xs text-amber-700 dark:text-amber-400">{r.error}</p>}
          {r.answer !== null && (facts || trap) && (
            <div className="space-y-1.5">
              <p className="text-[11px] text-muted-foreground">
                {trap ? "A trap: the only right answer is the refusal" : "Expected facts"}
              </p>
              <ul className="flex flex-wrap gap-1.5">
                {trap ? (
                  <FactChip ok={r.refused === true} label={r.refused ? "Refused" : "Answered instead of refusing"} />
                ) : (
                  facts?.map((f) => <FactChip key={f} ok={r.found.includes(f)} label={f} />)
                )}
              </ul>
            </div>
          )}
        </section>

        <section className="min-w-0 space-y-2 p-4" aria-label="Retrieved passages">
          <h4 className="text-[11px] font-medium tracking-wide text-muted-foreground uppercase">
            Top {r.hits.length} passages
          </h4>
          <p className="text-[11px] text-pretty text-muted-foreground">
            {r.targets.length
              ? r.targets.map((t) => `Answer pages in ${docTitle(t.doc_id, docs)}: ${t.pages.join(", ")}`).join(". ")
              : "No answer pages: the documents don't contain this."}
          </p>
          <ol className="divide-y divide-foreground/5 text-xs">
            {r.hits.map((h) => {
              const onTarget = r.targets.some((t) => t.doc_id === h.doc_id && t.pages.includes(h.page));
              const hit = onTarget && "font-medium text-emerald-700 dark:text-emerald-400";
              return (
                <li
                  key={h.chunk_id}
                  className="grid grid-cols-[1.25rem_minmax(0,1fr)_auto_2.75rem] items-center gap-x-2 py-1.5 tabular-nums"
                  title={[ranksText(h.ranks), h.capped && "moved down by the section limit"].filter(Boolean).join(" · ") || undefined}
                >
                  <span className="text-muted-foreground">{h.rank}</span>
                  <span className={cn("truncate", hit)} title={docTitle(h.doc_id, docs)}>
                    {onTarget && <Check className="mr-1 inline size-3 align-[-1px]" aria-label="answer page" />}
                    {docTitle(h.doc_id, docs)}
                  </span>
                  <span className={cn("text-muted-foreground", hit)}>p. {h.page}</span>
                  <span className="text-right text-muted-foreground">{h.score.toFixed(3)}</span>
                </li>
              );
            })}
          </ol>
          {r.hit_rank === null && r.targets.length > 0 && (
            <p className="text-[11px] text-muted-foreground">
              {r.best_rank !== null
                ? `The first answer page ranked ${r.best_rank}, outside the ${r.hits.length} passages the model saw.`
                : "No answer page in the top 50."}
            </p>
          )}
        </section>
      </div>

      <SearchSteps r={r} docs={docs} />

      {r.usage && (
        <div className="flex flex-wrap gap-x-4 gap-y-1 border-t bg-muted/40 px-4 py-2 text-[11px] text-muted-foreground tabular-nums">
          <span>{duration(r.timings.generate_ms)} to answer</span>
          <span>{num(r.usage.prompt_tokens)} prompt tokens</span>
          <span>{num(r.usage.answer_tokens)} answer tokens</span>
          {r.citations && <span>Cited {r.citations.valid.map((n) => `[${n}]`).join("") || "nothing"}</span>}
          {r.citations && r.citations.invalid.length > 0 && (
            <span className="text-red-700 dark:text-red-400">
              Invalid {r.citations.invalid.map((n) => `[${n}]`).join("")}
            </span>
          )}
        </div>
      )}
    </div>
  );
}

function FactChip({ ok, label }: { ok: boolean; label: string }) {
  const Icon = ok ? Check : X;
  return (
    <li
      className={cn(
        "inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-xs ring-1",
        ok ? "text-emerald-700 ring-emerald-500/40 dark:text-emerald-400" : "text-red-700 ring-red-500/40 dark:text-red-400",
      )}
    >
      <Icon className="size-3" aria-hidden />
      <span className="sr-only">{ok ? "found:" : "missing:"}</span>
      {label}
    </li>
  );
}

/** "Remote AI · 4B", as the pickers name it (rag_studio/rerank.py labels), from a run's recorded reranker. */
function rerankName(rr: Record<string, unknown>): string {
  const size = String(rr.model ?? rr.id ?? "").split("-").pop() ?? "";
  const where = rr.where === "colab" ? "Remote AI" : "This computer";
  return `${where} · ${size}${rr.quantization ? ` (${rr.quantization === "int8" ? "8-bit" : String(rr.quantization)})` : ""}`;
}
