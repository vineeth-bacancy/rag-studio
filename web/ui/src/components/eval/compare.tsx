import { useState } from "react";
import useSWR from "swr";
import { ArrowDown, ArrowRightLeft, ArrowUp, ChevronDown } from "lucide-react";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { Switch } from "@/components/ui/switch";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { ErrorState } from "@/components/common/states";
import type { CompareRow, CompareSide, GoldenSet, Outcome, RunBrief, RunComparison } from "@/lib/api";
import { NO_DATA, dateTimeLabel } from "@/lib/format";
import { searchName } from "@/lib/search";
import { cn } from "@/lib/utils";
import { OutcomeBadge, recallText } from "./badges";
import { OutcomeCounts } from "./ladder";

/**
 * Two runs side by side (spec §8): what differs in their setup, and every
 * question both asked, with the rows whose outcome or retrieval changed
 * marked. A is "before", B is "after". The comparison is computed by the
 * engine (eval.py `compare()`); this only shows it.
 */
export function RunCompare({
  runs,
  sets,
  a,
  b,
  onPick,
}: {
  runs: RunBrief[];
  sets: GoldenSet[];
  a: string | null;
  b: string | null;
  onPick: (a: string | null, b: string | null) => void;
}) {
  const finished = runs.filter((r) => r.status === "done");
  // Until the user picks, compare the two newest finished runs.
  const before = a ?? finished[1]?.id ?? null;
  const after = b ?? finished[0]?.id ?? null;
  const same = before !== null && before === after;
  const { data, error, isLoading, mutate } = useSWR<RunComparison>(
    before && after && !same ? `/eval/compare?a=${before}&b=${after}` : null,
  );
  const labels = Object.fromEntries(finished.map((r) => [r.id, runName(r)]));

  return (
    <Card id="run-compare">
      <CardHeader>
        <CardTitle as="h2">Compare two runs</CardTitle>
        <CardDescription>
          Before and after one change: which questions moved, and what in the setup differed. Rows that changed are
          marked.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-5">
        <div className="grid gap-3 sm:grid-cols-2">
          {(
            [
              ["compare-a", "Before (A)", before, (v: string) => onPick(v, after)],
              ["compare-b", "After (B)", after, (v: string) => onPick(before, v)],
            ] as const
          ).map(([id, label, value, pick]) => (
            <div key={id} className="min-w-0 space-y-1.5">
              <label htmlFor={id} className="text-sm font-medium">
                {label}
              </label>
              <Select items={labels} value={value} onValueChange={(v) => pick(v as string)}>
                <SelectTrigger id={id} className="w-full">
                  <SelectValue placeholder="Choose a run" />
                </SelectTrigger>
                <SelectContent>
                  {finished.map((r) => (
                    <SelectItem key={r.id} value={r.id}>
                      {runName(r)}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          ))}
        </div>

        {same ? (
          <p className="text-sm text-muted-foreground">Both sides are the same run. Choose a different run for A or B.</p>
        ) : error ? (
          <ErrorState message={error.message} error={error} onRetry={() => void mutate()} />
        ) : isLoading || !data ? (
          <Skeleton className="h-40 w-full" />
        ) : (
          <ComparisonBody data={data} sets={sets} />
        )}
      </CardContent>
    </Card>
  );
}

/** "Set A, Set B · 15 Oct, 19:03 · PyMuPDF + cleaning", or the run's own label. */
export function runName(r: RunBrief): string {
  const what = r.label ?? Object.values(r.sets).map((s) => s.name).join(", ");
  const index = r.config.index_label ? ` · ${r.config.index_label}` : "";
  const search = (r.config.retrieval && r.config.retrieval !== "dense") || r.config.section_cap
    ? ` · ${searchName(r.config.retrieval, r.config.section_cap)}`
    : "";
  const prompt = r.spec.mode === "answer" && r.config.prompt_version ? ` · prompt ${r.config.prompt_version}` : "";
  return `${what} · ${dateTimeLabel(r.started_at)}${index}${search}${prompt}`;
}

function ComparisonBody({ data, sets }: { data: RunComparison; sets: GoldenSet[] }) {
  const [changedOnly, setChangedOnly] = useState(false);
  const noTarget = new Set(
    sets.flatMap((s) => s.questions.filter((q) => !q.doc_id && !q.answers?.length).map((q) => `${s.id}:${q.id}`)),
  );
  const setName = (sid: string) => sets.find((s) => s.id === sid)?.name ?? sid;
  const recallSets = Object.keys({ ...data.summary.a.recall, ...data.summary.b.recall }).filter(
    (sid) => (data.summary.a.recall[sid]?.total ?? 0) > 0 || (data.summary.b.recall[sid]?.total ?? 0) > 0,
  );
  const rows = changedOnly ? data.rows.filter(isChanged) : data.rows;

  return (
    <div className="space-y-5">
      <section className="space-y-2" aria-labelledby="compare-setup">
        <h3 id="compare-setup" className="text-sm font-medium">
          What differs in the setup
        </h3>
        {data.config.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            Nothing: both runs used the same setup. Any difference below is noise, not a change.
          </p>
        ) : (
          <dl className="grid gap-x-4 gap-y-1.5 rounded-lg bg-muted/40 p-3 text-xs sm:grid-cols-[10rem_minmax(0,1fr)]">
            {data.config.filter((c) => !HIDDEN_KEYS.has(c.key)).map((c) => (
              <div key={c.key} className="contents">
                <dt className="font-medium text-muted-foreground" title={c.key}>
                  {keyLabel(c.key)}
                </dt>
                <dd className="min-w-0 break-words tabular-nums">
                  <span className="text-muted-foreground">{valueText(c.key, c.a)}</span>
                  <span aria-hidden className="px-1.5 text-muted-foreground">
                    →
                  </span>
                  <span className="sr-only">changed to</span>
                  <span className="font-medium">{valueText(c.key, c.b)}</span>
                </dd>
              </div>
            ))}
          </dl>
        )}
      </section>

      <section className="space-y-2" aria-labelledby="compare-scores">
        <h3 id="compare-scores" className="text-sm font-medium">
          Scores
        </h3>
        <div className="@container overflow-hidden rounded-lg ring-1 ring-foreground/10">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="pl-3">Measure</TableHead>
                <TableHead>Before (A)</TableHead>
                <TableHead className="pr-3">After (B)</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {recallSets.map((sid) => (
                <TableRow key={sid}>
                  <TableCell className="pl-3 text-sm">{setName(sid)} recall@5</TableCell>
                  <TableCell className="tabular-nums">{recallText(data.summary.a.recall[sid])}</TableCell>
                  <TableCell className="pr-3 font-medium tabular-nums">{recallText(data.summary.b.recall[sid])}</TableCell>
                </TableRow>
              ))}
              <TableRow>
                <TableCell className="pl-3 text-sm">Scorecard</TableCell>
                <TableCell>
                  <OutcomeCounts outcomes={data.summary.a.outcomes} />
                </TableCell>
                <TableCell className="pr-3">
                  <OutcomeCounts outcomes={data.summary.b.outcomes} />
                </TableCell>
              </TableRow>
            </TableBody>
          </Table>
        </div>
      </section>

      <section className="space-y-2" aria-labelledby="compare-questions">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h3 id="compare-questions" className="text-sm font-medium">
            Question by question{" "}
            <span className="font-normal text-muted-foreground tabular-nums">
              · {data.changed} of {data.rows.length} changed
            </span>
          </h3>
          <label className="flex cursor-pointer items-center gap-2 text-sm">
            <Switch checked={changedOnly} onCheckedChange={setChangedOnly} aria-label="Show changed questions only" />
            Changed only
          </label>
        </div>
        {rows.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            {data.rows.length === 0
              ? "These runs have no questions in common. Compare runs over the same test questions."
              : "No question changed outcome or retrieval between these runs."}
          </p>
        ) : (
          <div className="@container -mx-4">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-12 pl-4">ID</TableHead>
                  <TableHead>Question</TableHead>
                  <TableHead className="hidden @xl:table-cell">Before (A)</TableHead>
                  <TableHead className="hidden @xl:table-cell">After (B)</TableHead>
                  <TableHead className="pr-4 text-right">Change</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {rows.map((r) => (
                  <CompareRowView key={`${r.set}:${r.id}`} r={r} retrieval={!noTarget.has(`${r.set}:${r.id}`)} />
                ))}
              </TableBody>
            </Table>
          </div>
        )}
      </section>
    </div>
  );
}

function isChanged(r: CompareRow): boolean {
  return r.outcome_changed || r.retrieval_changed;
}

const RANK: Record<Outcome, number> = { PASS: 2, REFUSED: 1, WRONG: 0 };

/** Better, worse or changed both ways, from the outcome first, then retrieval. */
function direction(r: CompareRow): "better" | "worse" | "changed" | null {
  if (!isChanged(r)) return null;
  const o = r.a.outcome && r.b.outcome ? Math.sign(RANK[r.b.outcome] - RANK[r.a.outcome]) : 0;
  const hitA = r.a.hit_rank !== null;
  const hitB = r.b.hit_rank !== null;
  const h = hitA === hitB ? 0 : hitB ? 1 : -1;
  if (o > 0 && h >= 0) return "better";
  if (o < 0 && h <= 0) return "worse";
  if (o === 0 && h > 0) return "better";
  if (o === 0 && h < 0) return "worse";
  return "changed";
}

const DIRECTION = {
  better: { label: "Better", icon: ArrowUp, className: "text-foreground" },
  worse: { label: "Worse", icon: ArrowDown, className: "text-red-700 dark:text-red-400" },
  changed: { label: "Changed", icon: ArrowRightLeft, className: "text-amber-700 dark:text-amber-400" },
} as const;

function CompareRowView({ r, retrieval }: { r: CompareRow; retrieval: boolean }) {
  const [open, setOpen] = useState(false);
  const dir = direction(r);
  const d = dir ? DIRECTION[dir] : null;
  return (
    <>
      <TableRow className={cn(dir && "bg-amber-500/[0.06]", "has-aria-expanded:bg-muted/40", open && "border-b-0")}>
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
          {/* Narrow cards: both sides under the question. */}
          <div className="mt-1 grid gap-1 pl-5 text-[11px] @xl:hidden">
            <SideSummary label="A" side={r.a} retrieval={retrieval} />
            <SideSummary label="B" side={r.b} retrieval={retrieval} />
          </div>
        </TableCell>
        <TableCell className="hidden align-top @xl:table-cell">
          <SideCell side={r.a} retrieval={retrieval} />
        </TableCell>
        <TableCell className="hidden align-top @xl:table-cell">
          <SideCell side={r.b} retrieval={retrieval} />
        </TableCell>
        <TableCell className="pr-4 text-right align-top">
          {d ? (
            <span className={cn("inline-flex items-center gap-1 text-xs font-medium", d.className)}>
              <d.icon className="size-3.5" aria-hidden />
              {d.label}
            </span>
          ) : (
            <span className="text-xs text-muted-foreground">Same</span>
          )}
        </TableCell>
      </TableRow>
      {open && (
        <TableRow className="bg-muted/40 hover:bg-muted/40">
          <TableCell aria-hidden className="p-0" />
          <TableCell colSpan={4} className="pt-2 pr-4 pb-5 whitespace-normal">
            <div className="grid gap-3 @2xl:grid-cols-2">
              <AnswerPanel title="Before (A)" side={r.a} />
              <AnswerPanel title="After (B)" side={r.b} />
            </div>
          </TableCell>
        </TableRow>
      )}
    </>
  );
}

function hitText(side: CompareSide, retrieval: boolean): React.ReactNode {
  if (!retrieval) return <span className="text-muted-foreground">{NO_DATA}</span>;
  if (side.hit_rank !== null) return <span className="font-medium tabular-nums">Hit@{side.hit_rank}</span>;
  return (
    <span className="tabular-nums">
      <span className="font-medium text-red-700 dark:text-red-400">Miss</span>
      <span className="text-muted-foreground">{side.best_rank !== null ? ` (rank ${side.best_rank})` : " (not in top 50)"}</span>
    </span>
  );
}

function SideCell({ side, retrieval }: { side: CompareSide; retrieval: boolean }) {
  return (
    <div className="space-y-1">
      <OutcomeBadge outcome={side.outcome} error={side.error} />
      <p className="text-xs">{hitText(side, retrieval)}</p>
    </div>
  );
}

function SideSummary({ label, side, retrieval }: { label: string; side: CompareSide; retrieval: boolean }) {
  return (
    <div className="flex flex-wrap items-center gap-2">
      <span className="w-3 font-medium text-muted-foreground">{label}</span>
      <OutcomeBadge outcome={side.outcome} error={side.error} />
      {hitText(side, retrieval)}
    </div>
  );
}

function AnswerPanel({ title, side }: { title: string; side: CompareSide }) {
  return (
    <section className="min-w-0 space-y-2 rounded-lg bg-card p-3 ring-1 ring-foreground/10" aria-label={title}>
      <h4 className="text-[11px] font-medium tracking-wide text-muted-foreground uppercase">{title}</h4>
      {side.answer !== null ? (
        <p className="text-sm leading-relaxed text-pretty whitespace-pre-wrap">{side.answer.trim()}</p>
      ) : (
        <p className="text-sm text-muted-foreground">No answer was recorded.</p>
      )}
      {side.missing.length > 0 && (
        <p className="text-[11px] text-red-700 dark:text-red-400">Missing: {side.missing.join(", ")}</p>
      )}
      {side.error && <p className="text-[11px] text-amber-700 dark:text-amber-400">{side.error}</p>}
    </section>
  );
}

// --------------------------------------------------------------------------- setup keys

/** Keys that always differ between two index profiles and add nothing to the rows that explain it. */
const HIDDEN_KEYS = new Set(["index.config.id", "index.config.label", "index.config.description"]);

const KEY_LABELS: Record<string, string> = {
  mode: "What was measured",
  k: "Passages per answer",
  ollama_version: "Ollama",
  "index.profile": "Index profile",
  "index.label": "Index",
  "index.rows": "Passages",
  "index.config.parser": "Parser",
  "index.config.parser_version": "Parser version",
  "index.config.cleaner": "Cleaner",
  "index.config.contextual": "Context lines",
  "index.config.metric": "Metric",
  "index.embedding.model": "Embedding model",
  "index.embedding.digest": "Embedding digest",
  "index.embedding.quantization": "Embedding quantization",
  "index.embedding.dimensions": "Dimensions",
  "index.embedding.doc_prefix": "Passage prefix",
  "index.embedding.query_prefix": "Question prefix",
  "retrieval.method": "Search",
  "retrieval.section_cap": "Limit per section",
  "retrieval.fusion": "Fusion",
  "retrieval.rrf_k": "RRF k",
  "retrieval.depth": "Passages per list",
  "retrieval.metric": "Metric",
  "retrieval.exact": "Exact search",
  "retrieval.word_index.column": "Word index column",
  "index.config.chunker.method": "Chunker",
  "index.config.chunker.labels": "Section label format",
  "answer.model": "Answer model",
  "answer.digest": "Answer model digest",
  "answer.quantization": "Answer quantization",
  "answer.parameter_size": "Answer model size",
  "answer.temperature": "Temperature",
  "answer.num_ctx": "Context window",
  "answer.think": "Thinking",
  "answer.prompt_version": "Prompt",
  "answer.prompt_sha256": "Prompt sha256",
};

function keyLabel(key: string): string {
  if (KEY_LABELS[key]) return KEY_LABELS[key];
  if (key.startsWith("index.config.chunker.")) return `Chunker ${key.slice("index.config.chunker.".length).replace(/_/g, " ")}`;
  return key;
}

function valueText(key: string, v: unknown): string {
  if (v === null || v === undefined || v === "") return NO_DATA;
  if (typeof v === "string" && (key.endsWith("digest") || key.endsWith("sha256"))) return v.replace(/^sha256:/, "").slice(0, 12);
  if (typeof v === "boolean") return v ? "on" : "off";
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}

