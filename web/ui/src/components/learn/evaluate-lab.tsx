"use client";

import { useState } from "react";
import useSWR from "swr";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { CardSkeleton, ErrorState } from "@/components/common/states";
import { TechnicalDetails } from "@/components/common/technical-details";
import { HitLabel, OUTCOME, OutcomeBadge } from "@/components/eval/badges";
import { AnswerTry } from "@/components/learn/answer-try";
import { MarkedAnswer } from "@/components/learn/marked-answer";
import type { EvaluateLab as EvaluateLabData, EvaluateQuestion, Outcome, PassInterval, RunBrief } from "@/lib/api";
import { NO_DATA } from "@/lib/format";
import { cn } from "@/lib/utils";

const OUTCOMES: Outcome[] = ["PASS", "REFUSED", "WRONG"];
const GRID_ROWS: { key: "hit" | "miss" | "no_pages"; label: string; explain: string }[] = [
  { key: "hit", label: "Hit", explain: "An answer page was in the top passages the model read." },
  { key: "miss", label: "Miss", explain: "No answer page reached the model: it could only refuse or guess." },
  { key: "no_pages", label: "Trap", explain: "No page answers it. The right answer is the refusal." },
];
const CELL_MEANING: Record<string, string> = {
  "hit-PASS": "Found and answered: the pipeline working.",
  "hit-REFUSED": "The page was there and the model still refused: the answer step's fault.",
  "hit-WRONG": "The page was there and the answer was wrong: read wrong, or a fact missing.",
  "miss-PASS": "Right without the answer page: another passage happened to state it.",
  "miss-REFUSED": "Not found, and it said so: search's fault, safely handled.",
  "miss-WRONG": "Not found, and it answered anyway: the dangerous corner.",
  "no_pages-PASS": "Refused, as it should.",
  "no_pages-REFUSED": "",
  "no_pages-WRONG": "Answered a question the documents do not answer.",
};

function runName(r: Pick<RunBrief, "id" | "label" | "ladder">): string {
  return r.ladder ? `Row ${r.ladder.row}: ${r.ladder.label}` : (r.label ?? r.id);
}

function runOption(r: RunBrief): string {
  const o = r.summary?.outcomes;
  const score = o ? ` · ${o.PASS}·${o.REFUSED}·${o.WRONG}` : "";
  return `${runName(r)} · ${r.config.answer_model ?? "?"}${score}`;
}

function pct(x: number | null): string {
  return x === null ? NO_DATA : `${Math.round(x * 100)}%`;
}

/**
 * Step 12: evaluate. One recorded run's scorecard taken apart: the Hit/Miss
 * × outcome grid, one question's facts and where they matched, the reciprocal
 * ranks behind MRR, Set E's chain, and what 25 questions can and cannot show.
 */
export function EvaluateLab() {
  const runsList = useSWR<{ runs: RunBrief[] }>("/eval/runs");
  const runs = (runsList.data?.runs ?? []).filter((r) => r.status === "done" && r.spec.mode === "answer");
  const [picked, setPicked] = useState<string | null>(null);
  const [against, setAgainst] = useState<string>("none");
  const fallback = runs.find((r) => r.ladder && r.spec.sets.includes("set-a")) ?? runs[0];
  const runId = picked ?? fallback?.id ?? null;
  const key = runId
    ? `/learn/evaluate?${new URLSearchParams({ run: runId, ...(against !== "none" && against !== runId ? { against } : {}) })}`
    : null;
  const { data, error, isLoading, mutate } = useSWR<EvaluateLabData>(key, { keepPreviousData: true });
  const shown = data && data.run.id === runId ? data : null;
  const [qid, setQid] = useState<string | null>(null);

  if (!runsList.data) {
    return runsList.error ? (
      <ErrorState message={runsList.error.message} error={runsList.error} onRetry={() => runsList.mutate()} />
    ) : (
      <CardSkeleton height="h-96" />
    );
  }

  const options = Object.fromEntries(runs.map((r) => [r.id, runOption(r)]));
  const question = shown?.questions.find((q) => q.id === qid) ??
    shown?.questions.find((q) => q.scored && q.outcome !== "PASS") ??
    shown?.questions.find((q) => q.scored);

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle as="h2">A recorded run</CardTitle>
          <CardDescription>
            A run is every test question asked with one setup. Pick one to take its scorecard apart, and another to
            compare it with. Read from the saved runs: nothing is asked of a model.
          </CardDescription>
        </CardHeader>
        <CardContent className="grid gap-3 @container sm:grid-cols-2">
          <div className="min-w-0 space-y-1">
            <label htmlFor="eval-run" className="text-xs font-medium text-muted-foreground">
              Run
            </label>
            <Select value={runId ?? ""} items={options} onValueChange={(v) => v && (setPicked(v as string), setQid(null))}>
              <SelectTrigger id="eval-run" className="w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {runs.map((r) => (
                  <SelectItem key={r.id} value={r.id}>
                    {runOption(r)}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div className="min-w-0 space-y-1">
            <label htmlFor="eval-against" className="text-xs font-medium text-muted-foreground">
              Compare with (the run before)
            </label>
            <Select
              value={against}
              items={{ none: "No comparison", ...options }}
              onValueChange={(v) => v && setAgainst(v as string)}
            >
              <SelectTrigger id="eval-against" className="w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="none">No comparison</SelectItem>
                {runs
                  .filter((r) => r.id !== runId)
                  .map((r) => (
                    <SelectItem key={r.id} value={r.id}>
                      {runOption(r)}
                    </SelectItem>
                  ))}
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      {runs.length === 0 ? (
        <p className="text-sm text-muted-foreground">No finished run has answers yet. Run an evaluation first.</p>
      ) : error ? (
        <ErrorState message={error.message} error={error} onRetry={() => mutate()} />
      ) : !shown ? (
        <CardSkeleton height="h-96" />
      ) : (
        <div className={cn("space-y-4", isLoading && "opacity-70")}>
          <Grid data={shown} selected={question?.id ?? null} onSelect={setQid} />
          {question && <Question data={shown} q={question} />}
          <Ranking data={shown} />
          {shown.summary?.followups && <Followups data={shown} />}
          <HowSure data={shown} />
        </div>
      )}
    </div>
  );
}

function Grid({ data, selected, onSelect }: { data: EvaluateLabData; selected: string | null; onSelect: (id: string) => void }) {
  const scored = data.questions.filter((q) => q.scored).length;
  return (
    <Card className="@container">
      <CardHeader>
        <CardTitle as="h2" className="text-base">
          The scorecard
        </CardTitle>
        <CardDescription>
          Each answer gets one outcome: Pass (every expected fact is there, or a trap was refused), Refused (a safe
          failure) or Wrong (the dangerous one). Beside it, retrieval: did an answer page reach the model? Together
          they say which step failed. {scored} scored questions; pick one to see how it was scored.
        </CardDescription>
      </CardHeader>
      <CardContent className="overflow-x-auto">
        <table className="w-full min-w-[34rem] border-separate border-spacing-1 text-xs">
          <thead>
            <tr>
              <th className="w-28">
                <span className="sr-only">Retrieval</span>
              </th>
              {OUTCOMES.map((o) => (
                <th key={o} className="pb-1 text-left font-medium">
                  <OutcomeBadge outcome={o} />
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {GRID_ROWS.map((row) => (
              <tr key={row.key}>
                <th scope="row" className="pr-2 text-left align-top font-medium">
                  {row.label}
                  <span className="block font-normal text-pretty text-muted-foreground">{row.explain}</span>
                </th>
                {OUTCOMES.map((o) => {
                  const ids = data.grid[row.key][o];
                  const meaning = CELL_MEANING[`${row.key}-${o}`];
                  return (
                    <td
                      key={o}
                      className={cn(
                        "rounded-md p-2 align-top ring-1 ring-foreground/10",
                        ids.length && o === "WRONG" && "bg-red-500/5",
                        ids.length && o === "PASS" && "bg-emerald-500/5",
                      )}
                    >
                      <p className="text-base font-semibold tabular-nums">{ids.length}</p>
                      {meaning && <p className="mb-1 text-[11px] text-pretty text-muted-foreground">{meaning}</p>}
                      <div className="flex flex-wrap gap-1">
                        {ids.map((id) => (
                          <button
                            key={id}
                            type="button"
                            onClick={() => onSelect(id)}
                            aria-pressed={selected === id}
                            className={cn(
                              "rounded px-1.5 py-0.5 font-mono text-[11px] ring-1 ring-foreground/15 hover:bg-muted focus-visible:ring-[3px] focus-visible:ring-ring/50 focus-visible:outline-none",
                              selected === id && "bg-foreground text-background hover:bg-foreground",
                            )}
                          >
                            {id}
                          </button>
                        ))}
                      </div>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </CardContent>
    </Card>
  );
}

function Question({ data, q }: { data: EvaluateLabData; q: EvaluateQuestion }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle as="h2" className="text-base">
          <span className="font-mono">{q.id}</span>, scored
        </CardTitle>
        <CardDescription className="text-pretty">{q.question}</CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        {q.history && q.history.length > 0 && (
          <p className="text-xs text-pretty text-muted-foreground">
            A follow-up to: <span className="text-foreground">{q.history.at(-1)!.question}</span>
            {q.standalone_question ? (
              <>
                {" "}
                · searched for: <span className="text-foreground">{q.standalone_question}</span>
                {q.rewrite_ok === false ? " (rewritten wrong)" : q.rewrite_ok ? " (rewritten right)" : ""}
              </>
            ) : (
              " · searched as typed"
            )}
          </p>
        )}
        <dl className="grid gap-x-4 gap-y-1.5 text-xs sm:grid-cols-[auto_1fr]">
          <dt className="text-muted-foreground">Expected</dt>
          <dd>
            {q.trap ? (
              "The exact refusal (a trap)."
            ) : q.facts.length ? (
              q.facts.map((f) => (
                <span key={f.label} className={cn("mr-3", f.found ? "text-emerald-700 dark:text-emerald-400" : "text-red-700 dark:text-red-400")}>
                  {f.found ? "✓" : "✗"} {f.label}
                </span>
              ))
            ) : (
              NO_DATA
            )}
          </dd>
          <dt className="text-muted-foreground">Outcome</dt>
          <dd>
            <OutcomeBadge outcome={q.outcome} error={q.error} />
          </dd>
          <dt className="text-muted-foreground">Retrieval</dt>
          <dd>
            <HitLabel result={q} />
            {q.reciprocal !== null && (
              <span className="text-muted-foreground tabular-nums">
                {" "}
                · reciprocal rank {q.reciprocal ? q.reciprocal.toFixed(3) : "0 (not in the top 50)"}
              </span>
            )}
          </dd>
        </dl>
        {q.answer && (
          <div className="rounded-lg bg-muted/40 px-3 py-2.5">
            <MarkedAnswer text={q.answer} facts={q.facts} />
          </div>
        )}
        <p className="text-xs text-pretty text-muted-foreground">
          Facts are matched as text: a substring, one of several spellings, or a number pattern so that 45% does not
          count as 5%. That checks a fact is present, not that every other claim in the answer is supported. The
          faithfulness judge that would check that was cut from this version.
        </p>
        {q.answer && (
          <TechnicalDetails label="Edit this answer and score it again">
            <AnswerTry
              key={`${data.run.id}-${q.id}`}
              runId={data.run.id}
              questionId={q.id}
              recorded={q.answer}
              k={data.k ?? 5}
              refusal="I could not find this in the documents."
              show="score"
            />
          </TechnicalDetails>
        )}
      </CardContent>
    </Card>
  );
}

function Ranking({ data }: { data: EvaluateLabData }) {
  const rows = data.questions.filter((q) => q.reciprocal !== null);
  const r = data.summary?.ranking;
  const sum = rows.reduce((s, q) => s + (q.reciprocal ?? 0), 0);
  return (
    <Card>
      <CardHeader>
        <CardTitle as="h2" className="text-base">
          Ranking beyond the top {data.k ?? 5}
        </CardTitle>
        <CardDescription>
          Hit@{data.k ?? 5} only asks whether an answer page made the cut. Every run also searches {data.deep_k ?? 50}{" "}
          deep, so it knows where the first answer page really was. MRR averages 1 / that rank: 1.0 means every answer
          page was first, 0.5 that it was second on average. Recall@20 counts the questions whose page is in the top 20,
          the pool a reranker reads.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="flex h-24 items-end gap-0.5" role="img" aria-label="Reciprocal rank per question">
          {rows.map((q) => (
            <div key={`${q.set}-${q.id}`} className="flex h-full min-w-0 flex-1 flex-col justify-end" title={`${q.id}: rank ${q.best_rank ?? "not in the top 50"}`}>
              <div className={cn("w-full rounded-t-sm", q.reciprocal ? "bg-sky-600 dark:bg-sky-500" : "bg-red-500/60")} style={{ height: `${Math.max(4, (q.reciprocal ?? 0) * 100)}%` }} />
            </div>
          ))}
        </div>
        <p className="text-xs text-muted-foreground">
          One bar per question with an answer page, its height 1 / rank (red: not in the top 50, counted as 0).
        </p>
        <dl className="grid gap-x-4 gap-y-1 text-xs sm:grid-cols-[auto_1fr]">
          <dt className="text-muted-foreground">MRR</dt>
          <dd className="tabular-nums">
            {r?.mrr ?? NO_DATA}
            <span className="text-muted-foreground">
              {" "}
              = {sum.toFixed(2)} / {rows.length} questions
            </span>
          </dd>
          <dt className="text-muted-foreground">Recall@20</dt>
          <dd className="tabular-nums">
            {r?.recall20 ?? NO_DATA} of {r?.n ?? NO_DATA}
          </dd>
          {r && r.multi.n > 0 && (
            <>
              <dt className="text-muted-foreground">Two documents</dt>
              <dd className="tabular-nums">
                MRR {r.multi.mrr ?? NO_DATA} over {r.multi.n}: counted only once both pages are in
              </dd>
            </>
          )}
        </dl>
      </CardContent>
    </Card>
  );
}

function Followups({ data }: { data: EvaluateLabData }) {
  const f = data.summary!.followups!;
  const steps = [
    { label: "Follow-up questions", n: f.n },
    { label: "Rewritten right", n: f.rewritten ? f.rewrite_ok : null },
    { label: "Found (a Hit)", n: f.hits },
    { label: "Answered right (a Pass)", n: f.passes },
  ];
  return (
    <Card>
      <CardHeader>
        <CardTitle as="h2" className="text-base">
          Follow-ups, judged in order
        </CardTitle>
        <CardDescription>
          A follow-up like “And for 2027?” is judged as a chain: first whether Remote AI rewrote it into the right
          question, then whether that found the page, then the answer. A failure early explains everything after it.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <ol className="grid gap-2 sm:grid-cols-4">
          {steps.map((s) => (
            <li key={s.label} className="rounded-md px-3 py-2 ring-1 ring-foreground/10">
              <p className="text-lg font-semibold tabular-nums">{s.n ?? NO_DATA}</p>
              <p className="text-xs text-muted-foreground">{s.label}</p>
            </li>
          ))}
        </ol>
        {!f.rewritten && <p className="mt-2 text-xs text-muted-foreground">This run asked them as typed: nothing was rewritten.</p>}
      </CardContent>
    </Card>
  );
}

function IntervalBar({ label, iv }: { label: string; iv: PassInterval }) {
  if (iv.rate === null || iv.low === null || iv.high === null) return null;
  return (
    <div className="space-y-1">
      <p className="text-xs">
        <span className="font-medium">{label}</span>
        <span className="text-muted-foreground tabular-nums">
          {" "}
          · {iv.passes} of {iv.n} passed ({pct(iv.rate)}); the true pass rate could be {pct(iv.low)} to {pct(iv.high)}
        </span>
      </p>
      <div className="relative h-4 rounded-full bg-muted" role="img" aria-label={`${pct(iv.low)} to ${pct(iv.high)}`}>
        <div className="absolute inset-y-0 rounded-full bg-sky-600/30 dark:bg-sky-500/30" style={{ left: `${iv.low * 100}%`, width: `${(iv.high - iv.low) * 100}%` }} />
        <div className="absolute inset-y-0 w-1 -translate-x-1/2 rounded-full bg-sky-700 dark:bg-sky-400" style={{ left: `${iv.rate * 100}%` }} />
      </div>
    </div>
  );
}

function HowSure({ data }: { data: EvaluateLabData }) {
  const a = data.against;
  return (
    <Card>
      <CardHeader>
        <CardTitle as="h2" className="text-base">
          A hint, not proof
        </CardTitle>
        <CardDescription>
          At temperature 0 the same setup gives the same answers: two runs of the baseline on 2026-10-14 wrote D3 word
          for word the same. The uncertainty is not the model’s dice, it is the questions: {data.interval.n} is few, and
          a different {data.interval.n} would score differently. The bar is the range of pass rates this run is
          consistent with (a Wilson 95% interval).
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <IntervalBar label={runName(data.run)} iv={data.interval} />
        {a ? (
          <>
            <IntervalBar label={`${runName(a.run)} (before)`} iv={a.interval} />
            <dl className="grid gap-x-4 gap-y-1 text-xs sm:grid-cols-[auto_1fr]">
              <dt className="text-muted-foreground">Became a Pass</dt>
              <dd className="font-mono">{a.better.join(", ") || NO_DATA}</dd>
              <dt className="text-muted-foreground">Stopped being one</dt>
              <dd className="font-mono">{a.worse.join(", ") || NO_DATA}</dd>
              {a.other_changes.length > 0 && (
                <>
                  <dt className="text-muted-foreground">Other changes</dt>
                  <dd className="font-mono">{a.other_changes.join(", ")} (Refused and Wrong swapped)</dd>
                </>
              )}
              <dt className="text-muted-foreground">By chance?</dt>
              <dd className="text-pretty">
                {a.better.length + a.worse.length === 0
                  ? "No question changed between Pass and not, so the two runs cannot be told apart on these questions."
                  : `${a.better.length} better and ${a.worse.length} worse, of ${a.n} questions. If neither setup were better, a split at least this lopsided would happen ${pct(a.p)} of the time (the exact sign test on the questions that changed).`}{" "}
                {a.p >= 0.05 && a.better.length + a.worse.length > 0 && "Not enough to call it an improvement: a hint to measure more, not proof."}
              </dd>
            </dl>
            {a.config.length > 0 && (
              <TechnicalDetails label={`What differs between the two setups (${a.config.length})`}>
                <ul className="space-y-0.5 font-mono text-[10.5px]">
                  {a.config.map((c) => (
                    <li key={c.key} className="break-all">
                      {c.key}: {String(c.a ?? "-")} → {String(c.b ?? "-")}
                    </li>
                  ))}
                </ul>
              </TechnicalDetails>
            )}
          </>
        ) : (
          <p className="text-xs text-muted-foreground">
            Pick a run to compare with to see which questions changed, and whether that many could change by chance.
          </p>
        )}
        <p className="text-xs text-pretty text-muted-foreground">
          {OUTCOME.PASS.label} counts here; Refused and Wrong both count as not passing. That is why the ladder reads
          every row question by question, not as a total.
        </p>
      </CardContent>
    </Card>
  );
}
