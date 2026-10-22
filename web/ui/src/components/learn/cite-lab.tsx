"use client";

import useSWR from "swr";
import { ArrowRight, CircleSlash } from "lucide-react";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { CardSkeleton, ErrorState } from "@/components/common/states";
import { OutcomeBadge } from "@/components/eval/badges";
import { AnswerTry } from "@/components/learn/answer-try";
import { MarkedAnswer } from "@/components/learn/marked-answer";
import { useRecordedAnswer, type Preset } from "@/components/learn/recorded-picker";
import type { CiteLab as CiteLabData, DocumentFile, RecordedSource } from "@/lib/api";
import { docTitle, pageImageUrl } from "@/lib/documents";
import { NO_DATA } from "@/lib/format";
import { cn } from "@/lib/utils";

const PRESETS: Preset[] = [
  { set: "set-a", id: "A2", why: "Two facts in one answer (what replaced FDsys, and when), each with its own citation." },
  { set: "set-d", id: "D10", why: "Its answer needs two documents, so the citations must point into both." },
  { set: "set-a", id: "T1", why: "A trap: the documents do not say. Only the exact refusal passes." },
  { set: "set-d", id: "T4", why: "A trap where a model refused and then kept talking: the strict rule counts that as an answer." },
];

function seconds(ms: number | null | undefined): string {
  return ms === null || ms === undefined ? NO_DATA : `${(ms / 1000).toFixed(1)} s`;
}

/**
 * Step 10: generate and cite. One recorded answer with every [n] it wrote and
 * the page code mapped it to, the strict refusal rule, an editable copy read
 * by the same code, how the model wrote it, and the same passages answered by
 * the other answer models (ladder row 10).
 */
export function CiteLab() {
  const { data: docData } = useSWR<{ documents: DocumentFile[] }>("/documents");
  const docs = docData?.documents;
  const { qid, runId, picker, listError, retryList } = useRecordedAnswer(PRESETS, "cite");
  const key = runId ? `/learn/cite?${new URLSearchParams({ run: runId, question: qid })}` : null;
  const { data, error, isLoading, mutate } = useSWR<CiteLabData>(key, { keepPreviousData: true });
  const shown = data && data.run.id === runId && data.question_id === qid ? data : null;

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle as="h2">A recorded answer</CardTitle>
          <CardDescription>
            Pick a test question and a run that answered it. Everything below is read from that run, with its passages
            read back from the index build it searched. Nothing is asked of a model.
          </CardDescription>
        </CardHeader>
        <CardContent>{picker}</CardContent>
      </Card>

      {listError ? (
        <ErrorState message={listError.message} error={listError} onRetry={retryList} />
      ) : error ? (
        <ErrorState message={error.message} error={error} onRetry={() => mutate()} />
      ) : !runId ? null : !shown ? (
        <CardSkeleton height="h-96" />
      ) : (
        <div className={cn("space-y-4", isLoading && "opacity-70")}>
          <Mapped data={shown} docs={docs} />
          <Refusal data={shown} />
          <Card>
            <CardHeader>
              <CardTitle as="h2" className="text-base">
                Change the answer
              </CardTitle>
              <CardDescription>
                Edit it, or use a button, and the same citation and refusal code reads it again, against the{" "}
                {shown.sources.length} passages this run showed.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <AnswerTry
                key={`${shown.run.id}-${shown.question_id}`}
                runId={shown.run.id}
                questionId={shown.question_id}
                recorded={shown.answer}
                k={shown.sources.length}
                refusal={shown.refusal}
                show="citations"
              />
            </CardContent>
          </Card>
          <Written data={shown} />
          <Models data={shown} />
        </div>
      )}
    </div>
  );
}

function SourceRow({ n, s, docs }: { n: number; s: RecordedSource | undefined; docs: DocumentFile[] | undefined }) {
  if (!s) {
    return (
      <li className="flex items-baseline gap-2 rounded-md bg-red-500/10 px-3 py-2 text-xs text-red-800 ring-1 ring-red-500/30 dark:text-red-300">
        <span className="font-mono font-semibold">[{n}]</span>
        No passage {n} was shown to the model. The citation is reported as invalid: never dropped, never “fixed” to a
        nearby number.
      </li>
    );
  }
  return (
    <li className="flex gap-3 rounded-md px-3 py-2 ring-1 ring-foreground/10">
      {/* eslint-disable-next-line @next/next/no-img-element -- a PNG the API renders on demand */}
      <img
        src={pageImageUrl(s.doc_id, s.page, 48)}
        alt={`Page ${s.page} of ${docTitle(s.doc_id, docs)}`}
        loading="lazy"
        className="hidden h-24 w-auto shrink-0 rounded-sm bg-white ring-1 ring-foreground/10 @md:block"
      />
      <div className="min-w-0 space-y-1 text-xs">
        <p className="flex flex-wrap items-center gap-x-2 gap-y-0.5">
          <span className="font-mono font-semibold text-sky-800 dark:text-sky-300">[{n}]</span>
          <ArrowRight className="size-3 text-muted-foreground" aria-hidden />
          <span className="font-mono text-[11px] text-muted-foreground">{s.chunk_id}</span>
          <ArrowRight className="size-3 text-muted-foreground" aria-hidden />
          <span className="font-medium">
            {docTitle(s.doc_id, docs)}, page {s.page}
          </span>
          {s.page_label && s.page_label !== String(s.page) && (
            <span className="text-muted-foreground">(printed {s.page_label})</span>
          )}
        </p>
        {s.section_path && <p className="text-muted-foreground">{s.section_path}</p>}
        <p className="line-clamp-3 text-pretty text-muted-foreground">{s.text}</p>
      </div>
    </li>
  );
}

function Mapped({ data, docs }: { data: CiteLabData; docs: DocumentFile[] | undefined }) {
  const numbers = [...new Set(data.citations.numbers)];
  return (
    <Card className="@container">
      <CardHeader>
        <CardTitle as="h2" className="text-base">
          From [n] to a page
        </CardTitle>
        <CardDescription>
          The model only writes numbers. Code finds each [n], looks it up in the list of passages that answer was
          written from, and that passage carries its document and page. A number with no passage behind it stays
          visible as an invalid citation.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="rounded-lg bg-muted/40 px-3 py-2.5">
          <MarkedAnswer text={data.answer} marks={data.marks} />
        </div>
        {numbers.length === 0 ? (
          <p className="text-xs text-muted-foreground">This answer cites nothing, so there is nothing to map.</p>
        ) : (
          <ol className="space-y-2">
            {numbers.map((n) => (
              <SourceRow key={n} n={n} s={data.sources.find((s) => s.n === n)} docs={docs} />
            ))}
          </ol>
        )}
        <p className="text-xs text-muted-foreground">
          {data.matches_recorded
            ? "Mapped again now from the recorded text: the same result the run saved, because the mapping is code, not a model."
            : "Mapped again now from the recorded text: the result differs from the one the run saved, so the citation code changed since."}
        </p>
      </CardContent>
    </Card>
  );
}

function Refusal({ data }: { data: CiteLabData }) {
  const containsRefusal = data.answer.includes(data.refusal);
  return (
    <Card>
      <CardHeader>
        <CardTitle as="h2" className="text-base">
          The refusal
        </CardTitle>
        <CardDescription>
          When no passage states the answer, the rules ask for exactly one sentence and nothing else. The check is
          strict on purpose: a wrong fact followed by the refusal sentence is still a wrong fact.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3 text-sm">
        <p className="flex items-center gap-2 rounded-md bg-muted/40 px-3 py-2 text-pretty">
          <CircleSlash className="size-4 shrink-0 text-muted-foreground" aria-hidden />
          {data.refusal}
        </p>
        <dl className="grid gap-x-4 gap-y-1 text-xs sm:grid-cols-[auto_1fr]">
          <dt className="text-muted-foreground">This question</dt>
          <dd>
            {data.trap
              ? "A trap: the documents do not answer it, so only the exact refusal passes."
              : "Answerable: a refusal here is a safe failure (Refused), not a Pass."}
          </dd>
          <dt className="text-muted-foreground">The recorded answer</dt>
          <dd>
            {data.refused
              ? "Is exactly the refusal."
              : containsRefusal
                ? "Contains the refusal sentence but goes on, so it counts as an answer."
                : "Is an answer, not a refusal."}
          </dd>
          <dt className="text-muted-foreground">Scorecard</dt>
          <dd>
            <OutcomeBadge outcome={data.outcome} />
          </dd>
        </dl>
        {data.key_changed && (
          <p className="text-xs text-amber-700 dark:text-amber-400">
            The answer key for {data.question_id} has changed since this run scored it; a new run would use the current
            key.
          </p>
        )}
      </CardContent>
    </Card>
  );
}

function Written({ data }: { data: CiteLabData }) {
  const u = data.usage;
  const rate = u?.answer_tokens && u.answer_ms ? (u.answer_tokens / (u.answer_ms / 1000)).toFixed(1) : null;
  const readRate = u?.prompt_tokens && u.prompt_ms ? Math.round(u.prompt_tokens / (u.prompt_ms / 1000)) : null;
  return (
    <Card>
      <CardHeader>
        <CardTitle as="h2" className="text-base">
          How the model wrote it
        </CardTitle>
        <CardDescription>
          The answer model reads the whole prompt first, then writes one token at a time. Thinking is switched off and
          the temperature is 0, so the same prompt gives the same answer: a second run is not a second opinion.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <dl className="grid gap-x-4 gap-y-1 text-xs sm:grid-cols-[auto_1fr]">
          <dt className="text-muted-foreground">Model</dt>
          <dd>
            {data.model.model ?? NO_DATA}
            <span className="text-muted-foreground">
              {" "}
              · {data.model.parameter_size ?? NO_DATA}, {data.model.quantization ?? NO_DATA}, on this computer
            </span>
          </dd>
          <dt className="text-muted-foreground">Settings</dt>
          <dd>
            thinking {data.model.think === false ? "off" : NO_DATA} · temperature {data.model.temperature ?? NO_DATA} ·
            context window {data.model.num_ctx?.toLocaleString() ?? NO_DATA} tokens
          </dd>
          <dt className="text-muted-foreground">Reading the prompt</dt>
          <dd className="tabular-nums">
            {u?.prompt_tokens?.toLocaleString() ?? NO_DATA} tokens in {seconds(u?.prompt_ms)}
            {readRate ? ` (${readRate} tokens/s)` : ""}
          </dd>
          <dt className="text-muted-foreground">First word</dt>
          <dd className="tabular-nums">after {seconds(u?.first_token_ms)}</dd>
          <dt className="text-muted-foreground">Writing</dt>
          <dd className="tabular-nums">
            {u?.answer_tokens ?? NO_DATA} tokens in {seconds(u?.answer_ms)}
            {rate ? ` (${rate} tokens/s)` : ""}
          </dd>
          <dt className="text-muted-foreground">In all</dt>
          <dd className="tabular-nums">{seconds(data.timings?.generate_ms)}</dd>
        </dl>
      </CardContent>
    </Card>
  );
}

function Models({ data }: { data: CiteLabData }) {
  return (
    <Card className="@container">
      <CardHeader>
        <CardTitle as="h2" className="text-base">
          The same passages, other answer models
        </CardTitle>
        <CardDescription>
          Runs that searched exactly as this one did and changed only the answer model (ladder row 10). The passages and
          the prompt are identical, so any difference is the model’s.
        </CardDescription>
      </CardHeader>
      <CardContent>
        {data.models.length <= 1 ? (
          <p className="text-xs text-muted-foreground">
            Only {data.model.model ?? "this model"} was measured with these passages. Pick a ladder row 6 or 10 run to
            compare models.
          </p>
        ) : (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Model</TableHead>
                <TableHead>Outcome</TableHead>
                <TableHead className="hidden text-right @md:table-cell">Time</TableHead>
                <TableHead className="hidden @xl:table-cell">Answer</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {data.models.map((m) => (
                <TableRow key={m.value}>
                  <TableCell className="font-medium whitespace-nowrap">
                    {m.value}
                    {m.this_run && <span className="block text-[10px] font-normal text-muted-foreground">this run</span>}
                  </TableCell>
                  <TableCell>
                    <OutcomeBadge outcome={m.outcome} />
                  </TableCell>
                  <TableCell className="hidden text-right tabular-nums @md:table-cell">{seconds(m.generate_ms)}</TableCell>
                  <TableCell className="hidden max-w-[28rem] text-xs text-pretty whitespace-normal @xl:table-cell">
                    <span className="line-clamp-3">{m.answer}</span>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </CardContent>
    </Card>
  );
}
