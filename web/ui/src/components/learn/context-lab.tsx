"use client";

import { useState } from "react";
import useSWR from "swr";
import { Loader2, Target } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { CardSkeleton, ErrorState } from "@/components/common/states";
import { TechnicalDetails } from "@/components/common/technical-details";
import { Tile, TileGrid } from "@/components/common/tiles";
import type { ContextLab as ContextLabData, ContextPassage, ContextRankSide, DocumentFile, GoldenSet } from "@/lib/api";
import { docTitle } from "@/lib/documents";
import { NO_DATA, num } from "@/lib/format";
import { cn } from "@/lib/utils";

/** Test questions where a passage alone does not say what the question asks for, by id. */
const PRESETS: { set: string; id: string; why: string }[] = [
  { set: "set-b", id: "B4", why: "On the Docling index the answer ranks 8th: passages on U.S. and euro-area growth come first." },
  { set: "set-a", id: "A6", why: "The risks paragraph never says \"executive summary\"; a context line can." },
  { set: "set-d", id: "D10", why: "With EmbeddingGemma 2 its page drops out of the top 5 (rank 13 to over 50)." },
];

/**
 * Step 4: contextual retrieval. A passage, the line a model wrote to place it
 * in its document, and exactly what is embedded with and without that line;
 * for a test question, where its answer ranks on the index without context
 * lines and on the one with them. Everything is read from lines already
 * written: looking never asks a model for one.
 */
export function ContextLab() {
  const { data: golden } = useSWR<{ sets: GoldenSet[] }>("/golden");
  const { data: docData } = useSWR<{ documents: DocumentFile[] }>("/documents");
  const docs = docData?.documents;
  // null: the API's default index with context lines (it names it in the answer).
  const [profile, setProfile] = useState<string | null>(null);
  const [question, setQuestion] = useState<string | null>(PRESETS[0].id);
  const [chunk, setChunk] = useState<string | null>(null);

  const params = new URLSearchParams();
  if (profile) params.set("profile", profile);
  if (question) params.set("question", question);
  if (chunk) params.set("chunk", chunk);
  const query = params.toString();
  const key = `/learn/context${query ? `?${query}` : ""}`;
  const { data, error, isLoading, mutate } = useSWR<ContextLabData>(key, { keepPreviousData: true });

  const presets = PRESETS.flatMap((p) => {
    const q = golden?.sets.find((s) => s.id === p.set)?.questions.find((x) => x.id === p.id);
    return q ? [{ ...p, question: q.question }] : [];
  });

  if (error && !data) return <ErrorState message={error.message} error={error} onRetry={() => mutate()} />;
  if (!data) {
    return (
      <div role="status" aria-busy="true" className="space-y-2">
        <CardSkeleton height="h-96" />
        <p className="text-xs text-muted-foreground">Reading the context lines written so far.</p>
      </div>
    );
  }

  const cov = data.coverage;
  const missing = cov.passages - cov.written;
  const job = cov.job;
  const preset = presets.find((p) => p.id === question);

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle as="h2">Context lines</CardTitle>
          <CardDescription>
            A model reads the part of the document a passage sits in and writes one or two sentences saying what the
            passage alone does not: whose figures, which year, which table. The line is embedded with the passage.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-wrap items-end gap-x-5 gap-y-3">
            <div className="min-w-0 space-y-1">
              <label htmlFor="context-index" className="text-xs font-medium text-muted-foreground">
                Index
              </label>
              <Select
                value={data.profile}
                items={Object.fromEntries(data.profiles.map((p) => [p.id, p.label]))}
                onValueChange={(v) => {
                  if (!v) return;
                  setProfile(v as string);
                  setChunk(null);
                }}
              >
                <SelectTrigger id="context-index" className="w-full max-w-96 sm:w-96">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {data.profiles.map((p) => (
                    <SelectItem key={p.id} value={p.id}>
                      {p.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>

          <div className="space-y-1.5">
            <p className="text-xs font-medium text-muted-foreground">Test question</p>
            <div className="flex flex-wrap gap-1.5" role="group" aria-label="Test question to rank">
              {presets.map((p) => (
                <button
                  key={p.id}
                  type="button"
                  onClick={() => {
                    setQuestion(p.id);
                    setChunk(null);
                  }}
                  aria-pressed={question === p.id}
                  title={p.why}
                  className={cn(
                    "rounded-md px-2.5 py-1 text-left text-xs ring-1 ring-foreground/10 transition-colors hover:bg-muted focus-visible:ring-[3px] focus-visible:ring-ring/50 focus-visible:outline-none",
                    question === p.id && "bg-muted ring-foreground/40",
                  )}
                >
                  <span className="font-medium tabular-nums">{p.id}</span> {p.question}
                </button>
              ))}
              <button
                type="button"
                onClick={() => setQuestion(null)}
                aria-pressed={question === null}
                className={cn(
                  "rounded-md px-2.5 py-1 text-xs ring-1 ring-foreground/10 transition-colors hover:bg-muted focus-visible:ring-[3px] focus-visible:ring-ring/50 focus-visible:outline-none",
                  question === null && "bg-muted ring-foreground/40",
                )}
              >
                None
              </button>
            </div>
            {preset && <p className="text-xs text-pretty text-muted-foreground">{preset.why}</p>}
          </div>

          <div className="@container">
            <TileGrid className="grid-cols-2 @2xl:grid-cols-4">
              <Tile
                label="Context lines written"
                value={`${num(cov.written)} of ${num(cov.passages)}`}
                hint={missing === 0 ? "every passage" : `${Math.floor((cov.written / Math.max(cov.passages, 1)) * 100)}% so far`}
              />
              <Tile label="Windows" value={num(cov.blocks)} hint="parts of a section, each read for every passage in it" />
              <Tile label="Written by" value={data.describe.model} hint="on Remote AI" />
              <Tile
                label="Index with context lines"
                value={cov.built ? `${num(cov.rows)} passages` : "Not built yet"}
                hint={cov.built ? "built" : "it builds once every line is written"}
              />
            </TileGrid>
          </div>

          {(missing > 0 || !cov.built) && (
            <div className="space-y-1 rounded-lg bg-muted/30 p-3 text-xs text-pretty text-muted-foreground ring-1 ring-foreground/10">
              {missing > 0 && (
                <p>
                  {num(missing)} passages have no context line yet. They are written on Remote AI with{" "}
                  <code className="font-mono">./rag context write {data.profile}</code>; this page shows only the lines
                  already written.
                  {job?.status === "interrupted" &&
                    ` The last job stopped at ${num(job.done)} of ${num(job.total)}; writing again resumes from there.`}
                </p>
              )}
              {!cov.built && (
                <p>
                  The index with context lines is not built yet, so a question is ranked only on the index without them.
                  Build it once every line is written: <code className="font-mono">./rag index build {data.profile}</code>
                </p>
              )}
            </div>
          )}

          <TechnicalDetails
            rows={{
              Model: data.describe.model,
              Prompt: `${data.describe.prompt_version} · sha256 ${data.describe.prompt_sha256.slice(0, 16)}`,
              Windows: data.describe.window_policy,
              "Context size": `${num(data.describe.num_ctx)} tokens (num_ctx)`,
              "Model digest": cov.digest ? cov.digest.replace(/^sha256[-:]/, "").slice(0, 12) : NO_DATA,
              Backup: cov.export ? `data/context/${data.profile}.jsonl` : "not exported yet (needs every line)",
              Timings: Object.entries(data.timings)
                .map(([k, v]) => `${k.replace(/_ms$/, "")} ${num(Math.round(v))} ms`)
                .join(" · "),
            }}
          >
            <p className="text-muted-foreground">The prompt, word for word (the window goes between its parts):</p>
            <pre className="max-h-56 overflow-auto font-mono text-[11px] leading-relaxed whitespace-pre-wrap">
              {data.prompt.system}
              {"\n\n<document> … </document> <passage> … </passage>\n\n"}
              {data.prompt.suffix}
            </pre>
          </TechnicalDetails>
        </CardContent>
      </Card>

      {error ? (
        <ErrorState message={error.message} error={error} onRetry={() => mutate()} />
      ) : (
        <div className={cn("space-y-4", isLoading && "opacity-70")}>
          {isLoading && (
            <p className="flex items-center gap-1 text-xs text-muted-foreground" role="status">
              <Loader2 className="size-3.5 animate-spin" aria-hidden />
              {question ? "Searching both indexes..." : "Reading..."}
            </p>
          )}
          {data.passage ? (
            <PassageCard passage={data.passage} data={data} docs={docs} />
          ) : (
            <Card>
              <CardContent className="text-sm text-pretty text-muted-foreground">
                No passage to show yet: no context line has been written for this index. Write them on Remote AI with{" "}
                <code className="font-mono text-xs">./rag context write {data.profile}</code>
              </CardContent>
            </Card>
          )}
          {data.ranking && <RankCard ranking={data.ranking} data={data} docs={docs} />}
          <SamplesCard data={data} docs={docs} chosen={chunk ?? data.passage?.chunk_id ?? null} onPick={setChunk} />
        </div>
      )}
    </div>
  );
}

function PassageCard({ passage: p, data, docs }: { passage: ContextPassage; data: ContextLabData; docs: DocumentFile[] | undefined }) {
  const onAnswerPage = data.ranking?.targets.some((t) => t.doc_id === p.doc_id && t.pages.includes(p.page)) ?? false;
  return (
    <Card>
      <CardHeader>
        <CardTitle as="h2">One passage</CardTitle>
        <CardDescription>
          {docTitle(p.doc_id, docs)} · page {p.page} · Section {p.section_path ?? NO_DATA}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {onAnswerPage && (
          <Badge className="h-5 gap-1 bg-sky-600 px-1.5 text-[11px] font-normal text-white dark:bg-sky-500 dark:text-black">
            <Target className="size-3" aria-hidden />
            On an answer page of {data.ranking?.id}
          </Badge>
        )}

        <div className="space-y-1">
          <p className="text-xs font-medium text-muted-foreground">Context line (written by {data.describe.model})</p>
          {p.context ? (
            <p className="rounded-lg bg-sky-50 px-3 py-2 text-sm text-pretty ring-1 ring-sky-600/20 dark:bg-sky-950/40 dark:ring-sky-400/25">
              {p.context}
            </p>
          ) : (
            <p className="text-sm text-muted-foreground">No context line written for this passage yet.</p>
          )}
        </div>

        <div className="@container">
          <div className="grid gap-4 @3xl:grid-cols-2">
            <Embedded title="Embedded without the line" text={p.embedded_before} />
            {p.embedded_after && p.context ? (
              <Embedded title="Embedded with the line" text={p.embedded_after} mark={p.context} />
            ) : (
              <section className="min-w-0 space-y-1.5">
                <h3 className="text-xs font-medium text-muted-foreground">Embedded with the line</h3>
                <p className="text-xs text-muted-foreground">Nothing yet: this passage has no line.</p>
              </section>
            )}
          </div>
        </div>

        {p.ranks && Object.keys(p.ranks).length > 0 && (
          <p className="text-xs text-muted-foreground tabular-nums">
            This passage for {data.ranking?.id}:{" "}
            {(["plain", "ctx"] as const)
              .filter((side) => side in (p.ranks ?? {}))
              .map((side) => {
                const r = p.ranks?.[side];
                return `${side === "plain" ? "without the line" : "with the line"} ${r ? `rank ${r}` : "not in the top 50"}`;
              })
              .join(" · ")}
          </p>
        )}

        <p className="max-w-prose text-xs text-pretty text-muted-foreground">
          Only search reads the line. The answer model is given the passage and its source, exactly as on an index
          without context lines, so a better answer can only come from finding the right passage.
        </p>

        <TechnicalDetails
          label="The window the line was written from"
          rows={{
            Passages: `${num(p.window.passages)} of this section share it; this is number ${num(p.window.position)}`,
            Size: `${num(p.window.chars)} characters, about ${num(p.window.est_tokens)} tokens (characters / 3)`,
          }}
        >
          <pre className="max-h-48 overflow-auto font-mono text-[11px] leading-relaxed whitespace-pre-wrap">
            {p.window.head}
            {p.window.chars > p.window.head.length ? " …" : ""}
          </pre>
        </TechnicalDetails>
      </CardContent>
    </Card>
  );
}

/** The exact text the embedding model reads, with the context line marked. */
function Embedded({ title, text, mark }: { title: string; text: string; mark?: string }) {
  const at = mark ? text.indexOf(mark) : -1;
  return (
    <section className="min-w-0 space-y-1.5">
      <h3 className="text-xs font-medium text-muted-foreground">{title}</h3>
      <pre className="max-h-72 overflow-auto rounded-lg bg-muted/40 p-3 font-mono text-[11px] leading-relaxed break-words whitespace-pre-wrap ring-1 ring-foreground/10">
        {at < 0 || !mark ? (
          text
        ) : (
          <>
            {text.slice(0, at)}
            <mark className="rounded-sm bg-sky-200/70 px-0.5 text-foreground dark:bg-sky-800/60">{mark}</mark>
            {text.slice(at + mark.length)}
          </>
        )}
      </pre>
    </section>
  );
}

function RankCard({
  ranking: r,
  data,
  docs,
}: {
  ranking: NonNullable<ContextLabData["ranking"]>;
  data: ContextLabData;
  docs: DocumentFile[] | undefined;
}) {
  return (
    <Card>
      <CardHeader>
        <CardTitle as="h2">Where the answer ranks</CardTitle>
        <CardDescription>
          Test question {r.id}: {r.question} The same question, the same model and the same passages; only the context
          lines differ.
        </CardDescription>
      </CardHeader>
      <CardContent className="@container">
        <div className="grid gap-5 @3xl:grid-cols-2">
          <RankSide title="Without context lines" sub={data.plain_label} profile={data.plain_profile} side={r.plain} docs={docs} />
          <RankSide title="With context lines" sub={data.label} profile={data.profile} side={r.ctx} docs={docs} />
        </div>
      </CardContent>
    </Card>
  );
}

function RankSide({
  title,
  sub,
  profile,
  side,
  docs,
}: {
  title: string;
  sub: string | null;
  profile: string;
  side: ContextRankSide | null;
  docs: DocumentFile[] | undefined;
}) {
  return (
    <section className="min-w-0 space-y-2" aria-label={title}>
      <div className="space-y-0.5">
        <h3 className="text-sm font-medium">{title}</h3>
        {sub && <p className="text-xs text-muted-foreground">{sub}</p>}
      </div>
      {side === null ? (
        <p className="rounded-lg bg-muted/30 p-3 text-xs text-pretty text-muted-foreground ring-1 ring-foreground/10">
          Not built yet. Build it with <code className="font-mono">./rag index build {profile}</code>
        </p>
      ) : (
        <>
          <p className="text-xs tabular-nums">
            <span className="text-muted-foreground">First answer page: </span>
            {side.first_answer_rank === null ? (
              <span>not in the top 50</span>
            ) : (
              <span className={cn("font-medium", side.first_answer_rank <= 5 ? "text-sky-700 dark:text-sky-300" : undefined)}>
                rank {side.first_answer_rank}
              </span>
            )}
            <span className="text-muted-foreground"> (searching 50 deep)</span>
          </p>
          <ol className="space-y-1">
            {side.top.map((h) => (
              <li
                key={h.chunk_id}
                className={cn(
                  "flex items-baseline gap-2 rounded-md bg-card px-2.5 py-1.5 text-xs ring-1 ring-foreground/10",
                  h.answer_page && "ring-sky-500/60 dark:ring-sky-400/60",
                )}
              >
                <span className="font-semibold tabular-nums">{h.rank}</span>
                <span className="min-w-0 truncate" title={docTitle(h.doc_id, docs)}>
                  {docTitle(h.doc_id, docs)}
                </span>
                <span className="ml-auto shrink-0 text-muted-foreground tabular-nums">p. {h.page}</span>
                {h.answer_page && (
                  <Target className="size-3 shrink-0 text-sky-600 dark:text-sky-400" aria-label="Answer page" />
                )}
              </li>
            ))}
          </ol>
        </>
      )}
    </section>
  );
}

function SamplesCard({
  data,
  docs,
  chosen,
  onPick,
}: {
  data: ContextLabData;
  docs: DocumentFile[] | undefined;
  chosen: string | null;
  onPick: (chunkId: string) => void;
}) {
  return (
    <Card>
      <CardHeader>
        <CardTitle as="h2">More context lines</CardTitle>
        <CardDescription>A few lines from across the documents. Pick one to see it above.</CardDescription>
      </CardHeader>
      <CardContent>
        {data.samples.length === 0 ? (
          <p className="text-sm text-muted-foreground">No context lines written for this index yet.</p>
        ) : (
          <ul className="space-y-1.5">
            {data.samples.map((s) => (
              <li key={s.chunk_id}>
                <button
                  type="button"
                  onClick={() => onPick(s.chunk_id)}
                  aria-pressed={chosen === s.chunk_id}
                  className={cn(
                    "w-full space-y-1 rounded-lg bg-card px-3 py-2 text-left ring-1 ring-foreground/10 transition-colors hover:bg-muted/50 focus-visible:ring-[3px] focus-visible:ring-ring/50 focus-visible:outline-none",
                    chosen === s.chunk_id && "bg-muted/60 ring-foreground/40",
                  )}
                >
                  <span className="flex items-baseline gap-2 text-[11px] text-muted-foreground">
                    <span className="min-w-0 truncate">
                      {docTitle(s.doc_id, docs)} · Section {s.section_path ?? NO_DATA}
                    </span>
                    <span className="ml-auto shrink-0 tabular-nums">p. {s.page}</span>
                  </span>
                  <span className="block text-sm text-pretty">{s.context}</span>
                  <span className="line-clamp-2 block text-[11px] text-pretty text-muted-foreground">{s.text}</span>
                </button>
              </li>
            ))}
          </ul>
        )}
      </CardContent>
    </Card>
  );
}
