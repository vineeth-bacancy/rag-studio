"use client";

import { useState } from "react";
import useSWR from "swr";
import { Check } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { CardSkeleton, ErrorState } from "@/components/common/states";
import { TechnicalDetails } from "@/components/common/technical-details";
import { OutcomeBadge } from "@/components/eval/badges";
import type { DocumentFile, FigureCheckValue, FigureExample, FiguresLab as FiguresLabData, FigureSetCRun } from "@/lib/api";
import { docTitle } from "@/lib/documents";
import { NO_DATA } from "@/lib/format";
import { cn } from "@/lib/utils";

function exampleName(e: FigureExample): string {
  return e.qid ? `${e.label} (Set C ${e.qid})` : e.label;
}

function percent(v: number | null | undefined): string {
  return v === null || v === undefined ? NO_DATA : `${(v * 100).toFixed(1)}%`;
}

function valueName(v: FigureCheckValue): string {
  return [v.bar, v.segment, v.series].filter(Boolean).join(" / ") || NO_DATA;
}

/**
 * Step 11: figures. A picture the page text cannot explain, what the text
 * already said about it ("before"), what the vision model wrote ("after"),
 * and, for the Set C charts, the values read back against the values
 * measured from the drawing, then the answers each setup gave.
 */
export function FiguresLab() {
  const [pick, setPick] = useState<string | null>(null);
  const key = `/learn/figures${pick ? `?${new URLSearchParams({ figure: pick })}` : ""}`;
  const { data, error, isLoading, mutate } = useSWR<FiguresLabData>(key, { keepPreviousData: true });
  const { data: docData } = useSWR<{ documents: DocumentFile[] }>("/documents");
  const shown = data && (!pick || data.figure.figure_id === pick) ? data : null;
  const current = pick ?? data?.examples[0]?.figure_id ?? null;

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle as="h2">A figure, before and after</CardTitle>
          <CardDescription>
            Page text can say nothing about a chart: the numbers may only exist as the height of a bar. A vision model
            looks at the picture and writes down what it shows; that description becomes a passage the search can find.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {data && (
            <div className="min-w-0 space-y-1">
              <label htmlFor="figure-pick" className="text-xs font-medium text-muted-foreground">
                Figure
              </label>
              <Select
                value={current}
                items={Object.fromEntries(data.examples.map((e) => [e.figure_id, exampleName(e)]))}
                onValueChange={(v) => {
                  if (v) setPick(v as string);
                }}
              >
                <SelectTrigger id="figure-pick" className="w-full max-w-xl">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {data.examples.map((e) => (
                    <SelectItem key={e.figure_id} value={e.figure_id}>
                      {exampleName(e)}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          )}
          {data && <Status data={data} />}
        </CardContent>
      </Card>

      {error ? (
        <ErrorState message={error.message} error={error} onRetry={() => mutate()} />
      ) : !shown ? (
        <CardSkeleton height="h-96" />
      ) : (
        <div className={cn("space-y-4", isLoading && "opacity-70")}>
          <Figure data={shown} docs={docData?.documents} />
          {shown.question && <Question data={shown} />}
          {shown.truth && <Truth data={shown} />}
          <SetC runs={shown.set_c_runs} />
        </div>
      )}
    </div>
  );
}

function Status({ data }: { data: FiguresLabData }) {
  const s = data.status;
  const tiles: [string, number][] = [
    ["Figures found", s.figures],
    ["Described", s.described],
    ["In the index", s.indexed],
    ["Photos and logos (not indexed)", s.pictures],
    ["Bad outputs", s.bad],
    ["Not described yet", s.missing],
  ];
  return (
    <div className="space-y-2 @container">
      <dl className="grid grid-cols-2 gap-2 @sm:grid-cols-3 @3xl:grid-cols-6">
        {tiles.map(([label, n]) => (
          <div key={label} className="rounded-md bg-muted/50 px-2.5 py-1.5">
            <dt className="text-[11px] text-muted-foreground">{label}</dt>
            <dd className="text-sm font-medium tabular-nums">{n.toLocaleString()}</dd>
          </div>
        ))}
      </dl>
      <p className="text-xs text-muted-foreground">
        Described by {s.model} (prompt {s.prompt_version}).
      </p>
    </div>
  );
}

function Figure({ data, docs }: { data: FiguresLabData; docs: DocumentFile[] | undefined }) {
  const { figure, before, after } = data;
  const title = figure.title_above ?? figure.heading;
  const alt = title ?? `Figure on page ${figure.page}`;
  const timings: Record<string, React.ReactNode> = {
    "Time to describe": after.ms === null ? NO_DATA : `${(after.ms / 1000).toFixed(1)} s`,
    "Prompt tokens": after.prompt_tokens ?? NO_DATA,
    "Written tokens": after.eval_tokens ?? NO_DATA,
    Resolution: `${figure.dpi} dpi`,
    Crop: `[${figure.crop.map((n) => n.toFixed(1)).join(", ")}] pt`,
    "Image SHA-256": <span className="font-mono break-all">{figure.image_sha256}</span>,
  };
  return (
    <Card className="@container">
      <CardContent className="grid gap-5 @4xl:grid-cols-2">
        <figure className="min-w-0 space-y-2">
          <div className="overflow-hidden rounded-md bg-white ring-1 ring-foreground/15">
            {/* eslint-disable-next-line @next/next/no-img-element -- a PNG the API renders on demand */}
            <img src={figure.image} alt={alt} className="mx-auto max-h-[36rem] w-auto max-w-full object-contain" />
          </div>
          <figcaption className="space-y-0.5 text-xs">
            <p className="font-medium">
              {docTitle(figure.doc_id, docs)}, p. {figure.page_label ?? figure.page}
            </p>
            {figure.title_above && <p>{figure.title_above}</p>}
            {figure.heading && <p className="text-muted-foreground">{figure.heading}</p>}
            {figure.section_path.length > 0 && (
              <p className="text-muted-foreground">{figure.section_path.join(" › ")}</p>
            )}
          </figcaption>
        </figure>

        <div className="min-w-0 space-y-5">
          <section className="space-y-2">
            <h3 className="text-sm font-medium">Before: what the text already had</h3>
            <div className="space-y-1">
              <p className="text-xs font-medium text-muted-foreground">Printed inside the picture</p>
              {before.inside.length === 0 ? (
                <p className="text-xs">No text inside this picture: it is drawn as an image.</p>
              ) : (
                <ul className="flex flex-wrap gap-1">
                  {before.inside.map((t, i) => (
                    <li key={i}>
                      <Badge variant="outline" className="h-auto px-1.5 py-0.5 font-normal whitespace-normal">
                        {t}
                      </Badge>
                    </li>
                  ))}
                </ul>
              )}
            </div>
            <div className="space-y-1">
              <p className="text-xs font-medium text-muted-foreground">Printed beside it</p>
              {before.beside.length === 0 ? (
                <p className="text-xs">{NO_DATA}</p>
              ) : (
                <ul className="space-y-1 text-xs">
                  {before.beside.map((b, i) => (
                    <li key={i} className="text-pretty">
                      <span className="text-muted-foreground">{b.label}: </span>
                      {b.text}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </section>

          <section className="space-y-2">
            <h3 className="flex flex-wrap items-center gap-2 text-sm font-medium">
              After: what {data.model} wrote
              {after.kind && (
                <Badge variant="secondary" className="font-normal">
                  {after.kind}
                </Badge>
              )}
            </h3>
            {after.bad ? (
              <p className="rounded-md bg-amber-500/10 px-3 py-2 text-xs text-amber-800 dark:text-amber-300">
                The description was rejected: {after.bad.error} It is left out of the index.
              </p>
            ) : after.description === null ? (
              <p className="text-xs text-muted-foreground">
                Not described yet. Describe the figures on Remote AI first: ./rag figures describe
              </p>
            ) : (
              <p className="max-h-72 overflow-auto rounded-md bg-muted/50 px-3 py-2 text-xs whitespace-pre-wrap">
                {after.description}
              </p>
            )}
            <p className="text-xs">
              {after.indexed
                ? "In the index as a figure passage."
                : after.picture
                  ? "Not in the index: a photo or logo. Its facts are in the caption beside it."
                  : "Not in the index."}
            </p>
            <TechnicalDetails rows={timings}>
              {after.passage && (
                <div className="space-y-1">
                  <p className="font-medium">Passage that is embedded</p>
                  <pre className="max-h-48 overflow-auto font-mono text-[11px] whitespace-pre-wrap">{after.passage}</pre>
                </div>
              )}
              {after.label && (
                <div className="space-y-1">
                  <p className="font-medium">Label that is embedded with it</p>
                  <pre className="max-h-32 overflow-auto font-mono text-[11px] whitespace-pre-wrap">{after.label}</pre>
                </div>
              )}
            </TechnicalDetails>
          </section>
        </div>
      </CardContent>
    </Card>
  );
}

function Question({ data }: { data: FiguresLabData }) {
  const q = data.question!;
  return (
    <Card>
      <CardHeader>
        <CardTitle as="h2" className="text-base">
          Set C question {q.id}
        </CardTitle>
        <CardDescription>{q.question}</CardDescription>
      </CardHeader>
      <CardContent className="space-y-1 text-xs">
        <p>
          <span className="font-medium">Answer key: </span>
          {q.key}
        </p>
        <p className="text-pretty text-muted-foreground">{q.note}</p>
      </CardContent>
    </Card>
  );
}

function Truth({ data }: { data: FiguresLabData }) {
  const t = data.truth!;
  const c = t.check;
  return (
    <Card className="@container">
      <CardHeader>
        <CardTitle as="h2" className="text-base">
          Read from the image vs measured from the drawing
        </CardTitle>
        <CardDescription>{t.method}</CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        {!c ? (
          <p className="text-xs text-muted-foreground">No check against the drawing for this figure.</p>
        ) : (
          <>
            <p className="text-sm">
              Read {c.read} of {c.of} within {c.tolerance} {t.unit}; mean error {percent(c.mean_error_share)} of the
              axis. Values in {t.unit}.
            </p>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Value</TableHead>
                  <TableHead className="text-right">Measured</TableHead>
                  <TableHead className="text-right">Read</TableHead>
                  <TableHead className="hidden text-right @sm:table-cell">Error</TableHead>
                  <TableHead className="w-20 text-center whitespace-normal">Within tolerance</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {c.values.map((v, i) => (
                  <TableRow key={i}>
                    <TableCell className="whitespace-normal">{valueName(v)}</TableCell>
                    <TableCell className="text-right tabular-nums">{v.value}</TableCell>
                    <TableCell className="text-right tabular-nums">{v.reading ?? NO_DATA}</TableCell>
                    <TableCell className="hidden text-right tabular-nums @sm:table-cell">{percent(v.error_share)}</TableCell>
                    <TableCell className="text-center">
                      {v.read ? <Check className="mx-auto size-4 text-emerald-600" aria-label="Yes" /> : <span aria-label="No">{NO_DATA}</span>}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
            <p className="text-xs text-muted-foreground">
              Error is shown as a share of the axis. This check is a hint: the description was also read by hand.
            </p>
          </>
        )}
      </CardContent>
    </Card>
  );
}

function SetC({ runs }: { runs: FigureSetCRun[] }) {
  const sorted = [...runs].sort((a, b) => (b.started_at ?? "").localeCompare(a.started_at ?? ""));
  return (
    <Card className="@container">
      <CardHeader>
        <CardTitle as="h2" className="text-base">
          Set C: answers by setup
        </CardTitle>
        <CardDescription>The chart questions, answered with each setup that has been run.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {sorted.length === 0 ? (
          <p className="text-xs text-muted-foreground">No Set C runs yet. They arrive with ladder row 11.</p>
        ) : (
          sorted.map((r) => (
            <div key={r.id} className="space-y-1.5">
              <p className="text-xs">
                <span className="font-medium">{r.label ?? r.id}</span>
                <span className="text-muted-foreground">
                  {" "}
                  · {r.config.profile} · {r.config.answer_model}
                </span>
              </p>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Question</TableHead>
                    <TableHead>Result</TableHead>
                    <TableHead className="hidden text-right @sm:table-cell">Best rank</TableHead>
                    <TableHead>Answer</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {r.rows.map((row) => (
                    <TableRow key={row.id}>
                      <TableCell>{row.id}</TableCell>
                      <TableCell>
                        <OutcomeBadge outcome={row.outcome} error={row.error} />
                      </TableCell>
                      <TableCell className="hidden text-right tabular-nums @sm:table-cell">{row.best_rank ?? NO_DATA}</TableCell>
                      <TableCell className="max-w-96 whitespace-normal">
                        <span className="line-clamp-3 text-pretty" title={row.answer ?? undefined}>
                          {row.answer ?? NO_DATA}
                        </span>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          ))
        )}
      </CardContent>
    </Card>
  );
}
