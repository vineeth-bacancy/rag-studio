"use client";

import { useMemo, useState } from "react";
import useSWR from "swr";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { CardSkeleton, ErrorState } from "@/components/common/states";
import { MarkedText } from "@/components/documents/marked-text";
import { RULE_ORDER, RULE_STYLES, ruleLabel } from "@/components/documents/rule-styles";
import type { CleaningSummary, DocumentFile, PageInspection, ParserSummary } from "@/lib/api";
import { NO_DATA, num } from "@/lib/format";
import { cn } from "@/lib/utils";

type Parser = "pymupdf" | "docling";

/** Pages a rule acted on, and how many times in all. */
function ruleTotals(summary: ParserSummary | null | undefined): Record<string, { pages: number[]; count: number }> {
  const out: Record<string, { pages: number[]; count: number }> = {};
  for (const p of summary?.pages ?? []) {
    for (const [rule, n] of Object.entries(p.counts)) {
      out[rule] ??= { pages: [], count: 0 };
      out[rule].pages.push(p.page);
      out[rule].count += n;
    }
  }
  return out;
}

/**
 * Step 2: every cleaning rule, with how often it acted on a document, and on
 * any page it touched the text before and after, that rule's changes
 * highlighted. The counts and marks are the engine's own record; hovering a
 * highlight in the cleaned text shows what the extractor had there.
 */
export function CleanLab() {
  const { data: docData } = useSWR<{ documents: DocumentFile[] }>("/documents");
  const [docId, setDocId] = useState("wb_gep_2026_06");
  const [parser, setParser] = useState<Parser>("pymupdf");
  const [rule, setRule] = useState("th");
  const [page, setPage] = useState<number | null>(null);
  const { data: summary, error, mutate } = useSWR<CleaningSummary>(`/documents/${encodeURIComponent(docId)}/cleaning`);
  const docs = docData?.documents ?? [];

  const totals = useMemo(() => ruleTotals(summary?.[parser]), [summary, parser]);
  const other = useMemo(() => ruleTotals(summary?.[parser === "pymupdf" ? "docling" : "pymupdf"]), [summary, parser]);
  const pages = totals[rule]?.pages ?? [];
  const shown = page !== null && pages.includes(page) ? page : pages[0] ?? null;

  if (!docData) return <CardSkeleton height="h-96" />;

  return (
    <Card>
      <CardHeader>
        <CardTitle as="h2">Rule by rule</CardTitle>
        <CardDescription>
          The rules run in this order on every page. Choose one to see the pages it changed and exactly what it
          changed.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-5">
        <div className="flex flex-wrap items-end gap-3">
          <div className="min-w-0 space-y-1">
            <label htmlFor="clean-doc" className="text-xs font-medium text-muted-foreground">
              Document
            </label>
            <Select
              value={docId}
              items={Object.fromEntries(docs.map((d) => [d.doc_id, d.title ?? d.doc_id]))}
              onValueChange={(v) => {
                if (!v) return;
                setDocId(v as string);
                setPage(null);
              }}
            >
              <SelectTrigger id="clean-doc" className="w-full max-w-80 sm:w-80">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {docs.map((d) => (
                  <SelectItem key={d.doc_id} value={d.doc_id}>
                    {d.title ?? d.doc_id}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <Tabs
            value={parser}
            onValueChange={(v) => {
              setParser(v as Parser);
              setPage(null);
            }}
          >
            <TabsList>
              <TabsTrigger value="pymupdf">PyMuPDF</TabsTrigger>
              <TabsTrigger value="docling" disabled={!summary?.docling}>Docling</TabsTrigger>
            </TabsList>
          </Tabs>
        </div>

        {error ? (
          <ErrorState message={error.message} error={error} onRetry={() => mutate()} />
        ) : !summary ? (
          <Skeleton className="h-40 w-full" />
        ) : (
          <div className="space-y-2">
            <p className="text-xs font-medium text-muted-foreground" id="clean-rules">
              Rules, in the order they run
            </p>
            <ol className="flex flex-wrap gap-1.5" aria-labelledby="clean-rules">
              {RULE_ORDER.map((r, i) => {
                const t = totals[r];
                return (
                  <li key={r}>
                    <button
                      type="button"
                      aria-pressed={rule === r}
                      onClick={() => {
                        setRule(r);
                        setPage(null);
                      }}
                      className={cn(
                        "inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs ring-1 ring-foreground/15 transition-colors hover:bg-muted focus-visible:ring-[3px] focus-visible:ring-ring/50 focus-visible:outline-none",
                        rule === r && "bg-muted ring-foreground/40",
                        !t && "text-muted-foreground",
                      )}
                    >
                      <span className="text-muted-foreground tabular-nums">{i + 1}.</span>
                      <span className={cn("size-2 rounded-full", RULE_STYLES[r].swatch)} aria-hidden />
                      {ruleLabel(r)}
                      <span className="text-muted-foreground tabular-nums">{t ? num(t.count) : "0"}</span>
                    </button>
                  </li>
                );
              })}
            </ol>
            <p className="text-xs text-muted-foreground">
              {ruleLabel(rule)}: {num(totals[rule]?.count ?? 0)} changes on {num(pages.length)} pages with{" "}
              {parser === "pymupdf" ? "PyMuPDF" : "Docling"}
              {summary[parser === "pymupdf" ? "docling" : "pymupdf"]
                ? ` (${num(other[rule]?.count ?? 0)} with ${parser === "pymupdf" ? "Docling" : "PyMuPDF"})`
                : ""}
              .
            </p>
          </div>
        )}

        {summary && (pages.length === 0 || shown === null ? (
          <p className="rounded-lg bg-muted/30 px-3 py-4 text-sm text-muted-foreground ring-1 ring-foreground/10">
            This rule changed nothing in this document with this parser.
            {parser === "docling" && (rule === "nfkc" || rule === "th")
              ? " Docling decodes ligatures and the \"Th\" glyph itself, so there is nothing left to repair."
              : ""}
          </p>
        ) : (
          <RulePage docId={docId} page={shown} parser={parser} rule={rule} pages={pages} onPage={setPage} />
        ))}
      </CardContent>
    </Card>
  );
}

function RulePage({
  docId,
  page,
  parser,
  rule,
  pages,
  onPage,
}: {
  docId: string;
  page: number;
  parser: Parser;
  rule: string;
  pages: number[];
  onPage: (p: number) => void;
}) {
  const { data, error, isLoading, mutate } = useSWR<PageInspection>(
    `/documents/${encodeURIComponent(docId)}/pages/${page}/inspect`,
    { keepPreviousData: true },
  );
  const view = data ? (parser === "docling" ? data.docling : data.pymupdf) : null;
  // Only this rule's highlights: the others would bury it.
  const others = useMemo(() => new Set(RULE_ORDER.filter((r) => r !== rule)), [rule]);
  const explain = data?.rules.find((r) => r.id === (rule === "compound" ? "hyphen" : rule));
  const pageItems = Object.fromEntries(pages.map((p) => [String(p), `Page ${p}`]));

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-end gap-3">
        <div className="space-y-1">
          <label htmlFor="clean-page" className="text-xs font-medium text-muted-foreground">
            Pages this rule changed ({num(pages.length)})
          </label>
          <Select value={String(page)} items={pageItems} onValueChange={(v) => v && onPage(Number(v))}>
            <SelectTrigger id="clean-page" className="w-40">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {pages.map((p) => (
                <SelectItem key={p} value={String(p)}>
                  Page {p}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        {explain && <p className="max-w-prose flex-1 text-xs text-pretty text-muted-foreground">{explain.explain}</p>}
      </div>

      {error ? (
        <ErrorState message={error.message} error={error} onRetry={() => mutate()} />
      ) : !view ? (
        <Skeleton className="h-80 w-full" />
      ) : (
        <div className={cn("grid gap-4 md:grid-cols-2", isLoading && "opacity-70")}>
          <section className="min-w-0 space-y-1.5" aria-label="Before cleaning">
            <h3 className="text-sm font-medium">Before</h3>
            <div className="max-h-[30rem] overflow-y-auto rounded-lg bg-muted/30 p-3 ring-1 ring-foreground/10">
              <MarkedText text={view.text_raw} marks={view.raw_marks} hidden={others} />
            </div>
          </section>
          <section className="min-w-0 space-y-1.5" aria-label="After cleaning">
            <h3 className="text-sm font-medium">
              After <span className="text-xs font-normal text-muted-foreground">· hover a highlight for the original</span>
            </h3>
            <div className="max-h-[30rem] overflow-y-auto rounded-lg bg-muted/30 p-3 ring-1 ring-foreground/10">
              <MarkedText text={view.text} marks={view.marks} hidden={others} />
            </div>
          </section>
        </div>
      )}
      {view && (
        <p className="text-xs text-muted-foreground">
          Page {page}
          {data?.label ? ` (printed ${data.label})` : ""} · {num(view.counts[rule] ?? 0)} changes by this rule ·{" "}
          {view.text_raw.length ? num(view.text_raw.length) : NO_DATA} characters before, {num(view.text.length)} after
        </p>
      )}
    </div>
  );
}
