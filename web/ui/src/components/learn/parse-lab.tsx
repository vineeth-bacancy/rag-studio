"use client";

import { useMemo, useState } from "react";
import useSWR from "swr";
import { EyeOff } from "lucide-react";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { CardSkeleton, ErrorState } from "@/components/common/states";
import { TechnicalDetails } from "@/components/common/technical-details";
import { MarkedText } from "@/components/documents/marked-text";
import { PageCanvas, type Overlay } from "@/components/documents/page-canvas";
import type { DocumentFile, PageInspection } from "@/lib/api";
import { docTitle } from "@/lib/documents";
import { NO_DATA, num } from "@/lib/format";
import { cn } from "@/lib/utils";
import { DocPagePicker, type Preset } from "./doc-page-picker";

const GPO = "gpo_2018_annual_report";
const GEP = "wb_gep_2026_06";
const IMAC = "apple_imac_per_2024_10";

const PRESETS: Preset[] = [
  { docId: GPO, page: 29, label: "GPO p29: a footer hidden under a full-page fill" },
  { docId: GPO, page: 15, label: "GPO p15: text placed off the page" },
  { docId: GEP, page: 96, label: "GEP p96: printed as page 74" },
  { docId: IMAC, page: 10, label: "iMac p10: a table, flattened or kept" },
  { docId: GEP, page: 19, label: "GEP p19: the executive summary" },
];

const NOTHING = new Set<string>();

/**
 * Step 1: one page as the PDF draws it, beside what each parser read from it.
 * Raw text only: cleaning is the next step. Hidden text is outlined on the
 * page and highlighted in the text, so what a reader cannot see but a parser
 * reads is visible in both places.
 */
export function ParseLab() {
  const { data: docData } = useSWR<{ documents: DocumentFile[] }>("/documents");
  const [target, setTarget] = useState({ docId: GPO, page: 29 });
  const key = `/documents/${encodeURIComponent(target.docId)}/pages/${target.page}/inspect`;
  const { data, error, isLoading, mutate } = useSWR<PageInspection>(key, { keepPreviousData: true });
  const docs = docData?.documents ?? [];

  const overlays = useMemo<Overlay[]>(() => {
    if (!data) return [];
    const out: Overlay[] = data.pymupdf.hidden.map((h) => ({
      box: h.box, className: "bg-red-500/15 ring-2 ring-red-500", title: `Hidden text (${h.label}): ${h.text}`,
    }));
    for (const it of data.docling?.items ?? []) {
      if (it.label === "table" || it.label === "document_index") {
        out.push({ box: it.box, className: "ring-1 ring-sky-500/80 bg-sky-500/5", title: "Docling: table" });
      }
    }
    return out;
  }, [data]);

  if (!docData) return <CardSkeleton height="h-96" />;

  return (
    <Card>
      <CardHeader>
        <CardTitle as="h2">One page, two parsers</CardTitle>
        <CardDescription>
          Red outlines are text the page hides but a parser still reads. Blue outlines are tables Docling found.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-5">
        <DocPagePicker
          id="parse"
          docs={docs}
          docId={target.docId}
          page={target.page}
          pageCount={data?.doc_id === target.docId ? data.page_count : null}
          presets={PRESETS}
          onChange={(docId, page) => setTarget({ docId, page })}
        />

        {error ? (
          <ErrorState message={error.message} error={error} onRetry={() => mutate()} />
        ) : !data ? (
          <div className="grid gap-4 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]" role="status" aria-busy="true">
            <Skeleton className="aspect-[8.5/11] w-full" />
            <Skeleton className="h-96 w-full" />
          </div>
        ) : (
          <div className={cn("grid gap-5 lg:grid-cols-[minmax(0,4fr)_minmax(0,8fr)]", isLoading && "opacity-70")}>
            <div className="min-w-0 space-y-4">
              <PageCanvas
                docId={data.doc_id}
                page={data.page}
                width={data.width}
                height={data.height}
                overlays={overlays}
                alt={`Page ${data.page} of ${docTitle(data.doc_id, docs)}`}
              />
              <PageNumbers page={data.page} label={data.label} />
              <HiddenList data={data} />
            </div>
            <div className="min-w-0 @container">
              <div className="grid min-w-0 gap-4 @2xl:grid-cols-2">
                <ParserText
                  title="PyMuPDF"
                  hint="Text in the order the PDF stores it, one line per line on the page."
                  text={data.pymupdf.text_raw}
                  marks={data.pymupdf.raw_marks.filter((m) => m.rule === "hidden")}
                />
                <ParserText
                  title="Docling"
                  hint="A layout model's reading order, with tables as rows that name their columns."
                  text={data.docling?.text_raw ?? null}
                  marks={(data.docling?.raw_marks ?? []).filter((m) => m.rule === "hidden")}
                />
              </div>
            </div>
          </div>
        )}

        {data && (
          <TechnicalDetails
            rows={{
              "PyMuPDF characters": num(data.pymupdf.text_raw.length),
              "Docling characters": data.docling ? num(data.docling.text_raw.length) : NO_DATA,
              "Page size": `${Math.round(data.width)} × ${Math.round(data.height)} points`,
              "Hidden text test": "The page is drawn with and without each span; a span is hidden if no pixel changes.",
              "Near-empty page": data.pymupdf.near_empty ? "yes: 200 characters or fewer, so no passages" : "no",
            }}
          />
        )}
      </CardContent>
    </Card>
  );
}

/** The three numbers a page has, and which one a citation uses. */
function PageNumbers({ page, label }: { page: number; label: string | null }) {
  const rows = [
    { name: "Python index", value: String(page - 1), hint: "What code counts from: the first page is 0." },
    { name: "PDF page", value: String(page), hint: "What every citation uses: the first page is 1." },
    { name: "Printed page", value: label ?? NO_DATA, hint: "What the page itself shows, if it shows one." },
  ];
  return (
    <section aria-labelledby="page-numbers" className="space-y-2">
      <h3 id="page-numbers" className="text-sm font-medium">
        Three page numbers
      </h3>
      <dl className="grid grid-cols-3 gap-px overflow-hidden rounded-lg bg-border text-center">
        {rows.map((r) => (
          <div key={r.name} className="space-y-0.5 bg-card px-2 py-2" title={r.hint}>
            <dt className="text-[11px] text-muted-foreground">{r.name}</dt>
            <dd className="text-lg font-semibold tabular-nums">{r.value}</dd>
          </div>
        ))}
      </dl>
      <p className="text-xs text-pretty text-muted-foreground">
        Citations always use the PDF page. The printed number is kept beside it, because that is the one a reader
        looks for.
      </p>
    </section>
  );
}

function HiddenList({ data }: { data: PageInspection }) {
  const hidden = data.pymupdf.hidden;
  return (
    <section aria-labelledby="hidden-heading" className="space-y-2">
      <h3 id="hidden-heading" className="text-sm font-medium">
        Hidden on this page
      </h3>
      {hidden.length === 0 ? (
        <p className="text-sm text-muted-foreground">Nothing: every span of text on this page is visible.</p>
      ) : (
        <ul className="space-y-1.5 text-xs">
          {hidden.map((h, i) => (
            <li key={i} className="flex items-start gap-2">
              <EyeOff className="mt-0.5 size-3.5 shrink-0 text-red-600 dark:text-red-400" aria-hidden />
              <span className="min-w-0">
                <span className="font-medium">{h.label}</span>
                <span className="block truncate font-mono text-muted-foreground" title={h.text}>
                  {h.text}
                </span>
              </span>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

function ParserText({
  title,
  hint,
  text,
  marks,
}: {
  title: string;
  hint: string;
  text: string | null;
  marks: { start: number; end: number; rule: string }[];
}) {
  return (
    <section className="min-w-0 space-y-1.5" aria-label={`${title} text`}>
      <h3 className="text-sm font-medium">{title}</h3>
      <p className="text-xs text-muted-foreground">{hint}</p>
      <div className="max-h-[32rem] overflow-y-auto rounded-lg bg-muted/30 p-3 ring-1 ring-foreground/10">
        {text === null ? (
          <p className="text-sm text-muted-foreground">
            Docling has not converted this document yet. Run: ./rag ingest --profile docling-clean
          </p>
        ) : (
          <MarkedText text={text} marks={joinRuns(text, marks)} hidden={NOTHING} />
        )}
      </div>
    </section>
  );
}

/**
 * The guard marks hidden text glyph by glyph, so letter-spaced text ("U . S .
 * G O V E R N M E N T") came out as one box per letter. Marks of one rule that
 * only whitespace separates are drawn as one box.
 */
function joinRuns(text: string, marks: { start: number; end: number; rule: string }[]) {
  const out: { start: number; end: number; rule: string }[] = [];
  for (const m of [...marks].sort((a, b) => a.start - b.start)) {
    const last = out[out.length - 1];
    if (last && last.rule === m.rule && m.start >= last.end && !text.slice(last.end, m.start).trim()) {
      last.end = Math.max(last.end, m.end);
    } else {
      out.push({ ...m });
    }
  }
  return out;
}
