"use client";

import { useMemo, useState } from "react";
import useSWR from "swr";
import { Loader2 } from "lucide-react";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { CardSkeleton, ErrorState } from "@/components/common/states";
import { TechnicalDetails } from "@/components/common/technical-details";
import { useDebounced } from "@/hooks/use-debounced";
import type { ChunkLab as ChunkLabData, ChunkMethod, DocumentFile } from "@/lib/api";
import { NO_DATA, num } from "@/lib/format";
import { cn } from "@/lib/utils";
import { ChunkText, type Span } from "./chunk-text";
import { DocPagePicker } from "./doc-page-picker";
import { SimilarityCurve } from "./similarity-curve";

const METHODS: { id: ChunkMethod; label: string; overlap: boolean; explain: (size: number, overlap: number) => string }[] = [
  {
    id: "fixed", label: "Fixed-size", overlap: true,
    explain: (size, overlap) =>
      `A passage every ${num(size)} characters, its end moved back to a space; the next starts ${num(overlap)} characters earlier. The baseline did not move that start to a word, so passages began mid-word; the M3 fix does.`,
  },
  {
    id: "recursive", label: "Recursive", overlap: true,
    explain: (size) =>
      `Split at paragraph breaks, then line breaks, then sentence ends, then spaces, until every piece fits in ${num(size)} characters; then neighbours are joined back up to that size.`,
  },
  {
    id: "sentence", label: "Sentence", overlap: true,
    explain: (size) => `Whole sentences, packed up to ${num(size)} characters. The overlap is whole sentences too.`,
  },
  {
    id: "structure", label: "Structure-aware", overlap: false,
    explain: (size) =>
      `Whole paragraphs, packed up to ${num(size)} characters, never across a section, a contents page or reference list, or a table's edge. Each passage starts with its section label.`,
  },
  {
    id: "semantic", label: "Semantic", overlap: false,
    explain: () =>
      "Every sentence is embedded. A passage ends where two neighbouring sentences are less similar than the threshold: the chosen percentile of this page's distances.",
  },
];

function median(xs: number[]): number | null {
  if (!xs.length) return null;
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)];
}

/**
 * Step 3, the chunk playground: one page cut by every chunker the engine has,
 * with the size and overlap under your control. The cutting happens in the
 * engine (rag_studio/chunk.py), the same code the indexes use; this only
 * paints the passages it returns.
 */
export function ChunkLab() {
  const { data: docData } = useSWR<{ documents: DocumentFile[] }>("/documents");
  const [target, setTarget] = useState({ docId: "wb_gep_2026_06", page: 19 });
  const [parser, setParser] = useState<"pymupdf" | "docling">("pymupdf");
  const [method, setMethod] = useState<ChunkMethod>("fixed");
  const [size, setSize] = useState(1000);
  const [overlap, setOverlap] = useState(150);
  const [percentile, setPercentile] = useState(80);
  const [showBaseline, setShowBaseline] = useState(false);

  const settled = useDebounced({ size, overlap: Math.min(overlap, size - 50), percentile }, 300);
  const info = METHODS.find((m) => m.id === method)!;
  const params = new URLSearchParams({
    doc_id: target.docId,
    page: String(target.page),
    parser,
    method,
    size: String(settled.size),
    overlap: String(info.overlap ? settled.overlap : 0),
    percentile: String(settled.percentile),
  });
  const { data, error, isLoading, mutate } = useSWR<ChunkLabData>(`/learn/chunk?${params}`, { keepPreviousData: true });
  const docs = docData?.documents ?? [];
  // `keepPreviousData` keeps the last result while a slider moves; a result for
  // another page, parser or chunker is never drawn under the current one.
  const fresh = !!data && data.doc_id === target.docId && data.page === target.page
    && data.parser === parser && data.method === method;

  const view = useMemo(() => {
    if (!data) return null;
    const baseline = method === "fixed" && showBaseline && data.baseline;
    const source = baseline ? data.baseline! : data.passages;
    const spans: Span[] = source;
    const flagged = new Set(source.flatMap((s, j) => (s.starts_mid_word ? [j] : [])));
    // The engine's own section label, exactly as an index embeds it.
    const notes = data.sections?.map((s) => s.label);
    return { spans, flagged, notes };
  }, [data, method, showBaseline]);

  if (!docData) return <CardSkeleton height="h-96" />;

  const midWord = data?.baseline ? data.baseline.filter((s) => s.starts_mid_word).length : null;
  const lengths = view?.spans.map((s) => s.end - s.start) ?? [];

  return (
    <Card>
      <CardHeader>
        <CardTitle as="h2">Chunk playground</CardTitle>
        <CardDescription>
          Each colour is one passage, numbered where it starts. Outlined text belongs to two passages (the overlap);
          dimmed text belongs to none.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-5">
        <div className="flex flex-wrap items-end gap-3">
          <DocPagePicker
            id="chunk"
            docs={docs}
            docId={target.docId}
            page={target.page}
            pageCount={data?.doc_id === target.docId ? data.page_count : null}
            onChange={(docId, page) => setTarget({ docId, page })}
          />
          <div className="space-y-1">
            <span className="text-xs font-medium text-muted-foreground" id="chunk-parser">
              Parser
            </span>
            <Tabs value={parser} onValueChange={(v) => setParser(v as "pymupdf" | "docling")} aria-labelledby="chunk-parser">
              <TabsList>
                <TabsTrigger value="pymupdf">PyMuPDF + cleaning</TabsTrigger>
                <TabsTrigger value="docling">Docling + cleaning</TabsTrigger>
              </TabsList>
            </Tabs>
          </div>
        </div>

        <div className="space-y-2">
          <Tabs value={method} onValueChange={(v) => setMethod(v as ChunkMethod)}>
            <TabsList className="h-auto w-full flex-wrap sm:w-fit">
              {METHODS.map((m) => (
                <TabsTrigger key={m.id} value={m.id}>
                  {m.label}
                </TabsTrigger>
              ))}
            </TabsList>
          </Tabs>
          <p className="max-w-prose text-xs text-pretty text-muted-foreground">{info.explain(settled.size, settled.overlap)}</p>
        </div>

        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          <Slider id="chunk-size" label="Size" value={size} min={200} max={2000} step={50} unit="characters" onChange={setSize} />
          <Slider
            id="chunk-overlap"
            label="Overlap"
            value={Math.min(overlap, size - 50)}
            min={0}
            max={Math.min(500, size - 50)}
            step={10}
            unit="characters"
            onChange={setOverlap}
            disabled={!info.overlap}
            note={info.overlap ? undefined : "Not used by this chunker: it cuts at paragraphs or sentences, not at a count."}
          />
          {method === "semantic" && (
            <Slider
              id="chunk-percentile"
              label="Cut at this percentile of distances"
              value={percentile}
              min={50}
              max={99}
              step={1}
              display={(v) => `${v}th`}
              onChange={setPercentile}
              note="On GEP p19, 80 cuts right before “Risks to the outlook…”; 90 does not. An index uses one threshold for the whole document."
            />
          )}
        </div>

        {error ? (
          <ErrorState message={error.message} error={error} onRetry={() => mutate()} />
        ) : !data || !view || !fresh ? (
          <div role="status" aria-busy="true" className="space-y-2">
            <Skeleton className="h-6 w-80" />
            <Skeleton className="h-96 w-full" />
            {method === "semantic" && (
              <p className="text-xs text-muted-foreground">Embedding this page&apos;s sentences. The first time takes a few seconds.</p>
            )}
          </div>
        ) : (
          <div className={cn("space-y-4", isLoading && "opacity-70")}>
            <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-sm">
              <p className="tabular-nums">
                <span className="font-medium">{num(view.spans.length)} passages</span>
                <span className="text-muted-foreground"> · median {num(median(lengths))} characters</span>
              </p>
              {method === "fixed" && midWord !== null && (
                <p className="text-muted-foreground tabular-nums">
                  Starts mid-word: baseline {num(midWord)} of {num(data.baseline!.length)} · fixed{" "}
                  {num(data.passages.filter((p) => p.starts_mid_word).length)} of {num(data.passages.length)}
                </p>
              )}
              {isLoading && (
                <span className="flex items-center gap-1.5 text-xs text-muted-foreground" role="status">
                  <Loader2 className="size-3.5 animate-spin" aria-hidden />
                  Cutting...
                </span>
              )}
              {method === "fixed" && data.baseline && (
                <label htmlFor="chunk-baseline" className="ml-auto flex cursor-pointer items-center gap-2 text-xs">
                  <Checkbox id="chunk-baseline" checked={showBaseline} onCheckedChange={(on) => setShowBaseline(on === true)} />
                  Show the baseline&apos;s passages (✂ marks a mid-word start)
                </label>
              )}
            </div>

            {method === "semantic" && <SimilarityCurve data={data} />}

            {data.near_empty && (
              <p className="text-xs text-muted-foreground">
                This page has 200 characters or fewer, so no index cuts passages from it. The playground still shows
                how it would.
              </p>
            )}
            <div className="max-h-[36rem] overflow-y-auto rounded-lg bg-muted/20 p-3 ring-1 ring-foreground/10">
              <ChunkText
                text={data.text}
                spans={view.spans}
                notes={method === "structure" ? view.notes : undefined}
                flagged={method === "fixed" ? view.flagged : undefined}
              />
            </div>
          </div>
        )}

        {data && (
          <TechnicalDetails
            rows={{
              Chunker: `${data.method} · size ${data.size} · overlap ${info.overlap ? data.overlap : "not used"}${data.method === "semantic" ? ` · percentile ${data.percentile}` : ""}`,
              Parser: data.parser,
              "Page text": `${num(data.text.length)} characters, after cleaning`,
              "Embedding model": data.model ?? NO_DATA,
              "Section label": "Document: <title>, then Section: <section>, on the lines before the passage",
              "Cut in": `${data.ms} ms`,
              Indexes: "Fixed-size, structure-aware and semantic passages are index profiles; recursive and sentence are shown here only.",
            }}
          />
        )}
      </CardContent>
    </Card>
  );
}

function Slider({
  id,
  label,
  value,
  min,
  max,
  step,
  unit,
  display,
  onChange,
  disabled,
  note,
}: {
  id: string;
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  unit?: string;
  /** How the value reads beside the label; "<value> <unit>" by default. */
  display?: (v: number) => string;
  onChange: (v: number) => void;
  disabled?: boolean;
  note?: string;
}) {
  return (
    <div className={cn("min-w-0 space-y-1", disabled && "opacity-60")}>
      <div className="flex items-baseline justify-between gap-2">
        <label htmlFor={id} className="text-xs font-medium text-muted-foreground">
          {label}
        </label>
        <span className="text-xs tabular-nums">
          {disabled ? NO_DATA : display ? display(value) : `${num(value)} ${unit ?? ""}`.trim()}
        </span>
      </div>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full accent-primary"
      />
      {note && <p className="text-[11px] text-pretty text-muted-foreground">{note}</p>}
    </div>
  );
}
