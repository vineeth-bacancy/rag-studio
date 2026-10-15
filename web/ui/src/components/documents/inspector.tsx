import { useMemo, useState } from "react";
import useSWR from "swr";
import { AlertTriangle, ChevronLeft, ChevronRight, EyeOff, ScanSearch } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ErrorState } from "@/components/common/states";
import type { CleaningSummary, DocumentFile, PageInspection, ParserView } from "@/lib/api";
import { docTitle } from "@/lib/documents";
import { NO_DATA, num } from "@/lib/format";
import { cn } from "@/lib/utils";
import { CleaningReport } from "./cleaning-report";
import { MarkedText } from "./marked-text";
import { PageCanvas, type Overlay } from "./page-canvas";
import { RULE_ORDER, RULE_STYLES, ruleLabel } from "./rule-styles";

type View = "raw" | "clean" | "docling";

const VIEWS: { id: View; label: string; hint: string }[] = [
  { id: "raw", label: "PyMuPDF raw", hint: "Exactly what PyMuPDF extracted, with each defect a rule will fix highlighted." },
  { id: "clean", label: "PyMuPDF cleaned", hint: "After the guard and every cleaning rule. Hover a highlight to see the original." },
  { id: "docling", label: "Docling", hint: "Docling's layout-aware text, through the same rules. Tables come out as rows." },
];

const KIND_LABEL: Record<string, string> = { body: "Body text", toc: "Contents page", references: "Reference list" };

/**
 * The page inspector (spec §9.2): any page of any document, drawn as the PDF
 * draws it, next to what each parser read and what each cleaning rule changed.
 * Everything shown is the engine's own record of the change; nothing is
 * re-derived here.
 */
export function PageInspector({
  docs,
  docId,
  page,
  onChange,
}: {
  docs: DocumentFile[];
  docId: string;
  page: number;
  onChange: (docId: string, page: number) => void;
}) {
  const [view, setView] = useState<View>("clean");
  const [off, setOff] = useState<Set<string>>(new Set());
  const [layers, setLayers] = useState({ furniture: true, hidden: true, layout: true });
  const [pageDraft, setPageDraft] = useState<string | null>(null);

  const key = `/documents/${encodeURIComponent(docId)}/pages/${page}/inspect`;
  const { data, error, isLoading, mutate } = useSWR<PageInspection>(key, { keepPreviousData: true });
  const { data: summary } = useSWR<CleaningSummary>(`/documents/${encodeURIComponent(docId)}/cleaning`);
  const total = data?.doc_id === docId ? data.page_count : docs.find((d) => d.doc_id === docId)?.pages ?? null;

  const parser: ParserView | null = !data ? null : view === "docling" ? data.docling : data.pymupdf;
  const docTitles = Object.fromEntries(docs.map((d) => [d.doc_id, d.title ?? d.doc_id]));
  const findings = summary?.pymupdf?.pages ?? [];
  const findingItems = Object.fromEntries(
    findings.map((f) => [String(f.page), `Page ${f.page}: ${findingSummary(f.counts)}`]),
  );

  const go = (p: number) => {
    if (!total) return;
    setPageDraft(null);
    onChange(docId, Math.min(Math.max(1, p), total));
  };

  const overlays = useMemo<Overlay[]>(() => {
    if (!data) return [];
    const out: Overlay[] = [];
    const src = view === "docling" && data.docling ? data.docling : data.pymupdf;
    if (layers.furniture) {
      for (const f of src.furniture) {
        out.push({ box: f.box, className: "bg-zinc-500/10 ring-1 ring-zinc-500/70 ring-dashed", title: `Header or footer: ${f.text}` });
      }
    }
    if (layers.hidden) {
      for (const h of data.pymupdf.hidden) {
        out.push({ box: h.box, className: "bg-red-500/15 ring-2 ring-red-500", title: `Hidden text (${h.label}): ${h.text}` });
      }
    }
    if (layers.layout && view === "docling" && data.docling) {
      for (const it of data.docling.items) {
        if (it.label === "table" || it.label === "document_index") {
          out.push({ box: it.box, className: "ring-1 ring-sky-500/80 bg-sky-500/5", title: "Docling: table" });
        } else if (it.label === "picture") {
          out.push({ box: it.box, className: "ring-1 ring-violet-500/80 bg-violet-500/5", title: "Docling: picture" });
        }
      }
    }
    return out;
  }, [data, view, layers]);

  return (
    <Card>
      <CardHeader>
        <CardTitle as="h2" className="flex items-center gap-2">
          <ScanSearch className="size-4 text-muted-foreground" aria-hidden />
          Page inspector
        </CardTitle>
        <CardDescription>
          Any page as the PDF draws it, next to the text each parser extracted and what every cleaning rule changed.
          Hidden text is outlined in red: it was in the text layer, but nobody can see it on the page.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-5">
        <div className="flex flex-wrap items-end gap-3">
          <div className="min-w-0 space-y-1">
            <label htmlFor="inspect-doc" className="text-xs font-medium text-muted-foreground">
              Document
            </label>
            <Select value={docId} items={docTitles} onValueChange={(v) => v && onChange(v as string, 1)}>
              <SelectTrigger id="inspect-doc" className="w-full max-w-80 sm:w-80">
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
          <div className="space-y-1">
            <span className="text-xs font-medium text-muted-foreground" id="inspect-page-label">
              Page
            </span>
            <div className="flex items-center gap-1" role="group" aria-labelledby="inspect-page-label">
              <Button variant="outline" size="icon-sm" onClick={() => go(page - 1)} disabled={page <= 1} aria-label="Previous page">
                <ChevronLeft aria-hidden />
              </Button>
              <form
                onSubmit={(e) => {
                  e.preventDefault();
                  const n = Number(pageDraft);
                  if (Number.isInteger(n)) go(n);
                }}
              >
                <Input
                  aria-label="Page number"
                  inputMode="numeric"
                  className="h-7 w-14 text-center tabular-nums"
                  value={pageDraft ?? String(page)}
                  onChange={(e) => setPageDraft(e.target.value.replace(/\D/g, ""))}
                  onBlur={() => setPageDraft(null)}
                />
              </form>
              <span className="text-sm text-muted-foreground tabular-nums">of {num(total)}</span>
              <Button
                variant="outline"
                size="icon-sm"
                onClick={() => go(page + 1)}
                disabled={!total || page >= total}
                aria-label="Next page"
              >
                <ChevronRight aria-hidden />
              </Button>
            </div>
          </div>
          {findings.length > 0 && (
            <div className="min-w-0 space-y-1">
              <label htmlFor="inspect-jump" className="text-xs font-medium text-muted-foreground">
                Pages where a rule acted ({num(findings.length)})
              </label>
              <Select
                value={findingItems[String(page)] ? String(page) : null}
                items={findingItems}
                onValueChange={(v) => v && go(Number(v))}
              >
                <SelectTrigger id="inspect-jump" className="w-full max-w-72 sm:w-72">
                  <SelectValue placeholder="Jump to a page" />
                </SelectTrigger>
                <SelectContent>
                  {findings.map((f) => (
                    <SelectItem key={f.page} value={String(f.page)}>
                      {findingItems[String(f.page)]}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          )}
        </div>

        {summary && <CleaningReport summary={summary} />}

        {error && !data ? (
          <ErrorState message={error.message} error={error} onRetry={() => mutate()} />
        ) : !data || (isLoading && data.doc_id !== docId) ? (
          <div className="grid gap-4 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]" role="status" aria-busy="true">
            <Skeleton className="aspect-[8.5/11] w-full" />
            <div className="space-y-2">
              <Skeleton className="h-8 w-72" />
              <Skeleton className="h-96 w-full" />
              <p className="text-xs text-muted-foreground">
                Reading the page with both parsers. The first page of a document takes a few seconds: the guard draws
                every page twice to find hidden text.
              </p>
            </div>
          </div>
        ) : (
          <>
            <div className={cn("grid gap-5 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]", isLoading && "opacity-70")}>
              <div className="min-w-0 space-y-2">
                <PageCanvas
                  docId={data.doc_id}
                  page={data.page}
                  width={data.width}
                  height={data.height}
                  overlays={overlays}
                  alt={`Page ${data.page} of ${docTitle(data.doc_id, docs)}`}
                />
                <div className="flex flex-wrap items-center gap-1.5" role="group" aria-label="Boxes on the page">
                  <LayerToggle on={layers.furniture} swatch="bg-zinc-500" onClick={() => setLayers((l) => ({ ...l, furniture: !l.furniture }))}>
                    Headers and footers
                  </LayerToggle>
                  <LayerToggle on={layers.hidden} swatch="bg-red-500" onClick={() => setLayers((l) => ({ ...l, hidden: !l.hidden }))}>
                    Hidden text
                  </LayerToggle>
                  {view === "docling" && (
                    <LayerToggle on={layers.layout} swatch="bg-sky-500" onClick={() => setLayers((l) => ({ ...l, layout: !l.layout }))}>
                      Tables and pictures
                    </LayerToggle>
                  )}
                </div>
              </div>

              <div className="min-w-0 space-y-3">
                <Tabs value={view} onValueChange={(v) => setView(v as View)}>
                  <TabsList className="w-full sm:w-fit">
                    {VIEWS.map((v) => (
                      <TabsTrigger key={v.id} value={v.id} disabled={v.id === "docling" && !data.docling}>
                        {v.label}
                      </TabsTrigger>
                    ))}
                  </TabsList>
                </Tabs>
                <p className="text-xs text-muted-foreground">
                  {view === "docling" && !data.docling
                    ? "Docling has not converted this document yet. Run: ./rag ingest --profile docling-clean"
                    : VIEWS.find((v) => v.id === view)?.hint}
                </p>
                {parser && (
                  <RuleLegend
                    view={view}
                    parser={parser}
                    removed={data.pymupdf.hidden.length > 0 || parser.furniture.length > 0}
                    off={off}
                    onToggle={(r) => setOff(toggle(off, r))}
                  />
                )}
                <div className="max-h-[36rem] overflow-y-auto rounded-lg bg-muted/30 p-3 ring-1 ring-foreground/10">
                  {parser ? (
                    view === "raw" ? (
                      <MarkedText text={parser.text_raw} marks={parser.raw_marks} hidden={off} />
                    ) : (
                      <MarkedText text={parser.text} marks={parser.marks} hidden={off} />
                    )
                  ) : (
                    <p className="text-sm text-muted-foreground">No Docling output for this document yet.</p>
                  )}
                </div>
              </div>
            </div>
            <PageFacts data={data} parser={parser ?? data.pymupdf} />
          </>
        )}
      </CardContent>
    </Card>
  );
}

function toggle(set: Set<string>, rule: string): Set<string> {
  const next = new Set(set);
  if (next.has(rule)) next.delete(rule);
  else next.add(rule);
  return next;
}

function findingSummary(counts: Record<string, number>): string {
  return RULE_ORDER.filter((r) => counts[r])
    .map((r) => `${ruleLabel(r).toLowerCase()} ${counts[r]}`)
    .join(", ") || "hidden text";
}

function LayerToggle({
  on,
  swatch,
  onClick,
  children,
}: {
  on: boolean;
  swatch: string;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <Button variant="outline" size="xs" aria-pressed={on} onClick={onClick} className={cn(!on && "text-muted-foreground opacity-70")}>
      <span className={cn("size-2 rounded-full", on ? swatch : "bg-muted-foreground/40")} aria-hidden />
      {children}
    </Button>
  );
}

/** The rules that acted on this page, with counts; clicking one hides or shows its highlights. */
function RuleLegend({
  view,
  parser,
  removed,
  off,
  onToggle,
}: {
  view: View;
  parser: ParserView;
  /** Hidden text or headers and footers were taken out of this page. */
  removed: boolean;
  off: Set<string>;
  onToggle: (rule: string) => void;
}) {
  const marks = view === "raw" ? parser.raw_marks : parser.marks;
  const counts: Record<string, number> = {};
  for (const m of marks) counts[m.rule] = (counts[m.rule] ?? 0) + 1;
  const rules = RULE_ORDER.filter((r) => counts[r]);
  if (rules.length === 0) {
    return (
      <p className="text-xs text-muted-foreground">
        {removed
          ? "Nothing was changed in place. What was taken out of this page is listed below."
          : "No rule changed anything on this page."}
      </p>
    );
  }
  return (
    <div className="flex flex-wrap gap-1.5" role="group" aria-label="Highlights">
      {rules.map((r) => (
        <button
          key={r}
          type="button"
          aria-pressed={!off.has(r)}
          onClick={() => onToggle(r)}
          className={cn(
            "inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-[11px] ring-1 ring-foreground/15 transition-colors hover:bg-muted focus-visible:ring-[3px] focus-visible:ring-ring/50 focus-visible:outline-none",
            off.has(r) && "text-muted-foreground line-through opacity-70",
          )}
        >
          <span className={cn("size-2 rounded-full", RULE_STYLES[r].swatch)} aria-hidden />
          {ruleLabel(r)}
          <span className="text-muted-foreground tabular-nums">{counts[r]}</span>
        </button>
      ))}
    </div>
  );
}

/** What the page is (printed number, section, kind) and what was taken out of it. */
function PageFacts({ data, parser }: { data: PageInspection; parser: ParserView }) {
  const sections = [...new Set(parser.sections.map((s) => s.path.join(" > ")).filter(Boolean))];
  const kinds = [...new Set(parser.kinds.map((k) => KIND_LABEL[k.kind] ?? k.kind))];
  const examples = RULE_ORDER.filter((r) => r !== "furniture" && r !== "hidden")
    .map((r) => ({ rule: r, marks: parser.marks.filter((m) => m.rule === r) }))
    .filter((g) => g.marks.length > 0);

  return (
    <div className="grid gap-4 @container md:grid-cols-2">
      <section className="space-y-2" aria-labelledby="facts-heading">
        <h3 id="facts-heading" className="text-sm font-medium">About this page</h3>
        <dl className="grid grid-cols-[7rem_1fr] gap-x-3 gap-y-1.5 text-sm">
          <dt className="text-muted-foreground">PDF page</dt>
          <dd className="tabular-nums">{data.page}</dd>
          <dt className="text-muted-foreground">Printed page</dt>
          <dd className="tabular-nums">{data.label ?? NO_DATA}</dd>
          <dt className="text-muted-foreground">Section</dt>
          <dd className="min-w-0 space-y-0.5">{sections.length ? sections.map((s) => <p key={s} className="text-pretty">{s}</p>) : NO_DATA}</dd>
          <dt className="text-muted-foreground">Kind</dt>
          <dd>{kinds.join(", ") || NO_DATA}</dd>
          <dt className="text-muted-foreground">Passages</dt>
          <dd>{parser.near_empty ? "None: near-empty page (200 characters or fewer), skipped" : "Cut from the cleaned text"}</dd>
        </dl>
        {parser.flags.length > 0 && (
          <div className="space-y-1 rounded-md bg-amber-500/10 p-2.5 text-xs text-amber-800 ring-1 ring-amber-500/30 dark:text-amber-300">
            <p className="flex items-center gap-1.5 font-medium">
              <AlertTriangle className="size-3.5" aria-hidden />
              Instruction-like text on this page
            </p>
            {parser.flags.map((f, i) => (
              <p key={i}>
                {f.label}: &ldquo;{f.match}&rdquo;
              </p>
            ))}
          </div>
        )}
      </section>

      <section className="space-y-2" aria-labelledby="removed-heading">
        <h3 id="removed-heading" className="text-sm font-medium">Taken out of the text</h3>
        {data.pymupdf.hidden.length === 0 && parser.furniture.length === 0 ? (
          <p className="text-sm text-muted-foreground">Nothing: no hidden text, no headers or footers.</p>
        ) : (
          <ul className="space-y-1.5 text-xs">
            {data.pymupdf.hidden.map((h, i) => (
              <li key={`h${i}`} className="flex items-start gap-2">
                <EyeOff className="mt-0.5 size-3.5 shrink-0 text-red-600 dark:text-red-400" aria-hidden />
                <span className="min-w-0">
                  <span className="font-medium">Hidden text</span>
                  <span className="text-muted-foreground"> · {h.label}</span>
                  <span className="block truncate font-mono text-muted-foreground" title={h.text}>{h.text}</span>
                </span>
              </li>
            ))}
            {parser.furniture.map((f, i) => (
              <li key={`f${i}`} className="flex items-start gap-2">
                <span className="mt-1.5 size-2 shrink-0 rounded-full bg-zinc-500" aria-hidden />
                <span className="min-w-0">
                  <span className="font-medium">Header or footer</span>
                  <span className="text-muted-foreground"> · moved to metadata</span>
                  <span className="block truncate font-mono text-muted-foreground" title={f.text}>{f.text}</span>
                </span>
              </li>
            ))}
          </ul>
        )}
      </section>

      {examples.length > 0 && (
        <section className="space-y-2 md:col-span-2" aria-labelledby="changed-heading">
          <h3 id="changed-heading" className="text-sm font-medium">Changed in place</h3>
          <div className="grid gap-3 @lg:grid-cols-2">
            {examples.map((g) => (
              <div key={g.rule} className="space-y-1.5 rounded-lg p-2.5 ring-1 ring-foreground/10">
                <p className="flex items-center gap-1.5 text-xs font-medium">
                  <span className={cn("size-2 rounded-full", RULE_STYLES[g.rule].swatch)} aria-hidden />
                  {ruleLabel(g.rule)}
                  <Badge variant="secondary" className="h-4 px-1.5 text-[10px] tabular-nums">{g.marks.length}</Badge>
                </p>
                <ul className="space-y-0.5 font-mono text-[11px]">
                  {g.marks.slice(0, 5).map((m, i) => (
                    <li key={i} className="truncate">
                      <span className="text-muted-foreground">{(m.before || "").replace(/\n/g, "⏎") || "-"}</span>
                      <span className="px-1.5 text-muted-foreground" aria-label="became">→</span>
                      <span>{parser.text.slice(m.start, m.end) || "(removed)"}</span>
                    </li>
                  ))}
                  {g.marks.length > 5 && <li className="text-muted-foreground">and {g.marks.length - 5} more</li>}
                </ul>
              </div>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
