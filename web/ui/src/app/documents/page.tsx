"use client";

import { useRef, useState } from "react";
import useSWR from "swr";
import { FileText, FileUp, ScanSearch } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { PageInspector } from "@/components/documents/inspector";
import { PageHeader } from "@/components/common/page-header";
import { CardSkeleton, EmptyState, ErrorState } from "@/components/common/states";
import type { DocumentFile } from "@/lib/api";
import { Tile, TileGrid } from "@/components/common/tiles";
import { bytes, num } from "@/lib/format";

export default function DocumentsPage() {
  const { data, error, isLoading, mutate } = useSWR<{ documents: DocumentFile[] }>("/documents");
  const [inspect, setInspect] = useState<{ docId: string; page: number } | null>(null);
  const inspectorRef = useRef<HTMLDivElement>(null);
  const docs = data?.documents ?? [];
  // The inspector opens on the first document until one is chosen.
  const target = inspect ?? (docs[0] ? { docId: docs[0].doc_id, page: 1 } : null);

  return (
    <div className="mx-auto max-w-6xl space-y-6">
      <PageHeader href="/documents" />

      {error ? (
        <ErrorState message={error.message} error={error} onRetry={() => mutate()} />
      ) : isLoading || !data ? (
        <CardSkeleton height="h-28" />
      ) : data.documents.length === 0 ? (
        <EmptyState
          icon={FileUp}
          title="No documents yet"
          description="Add PDF reports to the library to ask questions about them."
          hint="Upload from the browser arrives in M8. Until then, copy PDFs into data/pdfs."
        />
      ) : (
        <ul className="grid gap-4 md:grid-cols-2 xl:grid-cols-3" aria-label="Documents">
          {data.documents.map((d) => (
            <li key={d.doc_id}>
              <Card className="h-full">
                <CardContent className="space-y-4">
                  <div className="flex items-start gap-3">
                    <div className="rounded-lg bg-muted p-2.5">
                      <FileText className="size-5 text-muted-foreground" aria-hidden />
                    </div>
                    <div className="min-w-0 space-y-0.5">
                      <h2 className="text-sm leading-snug font-medium text-pretty">{d.title ?? d.doc_id}</h2>
                      <p className="truncate font-mono text-[11px] text-muted-foreground">{d.doc_id}</p>
                    </div>
                  </div>
                  {/* The tile row's separators are its own background showing
                      through a 1px gap, so it sits on a plain wrapper. */}
                  <TileGrid className="grid-cols-3">
                    <Tile label="Pages" value={num(d.pages)} />
                    <Tile label="Passages" value={num(d.passages)} />
                    <Tile label="Size" value={bytes(d.bytes)} />
                  </TileGrid>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => {
                      setInspect({ docId: d.doc_id, page: 1 });
                      inspectorRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
                    }}
                  >
                    <ScanSearch aria-hidden />
                    Inspect pages
                  </Button>
                  <p className="text-xs text-muted-foreground">
                    {d.indexed
                      ? d.near_empty_pages && d.near_empty_pages.length > 0
                        ? `${d.near_empty_pages.length} near-empty pages skipped (covers, dividers, blank backs): ${d.near_empty_pages.join(", ")}.`
                        : "No near-empty pages: every page has passages."
                      : "Not in the search index yet. Build it with ./rag ingest."}
                  </p>
                </CardContent>
              </Card>
            </li>
          ))}
        </ul>
      )}

      {target && (
        <div ref={inspectorRef} className="scroll-mt-20">
          <PageInspector
            docs={docs}
            docId={target.docId}
            page={target.page}
            onChange={(docId, page) => setInspect({ docId, page })}
          />
        </div>
      )}
    </div>
  );
}
