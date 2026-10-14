import { useState } from "react";
import { ChevronDown, FileText } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import type { DocumentFile, Source } from "@/lib/api";
import { docTitle } from "@/lib/documents";
import { NO_DATA } from "@/lib/format";
import { cn } from "@/lib/utils";

const TYPE: Record<Source["type"], string> = { text: "Text", table: "Table", figure: "Figure" };

/**
 * Every passage the model was given, in the order it saw them, and whether
 * the answer cited it. An uncited source is still shown: what the model had
 * to work with matters as much as what it chose to use.
 */
export function Sources({
  sources,
  cited,
  docs,
  onOpen,
}: {
  sources: Source[];
  /** null while the answer is still being written: not known yet. */
  cited: number[] | null;
  docs: DocumentFile[] | undefined;
  onOpen: (source: Source) => void;
}) {
  return (
    <section aria-label="Sources" className="space-y-2">
      <h3 className="text-xs font-medium tracking-wide text-muted-foreground uppercase">
        Sources <span className="tabular-nums normal-case">({sources.length} passages)</span>
      </h3>
      <ol className="space-y-2">
        {sources.map((s) => (
          <SourceRow key={s.chunk_id} source={s} cited={cited === null ? null : cited.includes(s.n)} docs={docs} onOpen={onOpen} />
        ))}
      </ol>
    </section>
  );
}

function SourceRow({
  source: s,
  cited,
  docs,
  onOpen,
}: {
  source: Source;
  cited: boolean | null;
  docs: DocumentFile[] | undefined;
  onOpen: (source: Source) => void;
}) {
  const [open, setOpen] = useState(false);
  return (
    <li className={cn("rounded-lg px-3 py-2.5 ring-1 ring-foreground/10", cited === false ? "bg-muted/30" : "bg-card")}>
      <div className="flex items-start gap-3">
        <span
          className={cn(
            "mt-0.5 inline-flex size-6 shrink-0 items-center justify-center rounded-md text-xs font-semibold tabular-nums",
            cited ? "bg-chart-1/15 text-chart-1" : "bg-muted text-muted-foreground",
          )}
          aria-label={`Passage ${s.n}${cited === null ? "" : cited ? ", cited" : ", not cited"}`}
        >
          {s.n}
        </span>
        <div className="min-w-0 flex-1 space-y-1">
          <div className="flex flex-wrap items-baseline gap-x-2 gap-y-0.5">
            <p className="min-w-0 truncate text-sm font-medium">{docTitle(s.doc_id, docs)}</p>
            <p className="text-xs text-muted-foreground tabular-nums">page {s.page}</p>
          </div>
          <div className="flex flex-wrap items-center gap-1.5 text-[11px] text-muted-foreground">
            <span className="tabular-nums">rank {s.rank}</span>
            <span aria-hidden>·</span>
            <span>Section {s.section_path ?? NO_DATA}</span>
            <Badge variant="outline" className="h-4 px-1.5 text-[10px] font-normal">{TYPE[s.type]}</Badge>
            {cited === false && <span>not cited</span>}
          </div>
          <p className={cn("text-xs leading-relaxed text-muted-foreground", !open && "line-clamp-2")}>{s.text}</p>
        </div>
      </div>
      <div className="mt-1.5 flex flex-wrap items-center justify-end gap-1 pl-9">
        <Button variant="ghost" size="xs" onClick={() => setOpen((v) => !v)} aria-expanded={open}>
          <ChevronDown className={cn("transition-transform", open && "rotate-180")} aria-hidden />
          {open ? "Less" : "Full passage"}
        </Button>
        <Button variant="outline" size="xs" onClick={() => onOpen(s)}>
          <FileText aria-hidden />
          Open page {s.page}
        </Button>
      </div>
    </li>
  );
}
