import { useState } from "react";
import { AlertTriangle, ChevronDown, FileText } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import type { DocumentFile, Source } from "@/lib/api";
import { docTitle } from "@/lib/documents";
import { NO_DATA } from "@/lib/format";
import { ranksText } from "@/lib/search";
import { cn } from "@/lib/utils";

const TYPE: Record<Source["type"], string> = { text: "Text", table: "Table", figure: "Figure" };
/** Content that matches many questions and answers few: tagged so it can be recognised. */
const KIND: Partial<Record<Source["kind"], { label: string; hint: string }>> = {
  toc: { label: "Contents page", hint: "A table of contents: it names every section, so it matches many questions." },
  references: { label: "Reference list", hint: "A bibliography: titles full of topic words, but no findings." },
};

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
  promptVersion,
  contextModel,
}: {
  sources: Source[];
  /** The model that wrote the passages' context lines, when the index has them. */
  contextModel?: string;
  /** The prompt the model got: v3 and later fence passages off as data. */
  promptVersion?: string;
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
          <SourceRow
            key={s.chunk_id}
            source={s}
            cited={cited === null ? null : cited.includes(s.n)}
            docs={docs}
            onOpen={onOpen}
            fenced={promptVersion === "v3" || promptVersion === "v4"}
            contextModel={contextModel}
          />
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
  fenced,
  contextModel,
}: {
  contextModel?: string;
  source: Source;
  fenced: boolean;
  cited: boolean | null;
  docs: DocumentFile[] | undefined;
  onOpen: (source: Source) => void;
}) {
  const [open, setOpen] = useState(false);
  // Ranks per list say something only when more than one list was searched,
  // or when "Limit per section" moved the passage.
  const ranks = s.ranks && (Object.keys(s.ranks).length > 1 || s.capped) ? ranksText(s.ranks) : "";
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
            <p className="text-xs text-muted-foreground tabular-nums">
              page {s.page}
              {s.page_label && s.page_label !== String(s.page) && <span> (printed {s.page_label})</span>}
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-1.5 text-[11px] text-muted-foreground">
            <span className="tabular-nums">rank {s.rank}</span>
            {ranks && <span className="tabular-nums">({ranks})</span>}
            <span aria-hidden>·</span>
            <span>Section {s.section_path ?? NO_DATA}</span>
            <Badge variant="outline" className="h-4 px-1.5 text-[10px] font-normal">{TYPE[s.type]}</Badge>
            {KIND[s.kind] && (
              <Badge variant="secondary" className="h-4 px-1.5 text-[10px] font-normal" title={KIND[s.kind]?.hint}>
                {KIND[s.kind]?.label}
              </Badge>
            )}
            {s.capped && (
              <Badge
                variant="outline"
                className="h-4 px-1.5 text-[10px] font-normal"
                title="Its section already had passages ranked above it, so Limit per section moved it down."
              >
                Moved down (limit per section)
              </Badge>
            )}
            {cited === false && <span>not cited</span>}
          </div>
          {s.flags?.length > 0 && (
            <p className="flex items-start gap-1.5 text-xs text-amber-700 dark:text-amber-400">
              <AlertTriangle className="mt-0.5 size-3.5 shrink-0" aria-hidden />
              <span>
                Instruction-like text: &ldquo;{s.flags[0].match}&rdquo;.{" "}
                {fenced
                  ? "The prompt told the model that passages are data, never instructions."
                  : "This prompt version does not fence passages off; prompt v3 does."}
              </span>
            </p>
          )}
          {s.context && (
            <p className="text-[11px] leading-relaxed text-muted-foreground italic" title="One or two sentences written for this passage and embedded with it, so search can find it. The answer model did not see it.">
              <span className="font-medium not-italic">Context line (written by {contextModel ?? "Remote AI"}):</span> {s.context}
            </p>
          )}
          {open && s.label && (
            <p
              className="font-mono text-[11px] leading-relaxed whitespace-pre-wrap text-muted-foreground"
              title="The section label: put in front of the passage before it was embedded, and shown to the model with it."
            >
              {s.label}
            </p>
          )}
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
