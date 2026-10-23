import { useState } from "react";
import useSWR from "swr";
import { FileText } from "lucide-react";

import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Skeleton } from "@/components/ui/skeleton";
import { TechnicalDetails } from "@/components/common/technical-details";
import { boxesPath, type DocumentFile, type PassageBoxes, type Source } from "@/lib/api";
import { docTitle, pageImageUrl } from "@/lib/documents";

/**
 * The cited page as it looks in the PDF, with the passage the model was
 * shown outlined on it. This is the check a reader can make with their own
 * eyes: does the page really say what the answer claims?
 *
 * The outline comes from the engine: for the baseline index, the passage's
 * character span mapped to PyMuPDF's word boxes; for a cleaned index, the
 * passage's words aligned with the page's (cleaning changed the text), or
 * Docling's layout boxes. Nothing here is guessed from the text.
 */
export function PagePreview({
  source,
  profile,
  docs,
  open,
  onOpenChange,
}: {
  source: Source | null;
  /** The index the source came from. */
  profile: string;
  docs: DocumentFile[] | undefined;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[calc(100dvh-2rem)] gap-3 overflow-y-auto sm:max-w-3xl">
        {source && <PreviewBody key={source.chunk_id} source={source} profile={profile} docs={docs} />}
      </DialogContent>
    </Dialog>
  );
}

function PreviewBody({ source, profile, docs }: { source: Source; profile: string; docs: DocumentFile[] | undefined }) {
  const [loaded, setLoaded] = useState(false);
  const [failed, setFailed] = useState(false);
  const { data: boxes } = useSWR<PassageBoxes>(boxesPath(source, profile));
  // Baseline: true/false. Cleaned indexes: the share of the passage's words
  // found in place on the page; under half means the outline can't be trusted.
  const unaligned = boxes ? (typeof boxes.aligned === "number" ? boxes.aligned < 0.5 : !boxes.aligned) : false;
  const ratio = boxes ? boxes.height / boxes.width : 11 / 8.5;

  return (
    <>
      <DialogHeader className="pr-8">
        <DialogTitle className="flex items-center gap-2">
          <span className="inline-flex size-6 shrink-0 items-center justify-center rounded-md bg-chart-1/10 text-xs font-semibold text-chart-1 tabular-nums">
            {source.n}
          </span>
          <span className="truncate">{docTitle(source.doc_id, docs)}</span>
        </DialogTitle>
        <DialogDescription>
          Page {source.page} of the PDF{source.page_label && source.page_label !== String(source.page) ? ` (printed ${source.page_label})` : ""}. The outlined lines are the passage the model was given as [{source.n}].
        </DialogDescription>
      </DialogHeader>

      <div className="grid gap-4 md:grid-cols-[minmax(0,1fr)_16rem]">
        <div
          className="relative w-full overflow-hidden rounded-lg bg-white ring-1 ring-foreground/10"
          style={{ aspectRatio: `1 / ${ratio}` }}
        >
          {!loaded && !failed && <Skeleton className="absolute inset-0 rounded-none" />}
          {failed ? (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 p-6 text-center text-sm text-muted-foreground">
              <FileText className="size-6" aria-hidden />
              Couldn&apos;t render this page. The passage text is beside it.
            </div>
          ) : (
            // eslint-disable-next-line @next/next/no-img-element -- a PNG the API renders on demand; next/image would proxy and resize it again
            <img
              src={pageImageUrl(source.doc_id, source.page)}
              alt={`Page ${source.page} of ${docTitle(source.doc_id, docs)}`}
              className="absolute inset-0 size-full object-contain"
              onLoad={() => setLoaded(true)}
              onError={() => setFailed(true)}
            />
          )}
          {loaded &&
            boxes?.boxes.map((b, i) => (
              <span
                key={i}
                aria-hidden
                className="pointer-events-none absolute rounded-[2px] bg-chart-1/20 ring-1 ring-chart-1/70"
                style={{
                  left: `${b.x0 * 100}%`,
                  top: `${b.y0 * 100}%`,
                  width: `${(b.x1 - b.x0) * 100}%`,
                  height: `${(b.y1 - b.y0) * 100}%`,
                }}
              />
            ))}
        </div>

        <div className="min-w-0 space-y-3">
          <div className="space-y-1">
            <p className="text-xs font-medium text-muted-foreground">Passage [{source.n}] as the model read it</p>
            <p className="max-h-80 overflow-y-auto rounded-md bg-muted/50 p-3 text-xs leading-relaxed whitespace-pre-wrap">
              {source.text}
            </p>
          </div>
          {unaligned && (
            <p className="text-xs text-muted-foreground">
              The outline isn&apos;t available for this page: its words couldn&apos;t be matched to the text.
            </p>
          )}
          <TechnicalDetails
            rows={{
              Passage: source.chunk_id,
              Rank: `rank ${source.rank}`,
              Score: source.score.toFixed(3),
              Type: source.type,
              Characters: `${source.char_start}-${source.char_end}`,
              Index: profile,
              Outline:
                boxes && typeof boxes.aligned === "number"
                  ? `${Math.round(boxes.aligned * 100)}% of the passage's words matched on the page`
                  : "exact character span",
            }}
          />
        </div>
      </div>
    </>
  );
}
