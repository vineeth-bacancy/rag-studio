import { Fragment } from "react";

import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import type { DocumentFile, Source } from "@/lib/api";
import { docTitle } from "@/lib/documents";
import { cn } from "@/lib/utils";

/**
 * The model's answer with each `[n]` turned into a button that opens the
 * cited page. The model only ever wrote the number; which document and page
 * it means comes from the sources list the engine sent (rule 5).
 *
 * A number the model invented (outside 1..k) stays visible and is marked,
 * rather than being dropped: hiding it would make the answer look better
 * grounded than it is.
 *
 * A saved answer (a reopened conversation) keeps where each passage came
 * from but not its text, so its citations name the page and open nothing.
 * Before M8 they had no sources at all and every [n] was marked as invented.
 */
const CITE = /\[([\d,\s]+)\]/g;

export function AnswerText({
  text,
  sources,
  saved,
  docs,
  onCite,
  streaming = false,
}: {
  text: string;
  sources: Source[];
  /** A saved answer's passages: number, document and page only. */
  saved?: { n: number; doc_id: string; page: number }[];
  docs: DocumentFile[] | undefined;
  onCite: (source: Source) => void;
  streaming?: boolean;
}) {
  const given = saved && sources.length === 0 ? saved.length : sources.length;
  const parts: React.ReactNode[] = [];
  let last = 0;
  for (const m of text.matchAll(CITE)) {
    const at = m.index ?? 0;
    parts.push(text.slice(last, at));
    const nums = m[1].split(/[,\s]+/).filter(Boolean).map(Number);
    parts.push(
      <span key={`c${at}`} className="whitespace-nowrap">
        {nums.map((n, i) => {
          const src = sources.find((s) => s.n === n);
          const kept = src ? undefined : saved?.find((s) => s.n === n);
          return (
            <Fragment key={i}>
              {src ? (
                <Tooltip>
                  <TooltipTrigger
                    render={
                      <button
                        type="button"
                        onClick={() => onCite(src)}
                        className="mx-0.5 inline-flex h-5 min-w-5 items-center justify-center rounded-md bg-chart-1/10 px-1 align-text-top text-[11px] font-semibold text-chart-1 tabular-nums transition-shadow hover:ring-1 hover:ring-chart-1/50 focus-visible:ring-[3px] focus-visible:ring-ring/50 focus-visible:outline-none"
                        aria-label={`Source ${n}: ${docTitle(src.doc_id, docs)}, page ${src.page}`}
                      />
                    }
                  >
                    {n}
                  </TooltipTrigger>
                  <TooltipContent>
                    {docTitle(src.doc_id, docs)} · page {src.page}
                  </TooltipContent>
                </Tooltip>
              ) : kept ? (
                <span
                  className="mx-0.5 inline-flex h-5 min-w-5 items-center justify-center rounded-md bg-muted px-1 align-text-top text-[11px] font-semibold text-foreground tabular-nums"
                  title={`${docTitle(kept.doc_id, docs)} · page ${kept.page}. A saved answer keeps where each passage came from, not its text.`}
                >
                  {n}
                  <span className="sr-only">
                    : {docTitle(kept.doc_id, docs)}, page {kept.page}
                  </span>
                </span>
              ) : (
                <span
                  className="mx-0.5 inline-flex h-5 min-w-5 items-center justify-center rounded-md px-1 align-text-top text-[11px] font-semibold text-red-700 tabular-nums ring-1 ring-red-500/40 dark:text-red-400"
                  title={`The model cited [${n}], but it was only given ${given} passages.`}
                >
                  {n}?
                </span>
              )}
            </Fragment>
          );
        })}
      </span>,
    );
    last = at + m[0].length;
  }
  parts.push(text.slice(last));

  return (
    <p className={cn("leading-7 whitespace-pre-wrap text-pretty", streaming && "after:ml-0.5 after:inline-block after:h-4 after:w-1.5 after:translate-y-0.5 after:animate-pulse after:rounded-sm after:bg-foreground/60 after:content-['']")}>
      {parts}
    </p>
  );
}
