import type { CitationMark, FactCheck } from "@/lib/api";
import { cn } from "@/lib/utils";

type Decoration = { start: number; end: number; kind: "cite"; mark: CitationMark } | { start: number; end: number; kind: "fact" };

/**
 * An answer as the engine read it: each `[n]` group the citation code found
 * (a number with no passage behind it in red), and each expected fact where
 * the scorecard's rule matched it. Positions come from the engine; this only
 * draws them.
 */
export function MarkedAnswer({
  text,
  marks = [],
  facts = [],
  className,
}: {
  text: string;
  marks?: CitationMark[];
  facts?: FactCheck[];
  className?: string;
}) {
  const decorations: Decoration[] = [
    ...marks.map((m) => ({ start: m.start, end: m.end, kind: "cite" as const, mark: m })),
    ...facts.flatMap((f) => f.spans.map(([start, end]) => ({ start, end, kind: "fact" as const }))),
  ].sort((a, b) => a.start - b.start || b.end - a.end);

  const out: React.ReactNode[] = [];
  let at = 0;
  for (const d of decorations) {
    if (d.start < at) continue; // overlaps one already drawn
    if (d.start > at) out.push(text.slice(at, d.start));
    if (d.kind === "cite") {
      out.push(
        <span
          key={`c${d.start}`}
          className={cn(
            "mx-0.5 inline-flex items-center rounded px-1 font-mono text-[11px] font-semibold tabular-nums ring-1",
            d.mark.invalid.length
              ? "bg-red-500/10 text-red-700 ring-red-500/50 dark:text-red-400"
              : "bg-sky-500/10 text-sky-800 ring-sky-500/40 dark:text-sky-300",
          )}
          title={
            d.mark.invalid.length
              ? `No passage ${d.mark.invalid.join(", ")} was shown: reported, never dropped or fixed`
              : `Points at passage ${d.mark.valid.join(", ")}`
          }
        >
          {text.slice(d.start, d.end)}
        </span>,
      );
    } else {
      out.push(
        <mark
          key={`f${d.start}`}
          className="rounded-sm bg-emerald-500/15 px-0.5 text-inherit underline decoration-emerald-600 decoration-2 underline-offset-2 dark:decoration-emerald-400"
        >
          {text.slice(d.start, d.end)}
        </mark>,
      );
    }
    at = d.end;
  }
  if (at < text.length) out.push(text.slice(at));
  return <p className={cn("text-sm leading-relaxed text-pretty whitespace-pre-wrap", className)}>{out}</p>;
}
