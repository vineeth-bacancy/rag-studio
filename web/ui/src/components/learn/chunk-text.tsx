import { cn } from "@/lib/utils";

/**
 * One colour per passage, cycling. Never green: green means Healthy and Pass
 * (docs/product-language.md). Each passage also carries its number, so the
 * colour never stands alone.
 */
export const PASSAGE_COLOURS = [
  { bg: "bg-sky-500/15", chip: "bg-sky-700 text-white dark:bg-sky-500 dark:text-black" },
  { bg: "bg-violet-500/15", chip: "bg-violet-600 text-white dark:bg-violet-400 dark:text-black" },
  { bg: "bg-amber-500/20", chip: "bg-amber-700 text-white dark:bg-amber-500 dark:text-black" },
  { bg: "bg-teal-500/15", chip: "bg-teal-700 text-white dark:bg-teal-500 dark:text-black" },
  { bg: "bg-fuchsia-500/15", chip: "bg-fuchsia-600 text-white dark:bg-fuchsia-400 dark:text-black" },
];

export interface Span {
  start: number;
  end: number;
}

/** Text pieces between every passage edge, each with the passages covering it. */
function segments(length: number, spans: Span[]): { a: number; b: number; cover: number[] }[] {
  const points = [...new Set([0, length, ...spans.flatMap((s) => [s.start, s.end])])].sort((x, y) => x - y);
  const out = [];
  for (let i = 0; i < points.length - 1; i++) {
    const a = points[i], b = points[i + 1];
    out.push({ a, b, cover: spans.flatMap((s, j) => (s.start <= a && s.end >= b ? [j] : [])) });
  }
  return out;
}

/**
 * The page text with each passage tinted and numbered where it starts. Text
 * in two passages at once (the overlap) is outlined; text in none (spaces
 * between passages, a heading the section label already carries) is dimmed.
 * `notes` puts a short line before a passage, such as its section label.
 */
export function ChunkText({
  text,
  spans,
  notes,
  flagged,
  className,
}: {
  text: string;
  spans: Span[];
  notes?: (string | null)[];
  /** Passages to mark at their start, e.g. those that start mid-word. */
  flagged?: Set<number>;
  className?: string;
}) {
  const parts: React.ReactNode[] = [];
  const started = new Set<number>();
  for (const seg of segments(text.length, spans)) {
    spans.forEach((s, j) => {
      if (s.start === seg.a && !started.has(j)) {
        started.add(j);
        const c = PASSAGE_COLOURS[j % PASSAGE_COLOURS.length];
        if (notes?.[j]) {
          parts.push(
            <span key={`n${j}`} className="my-1 block font-sans text-[10.5px] whitespace-pre-wrap text-muted-foreground">
              {notes[j]}
            </span>,
          );
        }
        parts.push(
          <span
            key={`c${j}`}
            className={cn("mr-1 inline-block rounded px-1 font-sans text-[10px] leading-4 font-semibold tabular-nums", c.chip)}
            title={`Passage ${j + 1}: ${s.end - s.start} characters${flagged?.has(j) ? ", starts mid-word" : ""}`}
          >
            <span className="sr-only">Passage </span>
            {j + 1}
            {flagged?.has(j) && <span aria-label=", starts mid-word"> ✂</span>}
          </span>,
        );
      }
    });
    const piece = text.slice(seg.a, seg.b);
    if (!piece) continue;
    if (seg.cover.length === 0) {
      parts.push(
        <span key={`s${seg.a}`} className="text-muted-foreground/60">
          {piece}
        </span>,
      );
    } else if (seg.cover.length === 1) {
      parts.push(
        <span key={`s${seg.a}`} className={PASSAGE_COLOURS[seg.cover[0] % PASSAGE_COLOURS.length].bg}>
          {piece}
        </span>,
      );
    } else {
      parts.push(
        <span
          key={`s${seg.a}`}
          className="bg-foreground/10 underline decoration-foreground/50 decoration-dotted underline-offset-2"
          title={`In ${seg.cover.length} passages: ${seg.cover.map((j) => j + 1).join(" and ")} (overlap)`}
        >
          {piece}
        </span>,
      );
    }
  }
  return (
    <div className={cn("font-mono text-[11.5px] leading-relaxed break-words whitespace-pre-wrap", className)}>
      {parts.length ? parts : <span className="text-muted-foreground">No text on this page.</span>}
    </div>
  );
}
