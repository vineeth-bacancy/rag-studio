import { useCallback, useLayoutEffect, useRef, useState } from "react";
import { BookOpen, Target } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import type { DocumentFile, LabHit, LabList, SearchLab, SearchMethod } from "@/lib/api";
import { docTitle } from "@/lib/documents";
import { NO_DATA } from "@/lib/format";
import { cn } from "@/lib/utils";

/**
 * Hybrid sits in the middle: it is made of its two neighbours, so every
 * rank-movement line runs inward, from dense search on the left and from BM25
 * on the right, into the fused list.
 */
const COLUMNS: { id: SearchMethod; title: string; how: string }[] = [
  { id: "dense", title: "Dense", how: "By meaning: the nearest vectors" },
  { id: "hybrid", title: "Hybrid", how: "Both lists, fused by rank (RRF)" },
  { id: "bm25", title: "BM25", how: "By words: the word index" },
];

const KIND: Record<string, string> = { toc: "Contents page", references: "Reference list" };

interface Line {
  id: string;
  d: string;
  answer: boolean;
}

/**
 * The three result lists side by side, with a line joining each passage that
 * appears in two neighbouring lists. Lines are drawn from the rows' measured
 * positions, so they follow any width; when the columns stack (narrow
 * screens) there are no lines, and each hybrid row says where it came from.
 */
export function SearchColumns({
  data,
  docs,
  showScores,
}: {
  data: SearchLab;
  docs: DocumentFile[] | undefined;
  showScores: boolean;
}) {
  const box = useRef<HTMLDivElement>(null);
  const rows = useRef(new Map<string, HTMLElement>());
  const [lines, setLines] = useState<Line[]>([]);
  const [hover, setHover] = useState<string | null>(null);

  const register = useCallback((key: string) => (el: HTMLElement | null) => {
    if (el) rows.current.set(key, el);
    else rows.current.delete(key);
  }, []);

  useLayoutEffect(() => {
    const root = box.current;
    if (!root) return;
    const measure = () => {
      const origin = root.getBoundingClientRect();
      const hybrid = data.lists.hybrid?.hits ?? [];
      const out: Line[] = [];
      for (const side of ["dense", "bm25"] as const) {
        for (const h of data.lists[side]?.hits ?? []) {
          if (!hybrid.some((x) => x.chunk_id === h.chunk_id)) continue;
          const a = rows.current.get(`${side}:${h.chunk_id}`)?.getBoundingClientRect();
          const b = rows.current.get(`hybrid:${h.chunk_id}`)?.getBoundingClientRect();
          if (!a || !b) continue;
          // Side by side only: stacked columns get no lines.
          const leftToRight = side === "dense";
          if (leftToRight ? a.right > b.left : a.left < b.right) continue;
          const x1 = (leftToRight ? a.right : a.left) - origin.left;
          const x2 = (leftToRight ? b.left : b.right) - origin.left;
          const y1 = a.top + a.height / 2 - origin.top;
          const y2 = b.top + b.height / 2 - origin.top;
          const mid = (x1 + x2) / 2;
          out.push({ id: `${side}:${h.chunk_id}`, d: `M${x1},${y1} C${mid},${y1} ${mid},${y2} ${x2},${y2}`, answer: h.answer_page });
        }
      }
      setLines(out);
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(root);
    return () => ro.disconnect();
  }, [data, showScores]);

  return (
    <div ref={box} className="relative @container">
      {/* Side by side from a 56rem container (1280 px with the sidebar open);
          the Reranked placeholder joins as a fourth column only when there is room. */}
      <div className="grid gap-4 @4xl:grid-cols-3 @4xl:gap-x-8 @6xl:grid-cols-[repeat(3,minmax(0,1fr))_9rem] @6xl:gap-x-10">
        {COLUMNS.map((c) => (
          <Column
            key={c.id}
            id={c.id}
            title={c.title}
            how={c.how}
            list={data.lists[c.id]}
            data={data}
            docs={docs}
            showScores={showScores}
            hover={hover}
            onHover={setHover}
            register={register}
          />
        ))}
        <section aria-label="Reranked" className="space-y-2 rounded-lg border border-dashed p-3 @4xl:col-span-3 @6xl:col-span-1">
          <h3 className="text-sm font-medium">Reranked</h3>
          <p className="text-xs text-pretty text-muted-foreground">
            A slower model reads each passage with the question and reorders the top results. Arrives in M5.
          </p>
        </section>
      </div>
      <svg className="pointer-events-none absolute inset-0 hidden size-full overflow-visible @4xl:block" aria-hidden>
        {lines.map((l) => {
          const lit = hover !== null && l.id.endsWith(`:${hover}`);
          return (
            <path
              key={l.id}
              d={l.d}
              fill="none"
              className={cn(
                l.answer ? "text-sky-600 dark:text-sky-400" : "text-muted-foreground",
                hover !== null && !lit && "opacity-20",
              )}
              stroke="currentColor"
              strokeWidth={lit ? 2.5 : 1.25}
              strokeOpacity={l.answer || lit ? 0.9 : 0.45}
            />
          );
        })}
      </svg>
    </div>
  );
}

function Column({
  id,
  title,
  how,
  list,
  data,
  docs,
  showScores,
  hover,
  onHover,
  register,
}: {
  id: SearchMethod;
  title: string;
  how: string;
  list: LabList | null;
  data: SearchLab;
  docs: DocumentFile[] | undefined;
  showScores: boolean;
  hover: string | null;
  onHover: (chunk: string | null) => void;
  register: (key: string) => (el: HTMLElement | null) => void;
}) {
  return (
    <section aria-label={`${title} results`} className="min-w-0 space-y-2">
      <div className="space-y-0.5">
        <h3 className="text-sm font-medium">{title}</h3>
        <p className="text-xs text-muted-foreground">{how}</p>
        {list && <FirstAnswer data={data} rank={list.first_answer_rank} />}
      </div>
      {list === null ? (
        <p className="rounded-lg bg-muted/30 p-3 text-xs text-pretty text-muted-foreground ring-1 ring-foreground/10">
          This index has no word index yet, so BM25 and hybrid search are unavailable. Add it with{" "}
          <code className="font-mono">./rag ingest --profile {data.profile}</code>
        </p>
      ) : list.hits.length === 0 ? (
        <p className="text-xs text-muted-foreground">No passage shares a word with this question.</p>
      ) : (
        <ol className="space-y-1.5">
          {list.hits.map((h) => (
            <HitRow
              key={h.chunk_id}
              ref={register(`${id}:${h.chunk_id}`)}
              hit={h}
              column={id}
              docs={docs}
              showScores={showScores}
              lit={hover === h.chunk_id}
              dim={hover !== null && hover !== h.chunk_id}
              onHover={onHover}
            />
          ))}
        </ol>
      )}
    </section>
  );
}

function FirstAnswer({ data, rank }: { data: SearchLab; rank: number | null }) {
  if (!data.test_question) {
    return <p className="text-[11px] text-muted-foreground">Answer pages unknown: not a test question.</p>;
  }
  return (
    <p className="text-[11px] tabular-nums">
      <span className="text-muted-foreground">First answer page: </span>
      {rank === null ? (
        <span>not in the top 50</span>
      ) : (
        <span className={cn("font-medium", rank <= 5 ? "text-sky-700 dark:text-sky-300" : undefined)}>rank {rank}</span>
      )}
      <span className="text-muted-foreground"> (searching 50 deep)</span>
    </p>
  );
}

function HitRow({
  ref,
  hit: h,
  column,
  docs,
  showScores,
  lit,
  dim,
  onHover,
}: {
  ref: (el: HTMLElement | null) => void;
  hit: LabHit;
  column: SearchMethod;
  docs: DocumentFile[] | undefined;
  showScores: boolean;
  lit: boolean;
  dim: boolean;
  onHover: (chunk: string | null) => void;
}) {
  // A hybrid row says which lists it came from: the lines show it on wide
  // screens, these words on every screen.
  const from = column === "hybrid" && h.ranks
    ? [h.ranks.dense ? `dense ${h.ranks.dense}` : "not in dense", h.ranks.bm25 ? `BM25 ${h.ranks.bm25}` : "not in BM25"].join(" · ")
    : null;
  return (
    <li
      ref={ref}
      tabIndex={0}
      onMouseEnter={() => onHover(h.chunk_id)}
      onMouseLeave={() => onHover(null)}
      onFocus={() => onHover(h.chunk_id)}
      onBlur={() => onHover(null)}
      className={cn(
        "space-y-1 rounded-lg bg-card px-2.5 py-2 ring-1 ring-foreground/10 transition-opacity outline-none focus-visible:ring-[3px] focus-visible:ring-ring/50",
        h.answer_page && "ring-sky-500/60 dark:ring-sky-400/60",
        lit && "ring-2 ring-foreground/40",
        dim && "opacity-50",
      )}
    >
      <div className="flex items-baseline gap-2">
        <span className="text-xs font-semibold tabular-nums">{h.rank}</span>
        <span className="min-w-0 truncate text-xs" title={docTitle(h.doc_id, docs)}>
          {docTitle(h.doc_id, docs)}
        </span>
        <span className="ml-auto shrink-0 text-[11px] text-muted-foreground tabular-nums">
          p. {h.page}
          {h.page_label && h.page_label !== String(h.page) ? ` (${h.page_label})` : ""}
        </span>
      </div>
      <div className="flex flex-wrap items-center gap-1">
        {h.answer_page && (
          <Badge className="h-4 gap-1 bg-sky-600 px-1.5 text-[10px] font-normal text-white dark:bg-sky-500 dark:text-black">
            <Target className="size-2.5" aria-hidden />
            Answer page
          </Badge>
        )}
        {KIND[h.kind] && (
          <Badge variant="secondary" className="h-4 gap-1 px-1.5 text-[10px] font-normal">
            <BookOpen className="size-2.5" aria-hidden />
            {KIND[h.kind]}
          </Badge>
        )}
        {h.capped && (
          <Badge variant="outline" className="h-4 px-1.5 text-[10px] font-normal" title="Limit per section moved it down">
            Moved down (limit per section)
          </Badge>
        )}
        {from && <span className="text-[10.5px] text-muted-foreground tabular-nums">{from}</span>}
        {showScores && (
          <span className="ml-auto font-mono text-[10.5px] text-muted-foreground tabular-nums">
            {h.score.toFixed(3)}
          </span>
        )}
      </div>
      <p className="truncate text-[11px] text-muted-foreground" title={h.section_path ?? undefined}>
        Section {h.section_path ?? NO_DATA}
      </p>
      <p className="line-clamp-2 text-[11px] leading-snug text-pretty">{h.text}</p>
    </li>
  );
}
