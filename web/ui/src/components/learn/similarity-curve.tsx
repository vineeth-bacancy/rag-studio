import type { ChunkLab } from "@/lib/api";

const W = 640;
const H = 180;
const PAD = { left: 40, right: 12, top: 14, bottom: 28 };

/**
 * Semantic chunking's view of a page: how similar each sentence is to the
 * next. Where the line drops below the threshold, a passage ends. Colours come
 * from text colour classes (`currentColor`), so the chart follows the theme;
 * each point's title names the two sentences it compares.
 */
export function SimilarityCurve({ data }: { data: ChunkLab }) {
  const curve = data.curve ?? [];
  const threshold = data.threshold;
  if (curve.length < 2 || threshold === null) {
    return <p className="text-xs text-muted-foreground">Too few sentences on this page to compare.</p>;
  }
  const values = [...curve.map((c) => c.similarity), threshold];
  const lo = Math.max(0, Math.floor((Math.min(...values) - 0.05) * 20) / 20);
  const hi = Math.min(1, Math.ceil((Math.max(...values) + 0.05) * 20) / 20);
  const x = (i: number) => PAD.left + (i * (W - PAD.left - PAD.right)) / (curve.length - 1);
  const y = (v: number) => PAD.top + ((hi - v) * (H - PAD.top - PAD.bottom)) / (hi - lo || 1);
  const cuts = curve.filter((c) => c.cut).length;
  const snippet = (a: number, b: number) => data.text.slice(a, b).replace(/\s+/g, " ").trim();

  return (
    <figure className="space-y-1.5">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="h-auto w-full overflow-visible"
        role="img"
        aria-label={`Similarity of ${curve.length} pairs of neighbouring sentences, from ${lo.toFixed(2)} to ${hi.toFixed(2)}. ${cuts} fall below the threshold of ${threshold.toFixed(3)}, so the page is cut there.`}
      >
        {/* Axes and their two ticks. */}
        <g className="text-muted-foreground" fill="currentColor" fontSize="10">
          <line x1={PAD.left} x2={W - PAD.right} y1={H - PAD.bottom} y2={H - PAD.bottom} stroke="currentColor" strokeOpacity="0.3" />
          <text x={PAD.left - 6} y={y(hi) + 3} textAnchor="end">{hi.toFixed(2)}</text>
          <text x={PAD.left - 6} y={y(lo) + 3} textAnchor="end">{lo.toFixed(2)}</text>
          <text x={PAD.left} y={H - 8}>first sentence</text>
          <text x={W - PAD.right} y={H - 8} textAnchor="end">last sentence</text>
        </g>
        {/* The threshold: below it, a passage ends. */}
        <g className="text-amber-600 dark:text-amber-400">
          <line
            x1={PAD.left}
            x2={W - PAD.right}
            y1={y(threshold)}
            y2={y(threshold)}
            stroke="currentColor"
            strokeDasharray="4 3"
          />
          <text x={W - PAD.right} y={y(threshold) - 4} textAnchor="end" fontSize="10" fill="currentColor">
            threshold {threshold.toFixed(3)}
          </text>
        </g>
        <g className="text-sky-600 dark:text-sky-400">
          <polyline
            points={curve.map((c, i) => `${x(i)},${y(c.similarity)}`).join(" ")}
            fill="none"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeLinejoin="round"
          />
          {curve.map((c, i) => (
            <g key={i}>
              {c.cut && (
                <line
                  x1={x(i)}
                  x2={x(i)}
                  y1={PAD.top}
                  y2={H - PAD.bottom}
                  className="text-amber-600 dark:text-amber-400"
                  stroke="currentColor"
                  strokeOpacity="0.45"
                />
              )}
              <circle
                cx={x(i)}
                cy={y(c.similarity)}
                r={c.cut ? 4 : 2.5}
                className={c.cut ? "text-amber-600 dark:text-amber-400" : undefined}
                fill="currentColor"
              >
                <title>
                  {`Sentences ${i + 1} and ${i + 2}: similarity ${c.similarity.toFixed(3)}${c.cut ? " (a passage ends here)" : ""}\n…${snippet(Math.max(0, c.after - 60), c.after)} | ${snippet(c.next, c.next + 60)}…`}
                </title>
              </circle>
            </g>
          ))}
        </g>
      </svg>
      <figcaption className="text-xs text-pretty text-muted-foreground">
        Each point compares one sentence with the next. Large points under the dashed line are where a passage
        ends ({cuts} on this page). Hover a point to read the two sentences.
      </figcaption>
    </figure>
  );
}
