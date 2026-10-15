import { useMemo, useRef, useState } from "react";

import type { EmbedLab } from "@/lib/api";

/**
 * Every passage of an index as a dot: its vector projected to two dimensions
 * (PCA). The question lands among them, with a line to each of its five true
 * nearest passages, nearest in ALL dimensions, so a short line in this
 * picture is not what decides the search.
 *
 * Colour is the document (validated palette, all pairs, both themes: blue,
 * amber, magenta; globals.css). Equal scale on both axes, because a stretched
 * PCA plane would lie about distance. Hover uses the nearest dot within reach
 * (dataviz: no pinpoint targets on a dense scatter).
 */

export const DOC_COLORS = ["var(--chart-1)", "var(--chart-3)", "var(--chart-6)"];
const W = 640;
const H = 380;
const PAD = 18;
const REACH = 16; // px on screen

export function docColor(data: EmbedLab, doc: string): string {
  const i = data.documents.findIndex((d) => d.doc_id === doc);
  return DOC_COLORS[i] ?? "var(--muted-foreground)";
}

export function EmbedMap({
  data,
  focus,
  onFocus,
}: {
  data: EmbedLab;
  focus: string | null;
  onFocus: (id: string | null) => void;
}) {
  const svg = useRef<SVGSVGElement>(null);
  const [hover, setHover] = useState<{ i: number; left: number; top: number } | null>(null);
  const q = data.question;

  const { sx, sy } = useMemo(() => {
    const xs = data.points.map((p) => p.xy[0]).concat(q ? [q.xy[0]] : []);
    const ys = data.points.map((p) => p.xy[1]).concat(q ? [q.xy[1]] : []);
    const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
    const k = Math.min((W - 2 * PAD) / (x1 - x0 || 1), (H - 2 * PAD) / (y1 - y0 || 1));
    const ox = (W - k * (x1 - x0)) / 2;
    const oy = (H - k * (y1 - y0)) / 2;
    return { sx: (x: number) => ox + k * (x - x0), sy: (y: number) => H - (oy + k * (y - y0)) };
  }, [data.points, q]);

  const nearest = new Map((q?.nearest ?? []).map((h) => [h.id, h]));

  const move = (e: React.PointerEvent<SVGSVGElement>) => {
    const el = svg.current;
    if (!el) return;
    const box = el.getBoundingClientRect();
    const scale = box.width / W;
    const px = (e.clientX - box.left) / scale;
    const py = (e.clientY - box.top) / scale;
    let best = -1;
    let bestD = (REACH / scale) ** 2;
    data.points.forEach((p, i) => {
      const d = (sx(p.xy[0]) - px) ** 2 + (sy(p.xy[1]) - py) ** 2;
      if (d < bestD) {
        bestD = d;
        best = i;
      }
    });
    setHover(best < 0 ? null : { i: best, left: sx(data.points[best].xy[0]) * scale, top: sy(data.points[best].xy[1]) * scale });
    onFocus(best < 0 ? null : data.points[best].id);
  };

  const hovered = hover ? data.points[hover.i] : null;
  const focused = focus ? data.points.find((p) => p.id === focus) : null;

  return (
    <div className="relative">
      <svg
        ref={svg}
        viewBox={`0 0 ${W} ${H}`}
        className="h-auto w-full touch-none rounded-lg bg-muted/30 ring-1 ring-foreground/5"
        role="img"
        aria-label={`${data.rows} passages of the ${data.label} index as dots${q ? ", with the question and lines to its five nearest passages" : ""}.`}
        onPointerMove={move}
        onPointerLeave={() => {
          setHover(null);
          onFocus(null);
        }}
      >
        {data.points.map((p) => (
          <circle key={p.id} cx={sx(p.xy[0])} cy={sy(p.xy[1])} r={2.2} fill={docColor(data, p.doc)} fillOpacity={0.5} />
        ))}
        {q && (
          <g>
            {q.nearest.map((h) => (
              <line
                key={h.id}
                x1={sx(q.xy[0])}
                y1={sy(q.xy[1])}
                x2={sx(h.xy[0])}
                y2={sy(h.xy[1])}
                stroke="currentColor"
                strokeOpacity={focus === h.id ? 0.9 : 0.45}
                strokeWidth={focus === h.id ? 2 : 1.25}
                strokeDasharray="4 3"
                className="text-foreground"
              />
            ))}
            {/* Every dot first, then every label, so no dot hides another's
                rank (QA M4-04: ranks 1 and 2 sat under 3 and 4). */}
            {q.nearest.map((h) => (
              <circle key={h.id} cx={sx(h.xy[0])} cy={sy(h.xy[1])} r={6} fill={docColor(data, h.doc)} stroke="var(--card)" strokeWidth={2} />
            ))}
            {q.nearest.map((h) => (
              <g key={`label-${h.id}`}>
                <text
                  x={sx(h.xy[0])}
                  y={sy(h.xy[1]) - 9}
                  textAnchor="middle"
                  fontSize={11}
                  fontWeight={600}
                  className="fill-foreground"
                  style={{ paintOrder: "stroke", stroke: "var(--card)", strokeWidth: 3 }}
                >
                  {h.rank}
                </text>
              </g>
            ))}
            <circle cx={sx(q.xy[0])} cy={sy(q.xy[1])} r={8} fill="var(--background)" stroke="currentColor" strokeWidth={2.5} className="text-foreground" />
            <circle cx={sx(q.xy[0])} cy={sy(q.xy[1])} r={2.5} fill="currentColor" className="text-foreground" />
          </g>
        )}
        {(focused ?? hovered) && !nearest.has((focused ?? hovered)!.id) && (
          <circle
            cx={sx((focused ?? hovered)!.xy[0])}
            cy={sy((focused ?? hovered)!.xy[1])}
            r={5}
            fill="none"
            stroke="currentColor"
            strokeWidth={1.5}
            className="text-foreground"
          />
        )}
      </svg>
      {hovered && hover && (
        <div
          className="pointer-events-none absolute z-10 -translate-x-1/2 -translate-y-full rounded-md bg-popover px-2 py-1 text-xs whitespace-nowrap text-popover-foreground shadow-md ring-1 ring-foreground/10"
          style={{ left: hover.left, top: hover.top - 10 }}
        >
          {data.documents.find((d) => d.doc_id === hovered.doc)?.title ?? hovered.doc} · page {hovered.page}
          {nearest.has(hovered.id) && <span className="text-muted-foreground"> · nearest #{nearest.get(hovered.id)!.rank}</span>}
        </div>
      )}
    </div>
  );
}
