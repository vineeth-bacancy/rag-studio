import { useState } from "react";
import { FileText } from "lucide-react";

import { Skeleton } from "@/components/ui/skeleton";
import type { PageBox } from "@/lib/api";
import { pageImageUrl } from "@/lib/documents";
import { cn } from "@/lib/utils";

export interface Overlay {
  box: PageBox;
  /** Classes for the outline (colour per layer). */
  className: string;
  title: string;
}

/**
 * The page as the PDF draws it, with boxes over the regions a parser or a
 * rule found: furniture, hidden text, Docling's tables and pictures. Boxes are
 * in PDF points (top-left origin), the same space the engine reports.
 */
export function PageCanvas({
  docId,
  page,
  width,
  height,
  overlays,
  alt,
}: {
  docId: string;
  page: number;
  width: number;
  height: number;
  overlays: Overlay[];
  alt: string;
}) {
  const [loaded, setLoaded] = useState(false);
  const [failed, setFailed] = useState(false);
  return (
    <div
      className="relative w-full overflow-hidden rounded-lg bg-white ring-1 ring-foreground/10"
      style={{ aspectRatio: `${width} / ${height}` }}
    >
      {!loaded && !failed && <Skeleton className="absolute inset-0 rounded-none" />}
      {failed ? (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 p-6 text-center text-sm text-muted-foreground">
          <FileText className="size-6" aria-hidden />
          Couldn&apos;t render this page. Its text is beside it.
        </div>
      ) : (
        // eslint-disable-next-line @next/next/no-img-element -- a PNG the API renders on demand
        <img
          key={`${docId}:${page}`}
          src={pageImageUrl(docId, page, 130)}
          alt={alt}
          className="absolute inset-0 size-full object-contain"
          onLoad={() => setLoaded(true)}
          onError={() => setFailed(true)}
        />
      )}
      {loaded &&
        overlays.map((o, i) => {
          const [x0, y0, x1, y1] = o.box;
          // Text placed outside the page (GPO p15 carries p14's figures there)
          // has no place to be drawn; the list beside the page names it.
          if (x1 <= 0 || y1 <= 0 || x0 >= width || y0 >= height) return null;
          const l = Math.max(0, x0), t = Math.max(0, y0);
          return (
            <span
              key={i}
              title={o.title}
              className={cn("absolute rounded-[2px]", o.className)}
              style={{
                left: `${(l / width) * 100}%`,
                top: `${(t / height) * 100}%`,
                width: `${((Math.min(x1, width) - l) / width) * 100}%`,
                height: `${((Math.min(y1, height) - t) / height) * 100}%`,
              }}
            />
          );
        })}
    </div>
  );
}
