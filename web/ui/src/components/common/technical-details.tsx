import { ChevronRight } from "lucide-react";

import { cn } from "@/lib/utils";

/**
 * "▸ Technical details": plain language first, the endpoint, model tag and
 * file path one click away (docs/product-language.md, Advanced Details).
 * A native <details> so it works with the keyboard and screen readers for free.
 */
export function TechnicalDetails({
  rows,
  children,
  className,
  label = "Technical details",
}: {
  rows?: Record<string, React.ReactNode>;
  children?: React.ReactNode;
  className?: string;
  label?: string;
}) {
  const entries = Object.entries(rows ?? {});
  if (entries.length === 0 && !children) return null;
  return (
    <details className={cn("group text-xs", className)}>
      <summary className="inline-flex cursor-pointer list-none items-center gap-1 rounded-sm text-muted-foreground select-none hover:text-foreground focus-visible:ring-[3px] focus-visible:ring-ring/50 focus-visible:outline-none [&::-webkit-details-marker]:hidden">
        <ChevronRight className="size-3 transition-transform group-open:rotate-90" aria-hidden />
        {label}
      </summary>
      <div className="mt-2 space-y-2 rounded-md bg-muted/50 px-3 py-2">
        {entries.length > 0 && (
          <dl className="grid grid-cols-[7rem_1fr] gap-x-3 gap-y-1">
            {entries.map(([k, v]) => (
              <div key={k} className="contents">
                <dt className="text-muted-foreground">{k}</dt>
                <dd className="min-w-0 font-mono break-all">{v}</dd>
              </div>
            ))}
          </dl>
        )}
        {children}
      </div>
    </details>
  );
}
