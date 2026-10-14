import * as React from "react";

import { cn } from "@/lib/utils";

/**
 * Hairline-separated tile row. The separators are the parent's background
 * showing through a 1px grid gap, so this must never sit directly on
 * CardContent - its horizontal padding would show as a strip of border colour.
 */
export function TileGrid({
  className,
  style,
  children,
}: {
  className?: string;
  style?: React.CSSProperties;
  children: React.ReactNode;
}) {
  return (
    <div className={cn("grid gap-px overflow-hidden rounded-lg bg-border", className)} style={style}>
      {children}
    </div>
  );
}

export function Tile({
  label,
  value,
  hint,
  valueClass,
}: {
  label: string;
  value: React.ReactNode;
  hint?: React.ReactNode;
  valueClass?: string;
}) {
  return (
    <div className="space-y-1 bg-card px-4 py-3">
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className={cn("text-lg font-semibold tabular-nums", valueClass)}>{value}</div>
      {hint && <div className="text-[11px] text-muted-foreground">{hint}</div>}
    </div>
  );
}
