import { Activity, AlertTriangle, CheckCircle2, CircleDot, Info, XCircle } from "lucide-react";
import type { LucideIcon } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { Status } from "@/lib/api";

/**
 * The status vocabulary, rendered. Colour is always paired with an icon and a
 * word, never used alone, so colour-blind readers and greyscale screenshots
 * both work. Green means Healthy and nothing else in this app.
 */
export const STATUS: Record<Status, { label: string; icon: LucideIcon; text: string; ring: string; bg: string }> = {
  healthy: {
    label: "Healthy", icon: CheckCircle2,
    text: "text-emerald-700 dark:text-emerald-400", ring: "border-emerald-500/40", bg: "bg-emerald-500/10",
  },
  running: {
    label: "Running", icon: Activity,
    text: "text-sky-700 dark:text-sky-400", ring: "border-sky-500/40", bg: "bg-sky-500/10",
  },
  warning: {
    label: "Warning", icon: AlertTriangle,
    text: "text-amber-700 dark:text-amber-400", ring: "border-amber-500/40", bg: "bg-amber-500/10",
  },
  needs_attention: {
    label: "Needs attention", icon: XCircle,
    text: "text-red-700 dark:text-red-400", ring: "border-red-500/40", bg: "bg-red-500/10",
  },
  unavailable: {
    label: "Unavailable", icon: XCircle,
    text: "text-red-700 dark:text-red-400", ring: "border-red-500/40", bg: "bg-red-500/10",
  },
  info: {
    label: "Info", icon: Info,
    text: "text-muted-foreground", ring: "border-border", bg: "bg-muted",
  },
};

export function StatusIcon({ status, className }: { status: Status; className?: string }) {
  const s = STATUS[status] ?? { icon: CircleDot, text: "text-muted-foreground" };
  return <s.icon className={cn("size-4 shrink-0", s.text, className)} aria-hidden />;
}

export function StatusBadge({ status, label, className }: { status: Status; label?: string; className?: string }) {
  const s = STATUS[status];
  return (
    <Badge variant="outline" className={cn("gap-1 font-normal", s.ring, s.text, className)}>
      <s.icon className="size-3" aria-hidden />
      {label ?? s.label}
    </Badge>
  );
}
