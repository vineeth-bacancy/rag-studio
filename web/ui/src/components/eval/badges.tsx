import { CheckCircle2, CircleSlash, XCircle } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { StatusBadge } from "@/components/common/status";
import type { Outcome, RunResult, RunStatus } from "@/lib/api";
import { NO_DATA } from "@/lib/format";
import { cn } from "@/lib/utils";

/**
 * Scorecard outcomes (docs/product-language.md): Pass in the Healthy colour,
 * Refused neutral (a safe failure, never red), Wrong in red. Always with an
 * icon and a word, never colour alone.
 */
export const OUTCOME: Record<Outcome, { label: string; icon: typeof CheckCircle2; className: string }> = {
  PASS: {
    label: "Pass", icon: CheckCircle2,
    className: "border-emerald-500/40 text-emerald-700 dark:text-emerald-400",
  },
  REFUSED: {
    label: "Refused", icon: CircleSlash,
    className: "border-border text-muted-foreground",
  },
  WRONG: {
    label: "Wrong", icon: XCircle,
    className: "border-red-500/40 text-red-700 dark:text-red-400",
  },
};

export function OutcomeBadge({ outcome, error }: { outcome: Outcome | null; error?: string | null }) {
  if (error) {
    return (
      <Badge variant="outline" className="gap-1 border-amber-500/40 font-normal text-amber-700 dark:text-amber-400" title={error}>
        Error
      </Badge>
    );
  }
  if (!outcome) return <span className="text-muted-foreground">{NO_DATA}</span>;
  const o = OUTCOME[outcome];
  return (
    <Badge variant="outline" className={cn("gap-1 font-normal", o.className)}>
      <o.icon className="size-3" aria-hidden />
      {o.label}
    </Badge>
  );
}

/** "Hit@2", or "Miss" with where the answer page really was. */
export function HitLabel({ result }: { result: Pick<RunResult, "targets" | "hit_rank" | "best_rank"> }) {
  if (result.targets.length === 0) return <span className="text-muted-foreground">{NO_DATA}</span>;
  if (result.hit_rank !== null) {
    return <span className="font-medium tabular-nums">Hit@{result.hit_rank}</span>;
  }
  return (
    <span className="tabular-nums">
      <span className="font-medium text-red-700 dark:text-red-400">Miss</span>
      <span className="text-muted-foreground">
        {result.best_rank !== null ? ` (rank ${result.best_rank})` : " (not in top 50)"}
      </span>
    </span>
  );
}

const RUN_STATUS: Record<RunStatus, { status: "healthy" | "running" | "warning" | "needs_attention" | "info"; label: string }> = {
  done: { status: "healthy", label: "Done" },
  running: { status: "running", label: "Running" },
  cancelled: { status: "info", label: "Cancelled" },
  interrupted: { status: "warning", label: "Interrupted" },
  failed: { status: "needs_attention", label: "Failed" },
};

export function RunStatusBadge({ status }: { status: RunStatus }) {
  const s = RUN_STATUS[status];
  return <StatusBadge status={s.status} label={s.label} className="text-[11px]" />;
}

/** "8/9", or "-" when the set was not part of the run. */
export function recallText(r: { hits: number; total: number } | null | undefined): string {
  return r && r.total ? `${r.hits}/${r.total}` : NO_DATA;
}
