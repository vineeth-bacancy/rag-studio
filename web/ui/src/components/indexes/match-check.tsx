import { CheckCircle2, Loader2, ScanSearch } from "lucide-react";

import { Button } from "@/components/ui/button";
import { StatusBadge } from "@/components/common/status";
import type { MatchCheck, MatchPair } from "@/lib/api";
import { relativeTime } from "@/lib/format";
import { cn } from "@/lib/utils";

const SIDE: Record<string, string> = { local: "This computer", colab: "Colab GPU", stored: "the index" };

function pairLabel(p: MatchPair): string {
  if (p.kind === "noise") return `${SIDE[p.a]} vs itself, another order`;
  if (p.kind === "index") return `${SIDE[p.a]} vs the index`;
  return `This computer vs Colab GPU · ${p.what}`;
}

/**
 * The vector match check (spec §2.2): the same model on two computers must
 * give the same vectors, cosine >= 0.999, or an index built on one could not
 * be searched from the other. Through Ollama a text's vector also depends on
 * what was embedded just before it, so each computer is first measured
 * against itself (the noise floor); the comparison with the stored vectors
 * is shown, not judged.
 */
export function MatchCheckPanel({
  check,
  running,
  disabledReason,
  onRun,
}: {
  check: MatchCheck | null | undefined;
  running: boolean;
  disabledReason: string | null;
  onRun: () => void;
}) {
  const crossPairs = check?.pairs.filter((p) => p.kind === "cross") ?? [];
  const worst = crossPairs.length ? Math.min(...crossPairs.map((p) => p.min)) : null;
  return (
    <div className="space-y-2 rounded-lg bg-muted/40 p-3 ring-1 ring-foreground/5">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="flex items-center gap-1.5 text-sm font-medium">
          <ScanSearch className="size-4 text-muted-foreground" aria-hidden />
          Vector match check
        </p>
        {!check ? (
          <StatusBadge status="info" label="Not checked" />
        ) : check.passed === null ? (
          <StatusBadge status="info" label="One computer only" />
        ) : check.passed ? (
          <StatusBadge status="healthy" label="Match across computers" />
        ) : (
          <StatusBadge status="needs_attention" label="Does not match" />
        )}
      </div>

      {check ? (
        <>
          <ul className="space-y-1 text-xs">
            {check.pairs.map((p) => (
              <li key={`${p.a}-${p.b}-${p.what}`} className="flex flex-wrap items-baseline justify-between gap-x-3">
                <span className="text-muted-foreground">
                  {pairLabel(p)} · {p.n} {p.what}
                </span>
                <span
                  className={cn(
                    "font-mono tabular-nums",
                    p.kind === "cross" && p.min < check.threshold && "font-semibold text-red-700 dark:text-red-400",
                  )}
                >
                  lowest {p.min.toFixed(5)}
                </span>
              </li>
            ))}
          </ul>
          {Object.entries(check.sides)
            .filter(([, s]) => !s.available)
            .map(([side, s]) => (
              <p key={side} className="text-xs text-pretty text-muted-foreground">
                {SIDE[side]} skipped: {s.reason}
              </p>
            ))}
          <p className="text-xs text-pretty text-muted-foreground">
            {check.cross_machine && worst != null ? (
              <>
                Across computers the lowest cosine is <span className="font-mono tabular-nums">{worst.toFixed(5)}</span>;
                it needs at least {check.threshold}.{" "}
              </>
            ) : (
              <>Only one computer can embed for this index, so there is nothing to compare across. </>
            )}
            {check.noise_floor != null && (
              <>
                One computer against itself reaches{" "}
                <span className="font-mono tabular-nums">{check.noise_floor.toFixed(5)}</span>: through Ollama a
                text&apos;s vector depends slightly on what was embedded just before it.{" "}
              </>
            )}
            Checked {relativeTime(check.checked_at)}.
          </p>
        </>
      ) : (
        <p className="text-xs text-pretty text-muted-foreground">
          Embeds 50 passages and 5 test questions on this computer and on the Colab GPU, one per request, and compares
          them. A cosine of at least 0.999 means an index built on one can be searched from the other.
        </p>
      )}

      <div className="flex flex-wrap items-center gap-2">
        <Button size="sm" variant="outline" onClick={onRun} disabled={running || disabledReason != null} aria-busy={running}>
          {running ? <Loader2 className="size-4 animate-spin" aria-hidden /> : <CheckCircle2 className="size-4" aria-hidden />}
          {running ? "Checking… up to 4 minutes" : check ? "Check again" : "Check vectors"}
        </Button>
        {disabledReason && !running && <span className="text-xs text-muted-foreground">{disabledReason}</span>}
      </div>
    </div>
  );
}
