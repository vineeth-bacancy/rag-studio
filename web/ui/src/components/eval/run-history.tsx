import Link from "next/link";
import { GitCompareArrows } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import type { RunBrief } from "@/lib/api";
import { NO_DATA, dateTimeLabel } from "@/lib/format";
import { searchName, setupExtras } from "@/lib/search";
import { cn } from "@/lib/utils";
import { duration } from "@/components/ask/trace";
import { RunStatusBadge, recallText } from "./badges";
import { OutcomeCounts } from "./ladder";

/**
 * Every run, newest first. Selecting one opens its per-question detail;
 * "Compare" puts a finished run beside the finished run before it.
 */
export function RunHistory({
  runs,
  selected,
  onCompare,
}: {
  runs: RunBrief[];
  selected: string | null;
  onCompare?: (before: string, after: string) => void;
}) {
  const finished = runs.filter((r) => r.status === "done");
  return (
    <Card>
      <CardHeader>
        <CardTitle as="h2">Runs</CardTitle>
        <CardDescription>
          Every run keeps its full setup: the index build, the models and their exact versions, the prompt.
        </CardDescription>
      </CardHeader>
      <CardContent className="@container px-0">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="pl-4">Run</TableHead>
              <TableHead className="hidden @xl:table-cell">Recall@5</TableHead>
              <TableHead>Scorecard</TableHead>
              <TableHead className="hidden pr-4 text-right @md:table-cell">Status</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {runs.map((r) => {
              const recall = Object.entries(r.summary?.recall ?? {})
                .filter(([, v]) => v.total > 0)
                .map(([sid, v]) => `${r.sets[sid]?.name ?? sid} ${recallText(v)}`)
                .join(" · ");
              // The finished run before this one: what "Compare" puts beside it.
              const prev = r.status === "done" ? finished[finished.indexOf(r) + 1] : undefined;
              return (
                <TableRow key={r.id} className={cn(selected === r.id && "bg-muted/50")}>
                  <TableCell className="pl-4 align-top whitespace-normal">
                    <Link
                      href={`/evaluation?run=${r.id}`}
                      scroll={false}
                      className="rounded-sm text-sm font-medium underline-offset-2 hover:underline focus-visible:ring-[3px] focus-visible:ring-ring/50 focus-visible:outline-none"
                      aria-current={selected === r.id ? "true" : undefined}
                    >
                      {r.label ?? Object.values(r.sets).map((s) => s.name).join(", ")}
                    </Link>
                    <p className="text-[11px] text-muted-foreground">
                      {dateTimeLabel(r.started_at)}
                      {r.config.index_label ? ` · ${r.config.index_label} index` : ""} ·{" "}
                      {[searchName(r.config.retrieval, r.config.section_cap), ...setupExtras(r.config)].join(" · ")} ·{" "}
                      {r.spec.mode === "answer" ? r.config.answer_model : "retrieval only"}
                      {r.spec.mode === "answer" && r.config.prompt_version ? ` · prompt ${r.config.prompt_version}` : ""}
                      {r.seconds !== null ? ` · ${duration(r.seconds * 1000)}` : ""}
                      {r.ladder ? ` · ladder row ${r.ladder.row}` : ""}
                    </p>
                    <p className="text-[11px] text-muted-foreground tabular-nums @xl:hidden">{recall || NO_DATA}</p>
                    <div className="mt-1 flex flex-wrap items-center gap-2">
                      <span className="@md:hidden">
                        <RunStatusBadge status={r.status} />
                      </span>
                      {onCompare && prev && (
                        <Button
                          variant="ghost"
                          size="xs"
                          className="-ml-1.5 text-muted-foreground"
                          onClick={() => onCompare(prev.id, r.id)}
                          aria-label="Compare this run with the finished run before it"
                        >
                          <GitCompareArrows aria-hidden />
                          Compare
                        </Button>
                      )}
                    </div>
                  </TableCell>
                  <TableCell className="hidden align-top text-xs whitespace-normal tabular-nums @xl:table-cell">
                    {recall || NO_DATA}
                  </TableCell>
                  <TableCell className="align-top">
                    {r.status === "running" ? (
                      <span className="text-xs text-muted-foreground tabular-nums">
                        {r.progress.done} of {r.progress.total}
                      </span>
                    ) : (
                      <OutcomeCounts outcomes={r.summary?.outcomes} />
                    )}
                  </TableCell>
                  <TableCell className="hidden pr-4 text-right align-top @md:table-cell">
                    <RunStatusBadge status={r.status} />
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  );
}
