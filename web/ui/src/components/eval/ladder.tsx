import Link from "next/link";
import { AlertTriangle } from "lucide-react";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import type { Outcome } from "@/lib/api";
import { NO_DATA } from "@/lib/format";
import { cn } from "@/lib/utils";
import { OUTCOME, recallText } from "./badges";

export interface LadderRow {
  row: number;
  change: string;
  milestone: string;
  sets: Record<string, { run_id: string; recall: { hits: number; total: number } | null; outcomes: Record<Outcome, number> | null }>;
  runs: { id: string; label: string | null; started_at: string }[];
  config: { answer_model: string | null; embedding_model: string; retrieval: string } | null;
  mixed: boolean;
}

/** Pass / Refused / Wrong as three small numbers, each with its icon. */
export function OutcomeCounts({ outcomes }: { outcomes: Record<Outcome, number> | null | undefined }) {
  if (!outcomes) return <span className="text-muted-foreground">{NO_DATA}</span>;
  return (
    <span className="inline-flex items-center gap-2.5 tabular-nums">
      {(["PASS", "REFUSED", "WRONG"] as Outcome[]).map((o) => {
        const s = OUTCOME[o];
        return (
          <span key={o} className={cn("inline-flex items-center gap-0.5", s.className)} title={s.label}>
            <s.icon className="size-3" aria-hidden />
            <span className="sr-only">{s.label}</span>
            {outcomes[o]}
          </span>
        );
      })}
    </span>
  );
}

/**
 * The upgrade ladder (spec §8): each row is ONE change from the row above,
 * measured on the same test questions. A row stays "-" until a run is
 * recorded for it - no row is filled in by assumption.
 */
export function Ladder({ rows }: { rows: LadderRow[] }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle as="h2">The ladder</CardTitle>
        <CardDescription>
          Each row is one change from the row above, scored on the same test questions. An upgrade becomes a
          default only if it moves these numbers.
        </CardDescription>
      </CardHeader>
      <CardContent className="@container px-0">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="w-10 pl-4">Row</TableHead>
              <TableHead>Change</TableHead>
              <TableHead className="text-right">Set A recall@5</TableHead>
              <TableHead className="hidden text-right @lg:table-cell">Set B</TableHead>
              <TableHead className="hidden text-right @lg:table-cell">Set D</TableHead>
              <TableHead className="hidden @2xl:table-cell">Set A scorecard</TableHead>
              <TableHead className="hidden pr-4 @3xl:table-cell">Set D scorecard</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.map((r) => {
              const measured = r.runs.length > 0;
              const a = r.sets["set-a"];
              return (
                <TableRow key={r.row} className={cn(!measured && "text-muted-foreground")}>
                  <TableCell className="pl-4 align-top tabular-nums">{r.row}</TableCell>
                  <TableCell className="align-top whitespace-normal">
                    <p className={cn("text-sm", measured && "font-medium")}>{r.change}</p>
                    <p className="text-[11px] text-muted-foreground">
                      {measured ? (
                        <>
                          {r.config?.answer_model ?? NO_DATA} · {r.config?.embedding_model} · run{" "}
                          {r.runs.map((x, i) => (
                            <span key={x.id}>
                              {i > 0 && ", "}
                              <Link href={`/evaluation?run=${x.id}`} className="underline-offset-2 hover:underline">
                                {x.label ?? x.id}
                              </Link>
                            </span>
                          ))}
                        </>
                      ) : (
                        `Arrives in ${r.milestone}`
                      )}
                    </p>
                    {r.mixed && (
                      <p className="mt-1 flex items-center gap-1 text-[11px] text-amber-700 dark:text-amber-400">
                        <AlertTriangle className="size-3" aria-hidden />
                        These runs used different setups; don&apos;t read them together.
                      </p>
                    )}
                    {/* Narrow cards: the hidden columns' numbers, inline. */}
                    {measured && (
                      <p className="mt-1 text-[11px] text-muted-foreground tabular-nums @lg:hidden">
                        Set B {recallText(r.sets["set-b"]?.recall)} · Set D {recallText(r.sets["set-d"]?.recall)}
                      </p>
                    )}
                    {measured && (
                      <div className="mt-1 text-[11px] @2xl:hidden">
                        <OutcomeCounts outcomes={a?.outcomes} />
                      </div>
                    )}
                  </TableCell>
                  <TableCell className="text-right align-top font-medium tabular-nums">{recallText(a?.recall)}</TableCell>
                  <TableCell className="hidden text-right align-top tabular-nums @lg:table-cell">
                    {recallText(r.sets["set-b"]?.recall)}
                  </TableCell>
                  <TableCell className="hidden text-right align-top tabular-nums @lg:table-cell">
                    {recallText(r.sets["set-d"]?.recall)}
                  </TableCell>
                  <TableCell className="hidden align-top @2xl:table-cell">
                    <OutcomeCounts outcomes={a?.outcomes} />
                  </TableCell>
                  <TableCell className="hidden pr-4 align-top @3xl:table-cell">
                    <OutcomeCounts outcomes={r.sets["set-d"]?.outcomes} />
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
