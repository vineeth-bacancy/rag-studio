import Link from "next/link";
import { AlertTriangle } from "lucide-react";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import type { Outcome } from "@/lib/api";
import { NO_DATA } from "@/lib/format";
import { searchName } from "@/lib/search";
import { cn } from "@/lib/utils";
import { OUTCOME, recallText } from "./badges";

type SetResult = { run_id: string; recall: { hits: number; total: number } | null; outcomes: Record<Outcome, number> | null };

/** One setup measured for a ladder row (row 2 has two parsers). Mirrors eval.py `_variant()`. */
export interface LadderVariant {
  label: string;
  sets: Record<string, SetResult>;
  runs: { id: string; label: string | null; started_at: string }[];
  config: {
    answer_model: string | null;
    embedding_model: string;
    retrieval: string;
    section_cap?: number | null;
    index_label?: string | null;
    prompt_version?: string | null;
  } | null;
  /** The variant's runs used different index builds: don't read them together. */
  mixed: boolean;
}

export interface LadderRow {
  row: number;
  change: string;
  milestone: string;
  /** Empty until a run is recorded for the row. */
  variants: LadderVariant[];
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
 * measured on the same test questions. A row can hold more than one variant
 * when the change has alternatives (row 2: PyMuPDF or Docling, each with the
 * full cleaning order); each gets its own line. A row stays "-" until a run is
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
            {rows.flatMap((r) =>
              r.variants.length === 0
                ? [<EmptyLadderRow key={r.row} row={r} />]
                : r.variants.map((v, i) => (
                    <VariantRow key={`${r.row}:${v.label}`} row={r} variant={v} first={i === 0} last={i === r.variants.length - 1} />
                  )),
            )}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  );
}

function EmptyLadderRow({ row }: { row: LadderRow }) {
  return (
    <TableRow className="text-muted-foreground">
      <TableCell className="pl-4 align-top tabular-nums">{row.row}</TableCell>
      <TableCell className="align-top whitespace-normal">
        <p className="text-sm">{row.change}</p>
        <p className="text-[11px] text-muted-foreground">Arrives in {row.milestone}</p>
      </TableCell>
      <TableCell className="text-right align-top tabular-nums">{NO_DATA}</TableCell>
      <TableCell className="hidden text-right align-top tabular-nums @lg:table-cell">{NO_DATA}</TableCell>
      <TableCell className="hidden text-right align-top tabular-nums @lg:table-cell">{NO_DATA}</TableCell>
      <TableCell className="hidden align-top @2xl:table-cell">{NO_DATA}</TableCell>
      <TableCell className="hidden pr-4 align-top @3xl:table-cell">{NO_DATA}</TableCell>
    </TableRow>
  );
}

function VariantRow({ row, variant: v, first, last }: { row: LadderRow; variant: LadderVariant; first: boolean; last: boolean }) {
  const a = v.sets["set-a"];
  const many = row.variants.length > 1;
  return (
    // Variants of one row share a band: no rule between them, only after the last.
    <TableRow className={cn(!last && "border-b-0")}>
      <TableCell className="pl-4 align-top tabular-nums">
        {first ? row.row : <span className="sr-only">{row.row}</span>}
      </TableCell>
      <TableCell className="align-top whitespace-normal">
        {first && <p className="text-sm font-medium">{row.change}</p>}
        {many && <p className={cn("text-sm", first && "mt-1")}>{v.label}</p>}
        <p className="text-[11px] text-muted-foreground">
          {v.config?.index_label ? `${v.config.index_label} index · ` : ""}
          {v.config ? `${searchName(v.config.retrieval, v.config.section_cap)} · ` : ""}
          {v.config?.answer_model ?? NO_DATA}
          {v.config?.prompt_version ? ` · prompt ${v.config.prompt_version}` : ""} · {v.config?.embedding_model ?? NO_DATA} · run{" "}
          {v.runs.map((x, i) => (
            <span key={x.id}>
              {i > 0 && ", "}
              <Link href={`/evaluation?run=${x.id}`} className="underline-offset-2 hover:underline">
                {x.label ?? x.id}
              </Link>
            </span>
          ))}
        </p>
        {v.mixed && (
          <p className="mt-1 flex items-center gap-1 text-[11px] text-amber-700 dark:text-amber-400">
            <AlertTriangle className="size-3" aria-hidden />
            These runs used different index builds; don&apos;t read them together.
          </p>
        )}
        {/* Narrow cards: the hidden columns' numbers, inline. */}
        <p className="mt-1 text-[11px] text-muted-foreground tabular-nums @lg:hidden">
          Set B {recallText(v.sets["set-b"]?.recall)} · Set D {recallText(v.sets["set-d"]?.recall)}
        </p>
        <div className="mt-1 text-[11px] @2xl:hidden">
          <OutcomeCounts outcomes={a?.outcomes} />
        </div>
      </TableCell>
      <TableCell className="text-right align-top font-medium tabular-nums">{recallText(a?.recall)}</TableCell>
      <TableCell className="hidden text-right align-top tabular-nums @lg:table-cell">
        {recallText(v.sets["set-b"]?.recall)}
      </TableCell>
      <TableCell className="hidden text-right align-top tabular-nums @lg:table-cell">
        {recallText(v.sets["set-d"]?.recall)}
      </TableCell>
      <TableCell className="hidden align-top @2xl:table-cell">
        <OutcomeCounts outcomes={a?.outcomes} />
      </TableCell>
      <TableCell className="hidden pr-4 align-top @3xl:table-cell">
        <OutcomeCounts outcomes={v.sets["set-d"]?.outcomes} />
      </TableCell>
    </TableRow>
  );
}
