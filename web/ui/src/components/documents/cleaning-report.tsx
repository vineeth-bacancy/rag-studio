import { ChevronRight } from "lucide-react";

import type { CleaningSummary } from "@/lib/api";
import { NO_DATA, num } from "@/lib/format";

/**
 * What cleaning and the guard did to the whole document, PyMuPDF beside
 * Docling. A "-" means the rule does not apply to that parser (Docling
 * decodes "Th" and ligatures itself) or the parser has not run yet.
 */
export function CleaningReport({ summary }: { summary: CleaningSummary }) {
  const py = summary.pymupdf?.counts ?? null;
  const dl = summary.docling?.counts ?? null;
  const rows: { label: string; py: string; dl: string; note?: string }[] = [
    { label: "Hidden text dropped", py: chars(py?.hidden), dl: chars(dl?.hidden),
      note: "Characters in the text layer that the page never shows." },
    { label: "Ligatures replaced", py: n(py?.ligatures), dl: NO_DATA },
    { label: "\"Th\" glyphs repaired", py: n(py?.th), dl: NO_DATA },
    { label: "Footnote markers removed", py: n(py?.footnote), dl: NO_DATA },
    { label: "Letter-spaced runs closed up", py: n(py?.spacing), dl: NO_DATA },
    { label: "Header and footer lines moved", py: n(py?.furniture), dl: n(dl?.furniture) },
    { label: "Dot-leader lines collapsed", py: n(py?.leaders), dl: n(dl?.leaders) },
    { label: "Split words rejoined", py: n(py?.hyphen_joined), dl: NO_DATA },
    { label: "Compounds kept at a line break", py: n(py?.hyphen_kept),
      dl: dl?.hyphen ? `${num(dl.hyphen)} restored` : NO_DATA,
      note: "Docling joins every line-break hyphen; the restored ones are compounds it had merged." },
    { label: "Tables found", py: NO_DATA, dl: n(dl?.tables) },
  ];
  return (
    <details className="group rounded-lg ring-1 ring-foreground/10">
      <summary className="flex cursor-pointer list-none items-center gap-2 rounded-lg px-3 py-2 text-sm select-none hover:bg-muted/50 focus-visible:ring-[3px] focus-visible:ring-ring/50 focus-visible:outline-none [&::-webkit-details-marker]:hidden">
        <ChevronRight className="size-4 text-muted-foreground transition-transform group-open:rotate-90" aria-hidden />
        <span className="font-medium">This document&apos;s cleaning report</span>
        <span className="ml-auto text-xs text-muted-foreground">
          {summary.docling ? "PyMuPDF and Docling" : "PyMuPDF · Docling not run yet"}
        </span>
      </summary>
      <div className="overflow-x-auto border-t px-3 py-3">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left text-xs text-muted-foreground">
              <th className="py-1 pr-3 font-medium">Rule</th>
              <th className="py-1 pr-3 text-right font-medium">PyMuPDF</th>
              <th className="py-1 text-right font-medium">Docling</th>
            </tr>
          </thead>
          <tbody className="divide-y">
            {rows.map((r) => (
              <tr key={r.label}>
                <td className="py-1.5 pr-3">
                  {r.label}
                  {r.note && <span className="block text-[11px] text-muted-foreground">{r.note}</span>}
                </td>
                <td className="py-1.5 pr-3 text-right tabular-nums">{r.py}</td>
                <td className="py-1.5 text-right tabular-nums">{r.dl}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {!summary.docling && (
          <p className="pt-2 text-xs text-muted-foreground">
            Docling converts a document only when its index is built (about 3 seconds per page here). Run: ./rag ingest
            --profile docling-clean
          </p>
        )}
      </div>
    </details>
  );
}

function n(v: number | undefined | null): string {
  return v === undefined || v === null ? "0" : num(v);
}

function chars(v: number | undefined | null): string {
  return v ? `${num(v)} characters` : "0";
}
