import { useState } from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import type { DocumentFile } from "@/lib/api";
import { num } from "@/lib/format";

/** A ready-made page worth looking at, with why. */
export interface Preset {
  docId: string;
  page: number;
  label: string;
}

/**
 * Document, page and (optionally) ready-made pages. The page count comes from
 * the engine's answer for the page shown, or the library until then.
 */
export function DocPagePicker({
  id,
  docs,
  docId,
  page,
  pageCount,
  presets,
  onChange,
}: {
  /** Prefix for the controls' ids: two pickers can share a page. */
  id: string;
  docs: DocumentFile[];
  docId: string;
  page: number;
  pageCount: number | null;
  presets?: Preset[];
  onChange: (docId: string, page: number) => void;
}) {
  const [draft, setDraft] = useState<string | null>(null);
  const total = pageCount ?? docs.find((d) => d.doc_id === docId)?.pages ?? null;
  const titles = Object.fromEntries(docs.map((d) => [d.doc_id, d.title ?? d.doc_id]));
  const go = (p: number) => {
    setDraft(null);
    onChange(docId, total ? Math.min(Math.max(1, p), total) : Math.max(1, p));
  };
  const presetKey = (p: Preset) => `${p.docId}:${p.page}`;
  const current = presets?.find((p) => p.docId === docId && p.page === page);

  return (
    <div className="flex flex-wrap items-end gap-3">
      {presets && presets.length > 0 && (
        <div className="min-w-0 space-y-1">
          <label htmlFor={`${id}-preset`} className="text-xs font-medium text-muted-foreground">
            Suggested pages
          </label>
          <Select
            value={current ? presetKey(current) : null}
            items={Object.fromEntries(presets.map((p) => [presetKey(p), p.label]))}
            onValueChange={(v) => {
              const p = presets.find((x) => presetKey(x) === v);
              if (p) onChange(p.docId, p.page);
            }}
          >
            <SelectTrigger id={`${id}-preset`} className="w-full max-w-80 sm:w-80">
              <SelectValue placeholder="Choose a page" />
            </SelectTrigger>
            <SelectContent>
              {presets.map((p) => (
                <SelectItem key={presetKey(p)} value={presetKey(p)}>
                  {p.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      )}
      <div className="min-w-0 space-y-1">
        <label htmlFor={`${id}-doc`} className="text-xs font-medium text-muted-foreground">
          Document
        </label>
        <Select value={docId} items={titles} onValueChange={(v) => v && onChange(v as string, 1)}>
          <SelectTrigger id={`${id}-doc`} className="w-full max-w-72 sm:w-72">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {docs.map((d) => (
              <SelectItem key={d.doc_id} value={d.doc_id}>
                {d.title ?? d.doc_id}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
      <div className="space-y-1">
        <span className="text-xs font-medium text-muted-foreground" id={`${id}-page-label`}>
          Page
        </span>
        <div className="flex items-center gap-1" role="group" aria-labelledby={`${id}-page-label`}>
          <Button variant="outline" size="icon-sm" onClick={() => go(page - 1)} disabled={page <= 1} aria-label="Previous page">
            <ChevronLeft aria-hidden />
          </Button>
          <form
            onSubmit={(e) => {
              e.preventDefault();
              const n = Number(draft);
              if (Number.isInteger(n) && n > 0) go(n);
            }}
          >
            <Input
              aria-label="Page number"
              inputMode="numeric"
              className="h-7 w-14 text-center tabular-nums"
              value={draft ?? String(page)}
              onChange={(e) => setDraft(e.target.value.replace(/\D/g, ""))}
              onBlur={() => setDraft(null)}
            />
          </form>
          <span className="text-sm text-muted-foreground tabular-nums">of {num(total)}</span>
          <Button
            variant="outline"
            size="icon-sm"
            onClick={() => go(page + 1)}
            disabled={!total || page >= total}
            aria-label="Next page"
          >
            <ChevronRight aria-hidden />
          </Button>
        </div>
      </div>
    </div>
  );
}
