/**
 * Formatting helpers.
 *
 * One rule runs through all of these: `null` and `undefined` mean the value is
 * genuinely unavailable and render as "-", never as 0. Showing 0 for a missing
 * score would be inventing a number, which this app must never do.
 */

export const NO_DATA = "-";

export function isMissing(v: unknown): boolean {
  return v === null || v === undefined || (typeof v === "number" && !Number.isFinite(v));
}

const int = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });

export function num(v: number | null | undefined): string {
  return isMissing(v) ? NO_DATA : int.format(v as number);
}

/** 1359293444 -> "1.3 GB". Decimal units, as Ollama and Hugging Face print them. */
export function bytes(v: number | null | undefined): string {
  if (isMissing(v)) return NO_DATA;
  const n = v as number;
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)} GB`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(0)} MB`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(0)} KB`;
  return `${n} B`;
}

/** 40960 -> "40K tokens". */
export function tokens(v: number | null | undefined): string {
  if (isMissing(v)) return NO_DATA;
  const n = v as number;
  return n >= 1000 ? `${Math.round(n / 1024)}K tokens` : `${n} tokens`;
}

export function dateTimeLabel(iso: string | null | undefined): string {
  if (!iso) return NO_DATA;
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return NO_DATA;
  return d.toLocaleString(undefined, {
    day: "numeric", month: "short", hour: "2-digit", minute: "2-digit",
  });
}

/** "in 4 minutes", "2 hours ago". */
export function relativeTime(iso: string | null | undefined): string {
  if (!iso) return NO_DATA;
  const t = new Date(iso).getTime();
  if (Number.isNaN(t)) return NO_DATA;
  const diff = (t - Date.now()) / 1000;
  const rtf = new Intl.RelativeTimeFormat(undefined, { numeric: "auto" });
  const abs = Math.abs(diff);
  if (abs < 60) return rtf.format(Math.round(diff), "second");
  if (abs < 3600) return rtf.format(Math.round(diff / 60), "minute");
  if (abs < 86400) return rtf.format(Math.round(diff / 3600), "hour");
  return rtf.format(Math.round(diff / 86400), "day");
}
