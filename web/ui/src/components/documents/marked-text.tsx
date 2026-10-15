import { Fragment } from "react";

import { cn } from "@/lib/utils";
import { RULE_STYLES, ruleLabel } from "./rule-styles";

interface Mark {
  start: number;
  end: number;
  rule: string;
  /** Clean text only: what the extractor had there. */
  before?: string;
}

/**
 * Page text with each rule's work highlighted. Marks come from the engine
 * (the exact offsets each rule changed); this only paints them. Where two
 * marks overlap (a ligature inside a removed footer), the outer one wins.
 * A mark of zero length is something removed, shown as a small tag.
 */
export function MarkedText({
  text,
  marks,
  hidden,
  className,
}: {
  text: string;
  marks: Mark[];
  /** Rules whose highlights are switched off in the legend. */
  hidden: Set<string>;
  className?: string;
}) {
  const sorted = [...marks]
    .filter((m) => !hidden.has(m.rule) && RULE_STYLES[m.rule])
    .sort((a, b) => a.start - b.start || b.end - a.end);
  const parts: React.ReactNode[] = [];
  let cursor = 0;
  sorted.forEach((m, i) => {
    if (m.start < cursor) return;
    if (m.start > cursor) parts.push(<Fragment key={`t${i}`}>{text.slice(cursor, m.start)}</Fragment>);
    const style = RULE_STYLES[m.rule];
    const title = m.before !== undefined
      ? `${ruleLabel(m.rule)}: was “${m.before.replace(/\n/g, "⏎")}”`
      : ruleLabel(m.rule);
    if (m.end === m.start) {
      parts.push(
        <sup key={`m${i}`} className={cn("mx-px rounded px-0.5 text-[9px]", style.mark)} title={title}>
          <span className="sr-only">{title}</span>
          <span aria-hidden>{m.before?.trim() || "×"}</span>
        </sup>,
      );
    } else {
      parts.push(
        <mark key={`m${i}`} className={cn("rounded-[3px] px-px", style.mark)} title={title}>
          {text.slice(m.start, m.end)}
        </mark>,
      );
    }
    cursor = Math.max(cursor, m.end);
  });
  if (cursor < text.length) parts.push(<Fragment key="tail">{text.slice(cursor)}</Fragment>);
  return (
    <div className={cn("font-mono text-[11.5px] leading-relaxed break-words whitespace-pre-wrap", className)}>
      {parts.length ? parts : <span className="text-muted-foreground">No text on this page.</span>}
    </div>
  );
}
