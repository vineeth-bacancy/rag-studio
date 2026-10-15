/**
 * One colour per cleaning rule, used for its highlights in the text, its box
 * on the page and its chip in the legend. Colour never stands alone: every
 * mark has a tooltip naming the rule, and the legend names it in words.
 * Green is reserved for status (docs/product-language.md), so no rule uses it.
 */
export interface RuleStyle {
  /** Short name in the legend. */
  label: string;
  /** Classes for a highlighted span of text. */
  mark: string;
  /** Classes for the legend dot and page boxes. */
  swatch: string;
}

export const RULE_STYLES: Record<string, RuleStyle> = {
  hidden: {
    label: "Hidden text",
    mark: "bg-red-500/15 text-red-900 ring-1 ring-red-500/40 dark:text-red-200",
    swatch: "bg-red-500",
  },
  furniture: {
    label: "Header or footer",
    mark: "bg-zinc-500/15 text-muted-foreground line-through decoration-zinc-400/80",
    swatch: "bg-zinc-500",
  },
  nfkc: {
    label: "Ligature",
    mark: "bg-sky-500/20 text-sky-950 dark:text-sky-100",
    swatch: "bg-sky-500",
  },
  th: {
    label: "\"Th\" glyph",
    mark: "bg-violet-500/20 text-violet-950 dark:text-violet-100",
    swatch: "bg-violet-500",
  },
  footnote: {
    label: "Footnote marker",
    mark: "bg-orange-500/20 text-orange-950 dark:text-orange-100",
    swatch: "bg-orange-500",
  },
  spacing: {
    label: "Letter spacing",
    mark: "bg-teal-500/20 text-teal-950 dark:text-teal-100",
    swatch: "bg-teal-500",
  },
  leaders: {
    label: "Dot leaders",
    mark: "bg-slate-500/20 text-slate-950 dark:text-slate-100",
    swatch: "bg-slate-500",
  },
  hyphen: {
    label: "Split word joined",
    mark: "bg-amber-500/25 text-amber-950 dark:text-amber-100",
    swatch: "bg-amber-500",
  },
  compound: {
    label: "Compound kept",
    mark: "bg-fuchsia-500/20 text-fuchsia-950 dark:text-fuchsia-100",
    swatch: "bg-fuchsia-500",
  },
};

/** The order rules are listed in: the cleaning order (spec §4). */
export const RULE_ORDER = ["hidden", "nfkc", "th", "footnote", "spacing", "furniture", "leaders", "hyphen", "compound"];

export function ruleLabel(rule: string): string {
  return RULE_STYLES[rule]?.label ?? rule;
}
