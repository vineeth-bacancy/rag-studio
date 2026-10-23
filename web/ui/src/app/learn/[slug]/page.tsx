import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { ArrowLeft, ArrowRight, ChevronRight } from "lucide-react";

import { buttonVariants } from "@/components/ui/button";
import { ChunkLab } from "@/components/learn/chunk-lab";
import { CiteLab } from "@/components/learn/cite-lab";
import { CleanLab } from "@/components/learn/clean-lab";
import { ContextLab } from "@/components/learn/context-lab";
import { EmbedLab } from "@/components/learn/embed-lab";
import { EvaluateLab } from "@/components/learn/evaluate-lab";
import { FiguresLab } from "@/components/learn/figures-lab";
import { ParseLab } from "@/components/learn/parse-lab";
import { PromptLab } from "@/components/learn/prompt-lab";
import { RewriteLab } from "@/components/learn/rewrite-lab";
import { SafetyLab } from "@/components/learn/safety-lab";
import { SearchLab } from "@/components/learn/search-lab";
import { StoreLab } from "@/components/learn/store-lab";
import { APP_NAME, ROUTES } from "@/lib/routes";
import { STAGES } from "@/lib/stages";

/** The widget each interactive step shows. A step without one is not a page yet. */
const WIDGETS: Record<string, React.ComponentType> = {
  parse: ParseLab,
  clean: CleanLab,
  chunk: ChunkLab,
  context: ContextLab,
  embed: EmbedLab,
  store: StoreLab,
  search: SearchLab,
  rewrite: RewriteLab,
  prompt: PromptLab,
  generate: CiteLab,
  evaluate: EvaluateLab,
  safety: SafetyLab,
  figures: FiguresLab,
};

type Props = { params: Promise<{ slug: string }> };

/** Only the interactive steps are pages; any other slug is a 404. */
export const dynamicParams = false;

export function generateStaticParams() {
  return STAGES.filter((s) => s.explain && WIDGETS[s.slug]).map((s) => ({ slug: s.slug }));
}

/** "Chunk · How it works · RAG Studio". The learn layout's plain title stops
 * the root template from reaching this page, so the whole title is given. */
export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { slug } = await params;
  const stage = STAGES.find((s) => s.slug === slug);
  const learn = ROUTES["/learn"].title;
  return { title: { absolute: `${stage ? `${stage.title} · ` : ""}${learn} · ${APP_NAME}` } };
}

export default async function StepPage({ params }: Props) {
  const { slug } = await params;
  const index = STAGES.findIndex((s) => s.slug === slug);
  const stage = STAGES[index];
  const Widget = WIDGETS[slug];
  if (!stage?.explain || !Widget) notFound();

  // Previous and next among the steps that have a page, in pipeline order.
  const open = STAGES.filter((s) => s.explain && WIDGETS[s.slug]);
  const at = open.findIndex((s) => s.slug === slug);
  const prev = open[at - 1];
  const next = open[at + 1];

  return (
    <div className="mx-auto max-w-6xl space-y-6">
      <div className="space-y-2">
        <nav aria-label="Breadcrumb" className="flex items-center gap-1 text-xs text-muted-foreground">
          <Link href="/learn" className="rounded-sm underline-offset-2 hover:text-foreground hover:underline">
            {ROUTES["/learn"].title}
          </Link>
          <ChevronRight className="size-3" aria-hidden />
          <span>
            Step {index + 1} of {STAGES.length}
          </span>
        </nav>
        <h1 className="text-2xl font-semibold tracking-tight text-balance">{stage.title}</h1>
        <div className="max-w-prose space-y-1 text-sm text-pretty text-muted-foreground">
          {stage.explain.map((s) => (
            <p key={s}>{s}</p>
          ))}
        </div>
      </div>

      <Widget />

      <nav aria-label="Steps" className="flex flex-wrap items-center justify-between gap-3 border-t pt-4">
        {prev ? (
          <Link href={`/learn/${prev.slug}`} className={buttonVariants({ variant: "outline", size: "sm" })}>
            <ArrowLeft aria-hidden />
            {prev.title}
          </Link>
        ) : (
          <Link href="/learn" className={buttonVariants({ variant: "ghost", size: "sm" })}>
            <ArrowLeft aria-hidden />
            All steps
          </Link>
        )}
        {next ? (
          <Link href={`/learn/${next.slug}`} className={buttonVariants({ variant: "outline", size: "sm" })}>
            {next.title}
            <ArrowRight aria-hidden />
          </Link>
        ) : (
          <Link href="/learn" className={buttonVariants({ variant: "ghost", size: "sm" })}>
            All steps
            <ArrowRight aria-hidden />
          </Link>
        )}
      </nav>
    </div>
  );
}
