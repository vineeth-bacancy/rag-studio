import Link from "next/link";
import { ArrowRight } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { PageHeader } from "@/components/common/page-header";
import { STAGES } from "@/lib/stages";
import { cn } from "@/lib/utils";

export default function LearnPage() {
  return (
    <div className="mx-auto max-w-6xl space-y-8">
      <PageHeader href="/learn" />

      <p className="max-w-prose text-sm text-pretty text-muted-foreground">
        A question goes in at the top and a cited answer comes out at the bottom. Each step below gets an
        interactive section on your own three reports. Each card names a real case from these PDFs, from the
        project you built by hand.
      </p>

      <ol className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
        {STAGES.map((stage, i) => {
          const open = Boolean(stage.explain);
          return (
            <li key={stage.slug} id={stage.slug} className="scroll-mt-20">
              <Card
                className={cn(
                  "relative h-full",
                  open && "transition-colors has-[a:focus-visible]:ring-[3px] has-[a:focus-visible]:ring-ring/50 hover:bg-muted/30",
                )}
              >
                <CardHeader className="gap-2">
                  <div className="flex items-center justify-between gap-2">
                    <span
                      className="flex size-7 items-center justify-center rounded-full bg-muted text-xs font-semibold tabular-nums"
                      aria-hidden
                    >
                      {i + 1}
                    </span>
                    {open ? (
                      <Badge variant="secondary" className="font-normal">
                        Interactive
                      </Badge>
                    ) : (
                      <Badge variant="outline" className="font-normal text-muted-foreground">
                        Interactive in {stage.arrives}
                      </Badge>
                    )}
                  </div>
                  <CardTitle as="h2" className="text-base">
                    <span className="sr-only">Step {i + 1}: </span>
                    {open ? (
                      // The link covers the whole card (after:inset-0), so the
                      // card is one target and one tab stop.
                      <Link
                        href={`/learn/${stage.slug}`}
                        className="after:absolute after:inset-0 after:rounded-xl focus-visible:outline-none"
                      >
                        {stage.title}
                      </Link>
                    ) : (
                      stage.title
                    )}
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <p className="text-sm text-pretty">{stage.what}</p>
                  <p className="border-l-2 pl-3 text-xs text-pretty text-muted-foreground">{stage.example}</p>
                  {open && (
                    <p className="flex items-center gap-1 text-xs font-medium" aria-hidden>
                      Try it on your documents
                      <ArrowRight className="size-3.5" />
                    </p>
                  )}
                </CardContent>
              </Card>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
