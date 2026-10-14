import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { PageHeader } from "@/components/common/page-header";
import { STAGES } from "@/lib/stages";

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
        {STAGES.map((stage, i) => (
          <li key={stage.slug} id={stage.slug} className="scroll-mt-20">
            <Card className="h-full">
              <CardHeader className="gap-2">
                <div className="flex items-center justify-between gap-2">
                  <span
                    className="flex size-7 items-center justify-center rounded-full bg-muted text-xs font-semibold tabular-nums"
                    aria-hidden
                  >
                    {i + 1}
                  </span>
                  <Badge variant="outline" className="font-normal text-muted-foreground">
                    Interactive in {stage.arrives}
                  </Badge>
                </div>
                <CardTitle as="h2" className="text-base">
                  <span className="sr-only">Step {i + 1}: </span>
                  {stage.title}
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <p className="text-sm text-pretty">{stage.what}</p>
                <p className="border-l-2 pl-3 text-xs text-pretty text-muted-foreground">{stage.example}</p>
              </CardContent>
            </Card>
          </li>
        ))}
      </ol>
    </div>
  );
}
