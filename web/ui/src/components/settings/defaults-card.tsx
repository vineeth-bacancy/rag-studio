import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import type { TechniqueDefault } from "@/lib/api";

/**
 * What Ask and an evaluation run use unless you pick otherwise (spec §9.6).
 * Read-only on purpose: every technique is a switch on Ask and Evaluation,
 * and a default changes only when the scorecard says so (docs/ladder.md).
 */
export function DefaultsCard({ defaults }: { defaults: TechniqueDefault[] }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle as="h2" className="text-base">Defaults</CardTitle>
        <CardDescription>
          What Ask and a new run use unless you choose otherwise there. A default changes only when the scorecard
          shows it helps; the ladder on Evaluation has the numbers.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <dl className="grid gap-x-6 gap-y-2 text-sm sm:grid-cols-[12rem_minmax(0,1fr)]">
          {defaults.map((d) => (
            <div key={d.key} className="contents">
              <dt className="text-muted-foreground">{d.label}</dt>
              <dd className="min-w-0 break-words">
                {d.value}
                {d.id && <span className="ml-2 font-mono text-xs text-muted-foreground">{d.id}</span>}
              </dd>
            </div>
          ))}
        </dl>
      </CardContent>
    </Card>
  );
}
