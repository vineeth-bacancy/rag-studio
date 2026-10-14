import { ROUTES } from "@/lib/routes";
import { cn } from "@/lib/utils";

/**
 * The page's one <h1> and its one-line purpose, both read from `ROUTES`, so
 * the heading cannot disagree with the sidebar, the palette or the tab title.
 */
export function PageHeader({
  href,
  actions,
  className,
}: {
  href: string;
  actions?: React.ReactNode;
  className?: string;
}) {
  const route = ROUTES[href];
  return (
    <div className={cn("flex flex-wrap items-end justify-between gap-x-6 gap-y-3", className)}>
      <div className="min-w-0 space-y-1">
        <h1 className="text-2xl font-semibold tracking-tight text-balance">{route?.title ?? href}</h1>
        {route?.summary && <p className="max-w-prose text-sm text-pretty text-muted-foreground">{route.summary}</p>}
      </div>
      {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
    </div>
  );
}
