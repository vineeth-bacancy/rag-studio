import Link from "next/link";
import type { LucideIcon } from "lucide-react";
import { AlertTriangle, Inbox } from "lucide-react";

import { Card, CardContent } from "@/components/ui/card";
import { Button, buttonVariants } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { ApiError } from "@/lib/api";

/**
 * Empty states carry an explanation, not just an absence. Three parts
 * (docs/product-language.md): what is missing, why it matters where that is not
 * obvious, and one useful action.
 */
export function EmptyState({
  icon: Icon = Inbox,
  title,
  description,
  hint,
  action,
}: {
  icon?: LucideIcon;
  title: string;
  description: string;
  hint?: React.ReactNode;
  action?: { label: string; href?: string; onClick?: () => void };
}) {
  return (
    <Card className="border-dashed">
      <CardContent className="flex flex-col items-center justify-center gap-3 px-6 py-12 text-center">
        <div className="rounded-full bg-muted p-3">
          <Icon className="size-6 text-muted-foreground" aria-hidden />
        </div>
        <div className="max-w-md space-y-1.5">
          <p className="font-medium text-balance">{title}</p>
          <p className="text-sm text-pretty text-muted-foreground">{description}</p>
          {hint && <div className="pt-1 text-xs text-muted-foreground">{hint}</div>}
        </div>
        {action &&
          (action.href ? (
            // A real <a> styled as a button: Base UI's Button expects a native
            // <button>, and wrapping a Link in it gives navigation role="button".
            <Link href={action.href} className={buttonVariants({ variant: "outline", size: "sm" })}>
              {action.label}
            </Link>
          ) : (
            <Button size="sm" variant="outline" onClick={action.onClick}>
              {action.label}
            </Button>
          ))}
      </CardContent>
    </Card>
  );
}

export function ErrorState({
  title = "Couldn't load this",
  message,
  onRetry,
  action,
  error,
}: {
  title?: string;
  message: string;
  onRetry?: () => void;
  /** Another way out than "Try again" (for example "Ask as typed"). */
  action?: React.ReactNode;
  /**
   * The thrown error, when there is one. An `ApiError` carries a status, which
   * means the API answered and is running - so "start the app service" would
   * send the reader to fix the wrong thing. Only an error with no status (a
   * network failure) means nothing answered.
   */
  error?: unknown;
}) {
  const apiMayBeDown = !(error instanceof ApiError);
  return (
    <Alert variant="destructive" live="polite">
      <AlertTriangle className="size-4" aria-hidden />
      <AlertTitle>{title}</AlertTitle>
      <AlertDescription className="space-y-2">
        <p>{message}</p>
        {apiMayBeDown && (
          <p className="text-xs">
            The app service may not be running. Start it with{" "}
            <code className="rounded bg-muted px-1 py-0.5 font-mono">./rag web</code>.
          </p>
        )}
        {(onRetry || action) && (
          <div className="flex flex-wrap gap-2">
            {onRetry && (
              <Button size="sm" variant="outline" onClick={onRetry}>
                Try again
              </Button>
            )}
            {action}
          </div>
        )}
      </AlertDescription>
    </Alert>
  );
}

export function CardSkeleton({ height = "h-32" }: { height?: string }) {
  return (
    <div role="status" aria-busy="true">
      <span className="sr-only">Loading...</span>
      <Skeleton className={`w-full ${height}`} />
    </div>
  );
}
