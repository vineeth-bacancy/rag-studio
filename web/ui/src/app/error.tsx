"use client";

/**
 * The route-level error boundary. Sitting at the app root it renders INSIDE the
 * root layout, so the sidebar and every other page stay reachable when one
 * page fails. Next 16 names the recovery function `retry` (it was `reset`).
 */

import { useEffect } from "react";
import { AlertTriangle, RefreshCw, RotateCcw } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";

export default function RouteError({ error, retry }: { error: Error & { digest?: string }; retry: () => void }) {
  useEffect(() => {
    // Nothing leaves this machine, so there is no error service: the console
    // is where the stack is worth most.
    console.error("[route error]", error);
  }, [error]);

  return (
    <div className="mx-auto max-w-2xl space-y-4">
      <div className="space-y-1">
        <h1 className="text-2xl font-semibold tracking-tight">This page could not be shown.</h1>
        <p className="text-sm text-muted-foreground">Nothing else is affected. Every other page is in the sidebar.</p>
      </div>
      <Card className="border-amber-500/40 bg-amber-500/5">
        <CardContent className="space-y-3 py-5">
          <div className="flex items-start gap-2.5">
            <AlertTriangle className="mt-0.5 size-4 shrink-0 text-amber-600 dark:text-amber-400" aria-hidden />
            <div className="min-w-0 space-y-1">
              <p className="text-sm font-medium">What happened</p>
              {/* Verbatim: a single-user local tool has nobody to hide this from. */}
              <p className="font-mono text-xs break-words text-muted-foreground">
                {error.message || "The error carried no message."}
              </p>
            </div>
          </div>
          <div className="flex flex-wrap gap-2">
            <Button size="sm" onClick={() => retry()}>
              <RotateCcw className="size-4" aria-hidden />
              Try again
            </Button>
            <Button size="sm" variant="outline" onClick={() => window.location.reload()}>
              <RefreshCw className="size-4" aria-hidden />
              Reload the page
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
