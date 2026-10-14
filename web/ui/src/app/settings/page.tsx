"use client";

import useSWR from "swr";
import { Cloud, Laptop } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { PageHeader } from "@/components/common/page-header";
import { CardSkeleton, ErrorState } from "@/components/common/states";
import { StatusBadge } from "@/components/common/status";
import { TechnicalDetails } from "@/components/common/technical-details";
import { RolesCard } from "@/components/settings/roles-card";
import type { HealthReport } from "@/lib/api";

export default function SettingsPage() {
  const { data, error, isLoading, mutate } = useSWR<HealthReport>("/health");

  return (
    <div className="mx-auto max-w-5xl space-y-6">
      <PageHeader href="/settings" />

      <Card id="model-backend" className="scroll-mt-20">
        <CardHeader>
          <CardTitle as="h2" className="text-base">Model backend</CardTitle>
          <CardDescription>Which computer runs the AI models.</CardDescription>
        </CardHeader>
        <CardContent className="grid gap-3 sm:grid-cols-2">
          <div className="flex items-start gap-3 rounded-lg ring-1 ring-foreground/10 p-4">
            <Laptop className="mt-0.5 size-5 shrink-0 text-muted-foreground" aria-hidden />
            <div className="min-w-0 space-y-1.5">
              <div className="flex flex-wrap items-center gap-2">
                <p className="font-medium">Local AI</p>
                {data &&
                  (data.ollama.up ? (
                    <StatusBadge status="healthy" label="In use" />
                  ) : (
                    <StatusBadge status="unavailable" />
                  ))}
              </div>
              <p className="text-sm text-muted-foreground">
                Ollama on this computer. Private and always available, but slow for large models.
              </p>
              {data && <TechnicalDetails rows={{ Endpoint: data.ollama.host, Version: data.ollama.version ?? "-" }} />}
            </div>
          </div>
          <div className="flex items-start gap-3 rounded-lg border border-dashed p-4">
            <Cloud className="mt-0.5 size-5 shrink-0 text-muted-foreground" aria-hidden />
            <div className="min-w-0 space-y-1.5">
              <div className="flex flex-wrap items-center gap-2">
                <p className="font-medium">Remote AI</p>
                <Badge variant="outline" className="font-normal text-muted-foreground">Arrives in M4</Badge>
              </div>
              <p className="text-sm text-muted-foreground">
                A Colab GPU you start yourself, for building large indexes, reranking, figures and judging.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {error ? (
        <ErrorState message={error.message} error={error} onRetry={() => mutate()} />
      ) : isLoading || !data ? (
        <CardSkeleton height="h-64" />
      ) : (
        <RolesCard roles={data.roles} />
      )}
    </div>
  );
}
