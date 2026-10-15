"use client";

import useSWR from "swr";

import { PageHeader } from "@/components/common/page-header";
import { CardSkeleton, ErrorState } from "@/components/common/states";
import { ModelBackend } from "@/components/settings/model-backend";
import { RolesCard } from "@/components/settings/roles-card";
import type { HealthReport } from "@/lib/api";

export default function SettingsPage() {
  const { data, error, isLoading, mutate } = useSWR<HealthReport>("/health");

  return (
    <div className="mx-auto max-w-5xl space-y-6">
      <PageHeader href="/settings" />
      <ModelBackend />
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
