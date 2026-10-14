"use client";

import useSWR from "swr";
import { MessageSquareText } from "lucide-react";

import { PageHeader } from "@/components/common/page-header";
import { EmptyState } from "@/components/common/states";
import type { AppStatus } from "@/lib/api";

export default function AskPage() {
  const { data: status } = useSWR<AppStatus>("/status", { refreshInterval: 30_000 });

  return (
    <div className="mx-auto max-w-3xl space-y-6">
      <PageHeader href="/" />
      <EmptyState
        icon={MessageSquareText}
        title={status?.indexed ? "Asking arrives in M1" : "No search index yet"}
        description="Build the baseline index of your reports, then ask here. Answers stream in with numbered citations that open the cited page."
        hint={
          <>
            The index is built with <code className="rounded bg-muted px-1 py-0.5 font-mono">./rag ingest</code>,
            which arrives in M1.
          </>
        }
        action={{ label: "Open Indexes", href: "/indexes" }}
      />
    </div>
  );
}
