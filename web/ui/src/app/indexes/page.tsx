import { Database } from "lucide-react";

import { PageHeader } from "@/components/common/page-header";
import { EmptyState } from "@/components/common/states";

export default function IndexesPage() {
  return (
    <div className="mx-auto max-w-5xl space-y-6">
      <PageHeader href="/indexes" />
      <EmptyState
        icon={Database}
        title="No index yet"
        description="An index holds every passage of your documents as a vector, built with one embedding model. Search needs one before any question can be answered."
        hint={
          <>
            The baseline index (nomic-embed-text, the same setup you measured by hand) is built with{" "}
            <code className="rounded bg-muted px-1 py-0.5 font-mono">./rag ingest</code> in M1. Index profiles
            for newer models, and building on a Colab GPU, arrive in M4.
          </>
        }
      />
    </div>
  );
}
