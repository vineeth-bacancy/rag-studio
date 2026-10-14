import { ClipboardCheck } from "lucide-react";

import { PageHeader } from "@/components/common/page-header";
import { EmptyState } from "@/components/common/states";

export default function EvaluationPage() {
  return (
    <div className="mx-auto max-w-5xl space-y-6">
      <PageHeader href="/evaluation" />
      <EmptyState
        icon={ClipboardCheck}
        title="No evaluation runs yet"
        description="A run asks every test question with one setup and scores each answer as PASS, REFUSED or WRONG. Comparing runs is how each upgrade earns its place."
        hint="The first run, the baseline, arrives in M1 with your 11 verified questions."
      />
    </div>
  );
}
