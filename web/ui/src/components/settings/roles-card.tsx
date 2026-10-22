import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { StatusBadge } from "@/components/common/status";
import type { RoleRow } from "@/lib/api";

/**
 * Each role, its one model, and whether that model is ready. Shown on System
 * Health (is it installed?) and on Settings (what will answer?).
 *
 * A list, not a table: at 375px a three-column table squeezed the purpose
 * into a one-word column and pushed the status off-screen (QA M0-03). Rows
 * stack on phones and line up as columns from `sm` up.
 */
export function RolesCard({ roles }: { roles: RoleRow[] }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle as="h2" className="text-base">Models per role</CardTitle>
        <CardDescription>Each job the app gives a model, and the one model that does it.</CardDescription>
      </CardHeader>
      <CardContent>
        <ul className="divide-y">
          {roles.map((r) => (
            <li
              key={r.role}
              className="grid gap-x-6 gap-y-1.5 py-3 first:pt-0 last:pb-0 sm:grid-cols-[minmax(0,1fr)_12rem_8.5rem] sm:items-start"
            >
              <div className="min-w-0">
                <p className="text-sm font-medium">{r.label}</p>
                <p className="text-xs text-pretty text-muted-foreground">{r.purpose}</p>
              </div>
              <div className="space-y-0.5 sm:pt-0.5">
                <p className="font-mono text-xs break-all">
                  {r.model ?? <span className="font-sans text-muted-foreground">Not set yet</span>}
                </p>
                {r.model != null && (
                  <p className="text-[11px] text-muted-foreground">
                    {r.where === "remote" ? "Remote AI only" : r.runs_on === "python" ? "Local AI, in this app" : "Local AI"}
                  </p>
                )}
              </div>
              <div className="sm:text-right">
                <RoleStatus role={r} />
              </div>
            </li>
          ))}
        </ul>
      </CardContent>
    </Card>
  );
}

function RoleStatus({ role }: { role: RoleRow }) {
  if (role.model == null) {
    return (
      <Badge variant="outline" className="font-normal text-muted-foreground">
        Arrives in {role.arrives}
      </Badge>
    );
  }
  if (role.where === "remote") {
    // Runs on the Colab runtime only, with no local fallback: never "Not installed" here.
    if (!role.connected) return <StatusBadge status="info" label="Needs Remote AI" />;
    if (role.installed == null) return <StatusBadge status="unavailable" label="Unknown" />;
    if (!role.installed) return <StatusBadge status="needs_attention" label="Not on the runtime" />;
    return <StatusBadge status="healthy" label="On Remote AI" />;
  }
  if (role.installed == null) return <StatusBadge status="unavailable" label="Unknown" />;
  const has = role.runs_on === "python" ? "Downloaded" : "Installed";
  if (!role.installed) return <StatusBadge status="needs_attention" label={role.runs_on === "python" ? "Not downloaded" : "Not installed"} />;
  return <StatusBadge status="healthy" label={role.loaded ? "Loaded" : has} />;
}
