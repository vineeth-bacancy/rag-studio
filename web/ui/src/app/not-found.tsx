import Link from "next/link";
import { Compass } from "lucide-react";

import { EmptyState } from "@/components/common/states";
import { ROUTE_TITLES } from "@/lib/routes";

/** Renders inside the root layout, so the sidebar stays and every page is one click away. */
export default function NotFound() {
  return (
    <div className="mx-auto max-w-2xl space-y-4">
      <h1 className="sr-only">Page not found</h1>
      <EmptyState
        icon={Compass}
        title="There is no page at this address"
        description="The address is wrong. Nothing is broken: every page is listed below."
        action={{ label: "Go to Ask", href: "/" }}
      />
      <nav aria-label="All pages">
        <ul className="flex flex-wrap justify-center gap-1.5">
          {Object.entries(ROUTE_TITLES).map(([href, label]) => (
            <li key={href}>
              <Link
                href={href}
                className="inline-flex rounded-md border bg-background px-2.5 py-1 text-xs hover:bg-accent focus-visible:ring-[3px] focus-visible:ring-ring/50 focus-visible:outline-none"
              >
                {label}
              </Link>
            </li>
          ))}
        </ul>
      </nav>
    </div>
  );
}
