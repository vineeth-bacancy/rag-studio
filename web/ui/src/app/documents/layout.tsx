import type { Metadata } from "next";

import { ROUTE_TITLES } from "@/lib/routes";

/** The tab title. `metadata` needs a Server Component and the page is a client one. */
export const metadata: Metadata = { title: ROUTE_TITLES["/documents"] };

export default function Layout({ children }: { children: React.ReactNode }) {
  return children;
}
