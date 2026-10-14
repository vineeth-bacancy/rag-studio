import {
  BookOpenText, ClipboardCheck, Database, FileText, MessageSquareText, Settings, Stethoscope,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";

import { ROUTE_TITLES } from "@/lib/routes";

/**
 * The navigation, in order, shared by the sidebar and the command palette.
 * Only the route and its icon live here; the NAME comes from `ROUTE_TITLES`.
 */
export const NAV: { group: string; items: { href: string; icon: LucideIcon }[] }[] = [
  {
    group: "Use",
    items: [
      { href: "/", icon: MessageSquareText },
      { href: "/documents", icon: FileText },
    ],
  },
  {
    group: "Learn",
    items: [
      { href: "/learn", icon: BookOpenText },
      { href: "/evaluation", icon: ClipboardCheck },
    ],
  },
  {
    group: "System",
    items: [
      { href: "/indexes", icon: Database },
      { href: "/settings", icon: Settings },
      { href: "/health", icon: Stethoscope },
    ],
  },
];

export const navLabel = (href: string) => ROUTE_TITLES[href] ?? href;

export function isActive(pathname: string, href: string) {
  return href === "/" ? pathname === "/" : pathname === href || pathname.startsWith(`${href}/`);
}
