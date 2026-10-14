import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"

import { cn } from "@/lib/utils"

const alertVariants = cva(
  "group/alert relative grid w-full gap-0.5 rounded-lg border px-2.5 py-2 text-left text-sm has-data-[slot=alert-action]:relative has-data-[slot=alert-action]:pr-18 has-[>svg]:grid-cols-[auto_1fr] has-[>svg]:gap-x-2 *:[svg]:row-span-2 *:[svg]:translate-y-0.5 *:[svg]:text-current *:[svg:not([class*='size-'])]:size-4",
  {
    variants: {
      variant: {
        default: "bg-card text-card-foreground",
        destructive:
          "bg-card text-destructive *:data-[slot=alert-description]:text-destructive/90 *:[svg]:text-current",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
)

/**
 * How urgently this panel should be announced. Deviates from stock shadcn,
 * which hardcodes role="alert" on every Alert.
 *
 * `role="alert"` is an assertive live region: it interrupts whatever a screen
 * reader is currently saying. That is right for a validation error the user just
 * provoked, and wrong for a panel that is simply part of the page - and in this
 * app most Alerts are the latter (explaining that the journal is written by
 * `./trade journal`, that the ledger has no trades, where sizing is documented).
 * With the role always on, opening /journal interrupted the reader three times
 * to recite standing prose. Three call sites already re-stated role="alert"
 * explicitly, which is a good sign the narrow behaviour was the intended one.
 *
 *   false (default) - not a live region. Read in document order like any content
 *   "polite"        - role="status": announced when the reader is next idle.
 *                     For panels that appear in response to something changing
 *   "assertive"     - role="alert": interrupts. For errors the user must fix
 *                     before continuing
 *
 * A live region only announces if it is a live region BEFORE its content
 * arrives; every use here mounts the whole panel conditionally, which screen
 * readers announce on insertion.
 */
type AlertLive = false | "polite" | "assertive"

const LIVE_ROLE: Record<"polite" | "assertive", "status" | "alert"> = {
  polite: "status",
  assertive: "alert",
}

function Alert({
  className,
  variant,
  live = false,
  ...props
}: React.ComponentProps<"div"> &
  VariantProps<typeof alertVariants> & { live?: AlertLive }) {
  return (
    <div
      data-slot="alert"
      // Spread last so an explicit role on the call site still wins.
      role={live ? LIVE_ROLE[live] : undefined}
      className={cn(alertVariants({ variant }), className)}
      {...props}
    />
  )
}

function AlertTitle({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="alert-title"
      className={cn(
        "font-medium group-has-[>svg]/alert:col-start-2 [&_a]:underline [&_a]:underline-offset-3 [&_a]:hover:text-foreground",
        className
      )}
      {...props}
    />
  )
}

function AlertDescription({
  className,
  ...props
}: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="alert-description"
      className={cn(
        "text-sm text-balance text-muted-foreground md:text-pretty [&_a]:underline [&_a]:underline-offset-3 [&_a]:hover:text-foreground [&_p:not(:last-child)]:mb-4",
        className
      )}
      {...props}
    />
  )
}

function AlertAction({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="alert-action"
      className={cn("absolute top-2 right-2", className)}
      {...props}
    />
  )
}

export { Alert, AlertTitle, AlertDescription, AlertAction }
