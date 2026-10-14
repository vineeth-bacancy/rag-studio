"use client"

import { Separator as SeparatorPrimitive } from "@base-ui/react/separator"

import { cn } from "@/lib/utils"

/**
 * `data-vertical:self-stretch` is what makes a vertical divider with no height
 * of its own fill its flex row. It is also a trap, and worth naming here.
 *
 * `align-self: stretch` only stretches an item whose cross-axis size is `auto`.
 * Give the divider a definite height as well - `h-4` - and stretch no longer
 * applies, the item falls back to the cross-axis START, and it sits flush
 * against the top of the row while every sibling is centred. That is F-020: a
 * 1px rule at `y 0..16` in a 56px header whose siblings centre on y=28,
 * reading as a stray half-line.
 *
 * So any call site that sets a height must also centre it, WITH THE SAME
 * VARIANT: `data-vertical:h-4 data-vertical:self-center`. A bare `self-center`
 * keeps both classes (tailwind-merge only dedupes within one variant) and the
 * attribute-selector default wins - measured: computed align-self `stretch`.
 */
function Separator({
  className,
  orientation = "horizontal",
  ...props
}: SeparatorPrimitive.Props) {
  return (
    <SeparatorPrimitive
      data-slot="separator"
      orientation={orientation}
      className={cn(
        "shrink-0 bg-border data-horizontal:h-px data-horizontal:w-full data-vertical:w-px data-vertical:self-stretch",
        className
      )}
      {...props}
    />
  )
}

export { Separator }
