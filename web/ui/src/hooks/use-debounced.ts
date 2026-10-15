import { useEffect, useState } from "react";

/**
 * `value`, once it has stopped changing for `ms`. A slider fires on every
 * pixel; the chunk playground only needs to ask the engine when it settles.
 */
export function useDebounced<T>(value: T, ms = 300): T {
  const [settled, setSettled] = useState(value);
  useEffect(() => {
    const t = setTimeout(() => setSettled(value), ms);
    return () => clearTimeout(t);
  }, [value, ms]);
  return settled;
}
