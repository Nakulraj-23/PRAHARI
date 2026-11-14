import { useCallback, useEffect, useRef, useState } from 'react'

/** Fetch + optional polling. Returns {data, error, loading, reload}. */
export function useApi(fn, deps = [], pollMs = 0) {
  const [state, setState] = useState({ data: null, error: null, loading: true })
  const fnRef = useRef(fn)
  fnRef.current = fn
  const load = useCallback(() => {
    return fnRef.current()
      .then((data) => setState({ data, error: null, loading: false }))
      .catch((error) => setState((s) => ({ data: s.data, error, loading: false })))
  }, [])
  useEffect(() => {
    setState((s) => ({ ...s, loading: true }))
    load()
    if (!pollMs) return undefined
    const t = setInterval(load, pollMs)
    return () => clearInterval(t)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, pollMs])
  return { ...state, reload: load }
}
