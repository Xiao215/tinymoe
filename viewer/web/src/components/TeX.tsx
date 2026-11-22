import katex from "katex";
import "katex/dist/katex.min.css";
import { useMemo } from "react";

/** Inline (or display) LaTeX, typeset with KaTeX. Bad input shows the source in red
 * rather than throwing. */
export function TeX({ children, block = false }: { children: string; block?: boolean }) {
  const html = useMemo(
    () => katex.renderToString(children, { displayMode: block, throwOnError: false, strict: "ignore" }),
    [children, block],
  );
  return <span className={block ? "tex tex-block" : "tex"} dangerouslySetInnerHTML={{ __html: html }} />;
}
