import {
  useEffect,
  useRef,
  useState,
  type KeyboardEvent,
  type ReactNode,
  type RefObject,
} from "react";

/** Open/close state for a popover anchored in `root`: closes on an outside press or Escape. */
export function usePopover(
  root: RefObject<HTMLElement | null>,
): [boolean, (v: boolean | ((o: boolean) => boolean)) => void] {
  const [open, setOpen] = useState(false);
  useEffect(() => {
    if (!open) return;
    const onDown = (e: PointerEvent) => {
      if (root.current && !root.current.contains(e.target as Node))
        setOpen(false);
    };
    document.addEventListener("pointerdown", onDown);
    return () => document.removeEventListener("pointerdown", onDown);
  }, [open, root]);
  return [open, setOpen];
}

export const Chevron = () => (
  <svg className="chev" width="10" height="10" viewBox="0 0 10 10" aria-hidden>
    <path
      d="M2 3.5 5 6.5 8 3.5"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.3"
    />
  </svg>
);

interface SelectProps<T extends string | number> {
  label?: string;
  value: T;
  options: { value: T; label: string; hint?: ReactNode }[];
  onChange: (v: T) => void;
}

/** The one dropdown of the viewer: a labelled trigger and a listbox popover, keyboard
 * friendly (arrows, Home/End, Enter, Escape, type-ahead). Values keep their type. */
export function Select<T extends string | number>({
  label,
  value,
  options,
  onChange,
}: SelectProps<T>) {
  const root = useRef<HTMLDivElement>(null);
  const list = useRef<HTMLDivElement>(null);
  const [open, setOpen] = usePopover(root);
  const cur = Math.max(
    0,
    options.findIndex((o) => o.value === value),
  );
  const [active, setActive] = useState(cur);
  const typed = useRef({ text: "", at: 0 });

  useEffect(() => {
    if (open) setActive(cur);
  }, [open, cur]);
  useEffect(() => {
    if (open)
      list.current
        ?.querySelector<HTMLElement>(`[data-i="${active}"]`)
        ?.scrollIntoView({ block: "nearest" });
  }, [open, active]);

  const pick = (i: number) => {
    const o = options[i];
    if (o) onChange(o.value);
    setOpen(false);
  };
  const onKey = (e: KeyboardEvent) => {
    const k = e.key;
    if (!open) {
      if (k === "ArrowDown" || k === "ArrowUp" || k === "Enter" || k === " ") {
        e.preventDefault();
        setOpen(true);
      }
      return;
    }
    if (k === "Escape") setOpen(false);
    else if (k === "ArrowDown")
      setActive((i) => Math.min(options.length - 1, i + 1));
    else if (k === "ArrowUp") setActive((i) => Math.max(0, i - 1));
    else if (k === "Home") setActive(0);
    else if (k === "End") setActive(options.length - 1);
    else if (k === "Enter" || k === " ") pick(active);
    else if (k.length === 1) {
      const now = Date.now();
      typed.current = {
        text:
          (now - typed.current.at < 700 ? typed.current.text : "") +
          k.toLowerCase(),
        at: now,
      };
      const hit = options.findIndex((o) =>
        o.label.toLowerCase().startsWith(typed.current.text),
      );
      if (hit >= 0) setActive(hit);
    } else return;
    e.preventDefault();
  };

  return (
    <div className="sel" ref={root} onKeyDown={onKey}>
      {label && <span className="sel-label">{label}</span>}
      <div className="sel-anchor">
        <button
          type="button"
          className={`sel-trigger${open ? " open" : ""}`}
          onClick={() => setOpen((o) => !o)}
          aria-haspopup="listbox"
          aria-expanded={open}
          aria-label={label}
        >
          <span className="sel-value">
            {options[cur]?.label ?? String(value)}
          </span>
          <Chevron />
        </button>
        {open && (
          <div
            className="pop sel-pop"
            role="listbox"
            ref={list}
            aria-label={label}
          >
            {options.map((o, i) => (
              <div
                key={String(o.value)}
                data-i={i}
                role="option"
                aria-selected={i === cur}
                className={`opt${i === active ? " active" : ""}${i === cur ? " current" : ""}`}
                onPointerEnter={() => setActive(i)}
                onClick={() => pick(i)}
              >
                <span className="opt-check" aria-hidden>
                  {i === cur ? "✓" : ""}
                </span>
                <span className="opt-label">{o.label}</span>
                {o.hint && <span className="opt-hint">{o.hint}</span>}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
