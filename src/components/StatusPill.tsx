type Tone = "accent" | "success" | "warning" | "danger" | "neutral";

export function StatusPill({
  tone = "neutral",
  label,
}: {
  tone?: Tone;
  label: string;
}) {
  return (
    <span className={`pill pill--${tone}`} role="status">
      <span className="pill__dot" aria-hidden="true" />
      {label}
    </span>
  );
}
