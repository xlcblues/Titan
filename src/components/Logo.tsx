export function LogoMark({
  size = 24,
  className,
}: {
  size?: number;
  className?: string;
}) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 512 512"
      fill="none"
      role="img"
      aria-label="Titan"
      className={className}
    >
      <g fill="currentColor">
        <path d="M 0 0 H 132 V 44 H 88 V 132 H 44 V 44 H 0 Z" transform="translate(196 236)" />
        <path d="M 0 0 L 132 0 L 180 24 L 48 24 Z" transform="translate(196 236)" opacity="0.55" />
        <path d="M 132 0 L 180 24 L 180 68 L 132 44 Z" transform="translate(196 236)" opacity="0.3" />
        <path d="M 88 44 L 136 68 L 136 156 L 88 132 Z" transform="translate(196 236)" opacity="0.3" />
      </g>
      <path
        d="M 113.7 209.6 A 175 175 0 0 1 458.3 209.6"
        fill="none"
        stroke="currentColor"
        strokeWidth="20"
        strokeLinecap="round"
        opacity="0.6"
      />
      <circle cx="458.3" cy="209.6" r="13" fill="currentColor" opacity="0.85" />
    </svg>
  );
}

export function LogoWordmark({
  size = 24,
  className,
}: {
  size?: number;
  className?: string;
}) {
  return (
    <span className={`logo-wordmark ${className ?? ""}`}>
      <LogoMark size={size} />
      <span className="logo-wordmark__text">Titan</span>
    </span>
  );
}
