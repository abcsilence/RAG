// The double-pennant shape of Nepal's flag.
export default function Logo({ size = 28 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden="true">
      <path
        d="M7 3 L24 15.5 H12 L25 29 H7 Z"
        fill="#DC143C"
        stroke="#003893"
        strokeWidth="2.2"
        strokeLinejoin="round"
      />
    </svg>
  )
}
