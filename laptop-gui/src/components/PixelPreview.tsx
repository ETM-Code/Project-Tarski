/** Renders a small 6x6 pixel grid preview. */

interface PixelPreviewProps {
  pixels: number[];
  size?: number;
  label?: string;
  highlighted?: boolean;
  onClick?: () => void;
}

const GRID = 6;

export function PixelPreview({
  pixels,
  size = 60,
  label,
  highlighted = false,
  onClick,
}: PixelPreviewProps) {
  const cellSize = size / GRID;

  return (
    <div
      className="flex flex-col items-center gap-1"
      style={{ cursor: onClick ? 'pointer' : 'default' }}
      onClick={onClick}
    >
      <svg
        width={size}
        height={size}
        viewBox={`0 0 ${size} ${size}`}
        style={{
          border: `1px solid ${highlighted ? '#3b82f6' : '#334155'}`,
          borderRadius: 4,
          background: '#000',
        }}
      >
        {Array.from({ length: GRID }, (_, gy) =>
          Array.from({ length: GRID }, (_, gx) => {
            const val = pixels[gy * GRID + gx] ?? 0;
            const b = Math.floor(Math.max(0, Math.min(1, val)) * 255);
            return (
              <rect
                key={`${gx}-${gy}`}
                x={gx * cellSize}
                y={gy * cellSize}
                width={cellSize}
                height={cellSize}
                fill={`rgb(${b},${b},${b})`}
              />
            );
          }),
        )}
      </svg>
      {label !== undefined && (
        <span className="text-[10px]" style={{ color: '#64748b' }}>
          {label}
        </span>
      )}
    </div>
  );
}
