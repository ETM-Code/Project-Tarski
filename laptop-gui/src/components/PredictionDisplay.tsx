interface PredictionDisplayProps {
  prediction: number | null;
  trueLabel: number | null;
  correct: boolean | null;
  spikeCounts: number[];
  timestep: number;
  timeMs: number;
}

export function PredictionDisplay({
  prediction,
  trueLabel,
  correct,
  spikeCounts,
  timestep,
  timeMs,
}: PredictionDisplayProps) {
  const hasPrediction = prediction !== null && prediction !== undefined;
  const maxCount = Math.max(...spikeCounts, 1);

  // Determine color based on correctness
  const predColor =
    correct === true
      ? '#34d399'
      : correct === false
        ? '#f87171'
        : '#e2e8f0';
  const borderColor =
    correct === true
      ? '#059669'
      : correct === false
        ? '#dc2626'
        : '#334155';

  return (
    <div
      className="flex flex-col gap-3 p-4 rounded-lg"
      style={{
        background: '#1e293b',
        border: `2px solid ${borderColor}`,
      }}
    >
      {/* Big prediction digit */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div
            className="text-6xl font-bold leading-none"
            style={{
              color: hasPrediction ? predColor : '#334155',
              textShadow: hasPrediction
                ? `0 0 20px ${predColor}40`
                : 'none',
            }}
          >
            {hasPrediction ? prediction : '?'}
          </div>

          <div className="flex flex-col gap-1">
            <span
              className="text-[12px] font-bold tracking-wider"
              style={{ color: '#94a3b8' }}
            >
              PREDICTION
            </span>
            {trueLabel !== null && trueLabel !== undefined && (
              <div className="flex items-center gap-2">
                <span className="text-[11px]" style={{ color: '#64748b' }}>
                  True:{' '}
                  <span className="font-bold" style={{ color: '#e2e8f0' }}>
                    {trueLabel}
                  </span>
                </span>
                {correct !== null && (
                  <span
                    className="text-[11px] font-bold px-1.5 py-0.5 rounded"
                    style={{
                      background:
                        correct
                          ? 'rgba(5,150,105,0.2)'
                          : 'rgba(220,38,38,0.2)',
                      color: correct ? '#6ee7b7' : '#fca5a5',
                    }}
                  >
                    {correct ? 'CORRECT' : 'WRONG'}
                  </span>
                )}
              </div>
            )}
            <span className="text-[10px]" style={{ color: '#475569' }}>
              {timestep} steps / {timeMs.toFixed(1)}ms
            </span>
          </div>
        </div>
      </div>

      {/* Spike count bars */}
      {spikeCounts.length > 0 && (
        <div className="flex flex-col gap-1">
          <span
            className="text-[10px] font-bold tracking-wider mb-1"
            style={{ color: '#64748b' }}
          >
            OUTPUT SPIKE COUNTS
          </span>
          <div className="flex items-end gap-1" style={{ height: 48 }}>
            {spikeCounts.map((count, i) => {
              const height = Math.max(2, (count / maxCount) * 44);
              const isWinner = i === prediction;
              return (
                <div
                  key={i}
                  className="flex flex-col items-center gap-0.5 flex-1"
                >
                  <div
                    className="w-full rounded-t"
                    style={{
                      height,
                      background: isWinner ? '#3b82f6' : '#334155',
                      transition: 'height 0.15s ease-out',
                    }}
                  />
                  <span
                    className="text-[9px]"
                    style={{
                      color: isWinner ? '#93c5fd' : '#475569',
                      fontWeight: isWinner ? 700 : 400,
                    }}
                  >
                    {i}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
