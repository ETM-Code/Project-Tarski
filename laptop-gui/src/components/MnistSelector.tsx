import { useState, useCallback } from 'react';
import type { ClientMessage } from '../types/protocol';

interface MnistSelectorProps {
  send: (msg: ClientMessage) => void;
  currentIndex: number | null;
  totalSamples: number;
  onLoadSample: (index: number) => void;
}

export function MnistSelector({
  send,
  currentIndex,
  totalSamples,
  onLoadSample,
}: MnistSelectorProps) {
  const [inputIndex, setInputIndex] = useState('');

  const loadByIndex = useCallback(
    (idx: number) => {
      if (idx >= 0 && (totalSamples === 0 || idx < totalSamples)) {
        send({ type: 'LoadSample', index: idx });
        onLoadSample(idx);
      }
    },
    [send, totalSamples, onLoadSample],
  );

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const idx = parseInt(inputIndex, 10);
    if (!isNaN(idx)) {
      loadByIndex(idx);
      setInputIndex('');
    }
  };

  return (
    <div
      className="flex flex-col gap-2 p-3 rounded-lg"
      style={{ background: '#1e293b', border: '1px solid #334155' }}
    >
      <div className="flex items-center justify-between">
        <span
          className="text-[11px] font-bold tracking-wider"
          style={{ color: '#94a3b8' }}
        >
          MNIST TEST SET
        </span>
        {currentIndex !== null && (
          <span className="text-[10px]" style={{ color: '#475569' }}>
            #{currentIndex}
            {totalSamples > 0 && ` / ${totalSamples}`}
          </span>
        )}
      </div>

      <form onSubmit={handleSubmit} className="flex items-center gap-2">
        <input
          type="number"
          value={inputIndex}
          onChange={(e) => setInputIndex(e.target.value)}
          placeholder="Index"
          min={0}
          max={totalSamples > 0 ? totalSamples - 1 : undefined}
          className="flex-1 px-2 py-1 rounded text-[11px]"
          style={{
            background: '#0f172a',
            border: '1px solid #334155',
            color: '#e2e8f0',
            fontFamily: 'inherit',
          }}
        />
        <button
          type="submit"
          className="px-3 py-1 rounded text-[11px] font-bold transition-opacity hover:opacity-80"
          style={{ background: '#1d4ed8', color: '#fff' }}
        >
          LOAD
        </button>
      </form>

      {/* Navigation buttons */}
      <div className="flex items-center gap-2">
        <button
          onClick={() => loadByIndex(Math.max(0, (currentIndex ?? 0) - 1))}
          className="flex-1 py-1 rounded text-[11px] font-medium transition-opacity hover:opacity-80"
          style={{ background: '#0f172a', border: '1px solid #334155', color: '#94a3b8' }}
        >
          PREV
        </button>
        <button
          onClick={() => loadByIndex((currentIndex ?? -1) + 1)}
          className="flex-1 py-1 rounded text-[11px] font-medium transition-opacity hover:opacity-80"
          style={{ background: '#0f172a', border: '1px solid #334155', color: '#94a3b8' }}
        >
          NEXT
        </button>
        <button
          onClick={() => loadByIndex(Math.floor(Math.random() * Math.max(1, totalSamples)))}
          className="flex-1 py-1 rounded text-[11px] font-medium transition-opacity hover:opacity-80"
          style={{ background: '#0f172a', border: '1px solid #334155', color: '#94a3b8' }}
        >
          RANDOM
        </button>
      </div>

      <div className="text-[10px]" style={{ color: '#475569' }}>
        Arrow keys: Prev/Next
      </div>
    </div>
  );
}
