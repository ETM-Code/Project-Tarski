import { useRef, useEffect } from 'react';
import type { SerialEntry } from '../hooks/useWebSocket';

interface SerialMonitorProps {
  entries: SerialEntry[];
}

function hexByte(b: number): string {
  return b.toString(16).padStart(2, '0').toUpperCase();
}

function formatSignal(b: number): string {
  switch (b) {
    case 0x04: return 'TRN_END';
    case 0x05: return 'SIG';
    case 0x06: return 'ACK';
    case 0x15: return 'NAK';
    case 0x44: return 'DAC';
    case 0x46: return 'FLAG';
    case 0x4d: return 'MEAS';
    case 0x4f: return 'OUT';
    case 0x53: return 'SYN';
    default: return hexByte(b);
  }
}

export function SerialMonitor({ entries }: SerialMonitorProps) {
  const logRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    logRef.current?.scrollTo(0, logRef.current.scrollHeight);
  }, [entries]);

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
          SERIAL MONITOR
        </span>
        <span className="text-[10px]" style={{ color: '#475569' }}>
          {entries.length} messages
        </span>
      </div>

      <div
        ref={logRef}
        className="overflow-y-auto font-mono text-[11px] leading-relaxed"
        style={{
          maxHeight: 160,
          background: '#0f172a',
          padding: 8,
          borderRadius: 4,
          border: '1px solid #1e293b',
        }}
      >
        {entries.length === 0 && (
          <span style={{ color: '#475569' }}>Waiting for serial data...</span>
        )}
        {entries.map((entry, i) => (
          <div key={i} className="flex gap-2">
            <span
              style={{
                color: entry.dir === 'TX' ? '#34d399' : '#3b82f6',
                minWidth: 20,
              }}
            >
              {entry.dir}
            </span>
            <span style={{ color: '#475569', minWidth: 55 }}>
              {entry.time.toFixed(1)}ms
            </span>
            <span style={{ color: '#e2e8f0' }}>
              {entry.bytes.map(formatSignal).join(' ')}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
