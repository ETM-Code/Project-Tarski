import { useState } from 'react';

interface ConnectionStatusProps {
  connected: boolean;
  wsUrl: string;
  onUrlChange: (url: string) => void;
}

export function ConnectionStatus({ connected, wsUrl, onUrlChange }: ConnectionStatusProps) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(wsUrl);

  return (
    <div className="flex items-center gap-3">
      <div className="flex items-center gap-2">
        <div
          className="w-2 h-2 rounded-full"
          style={{
            background: connected ? '#34d399' : '#f87171',
            boxShadow: connected ? '0 0 6px #34d399' : '0 0 6px #f87171',
          }}
        />
        <span
          className="text-[11px] font-medium"
          style={{ color: connected ? '#34d399' : '#f87171' }}
        >
          {connected ? 'CONNECTED' : 'DISCONNECTED'}
        </span>
      </div>

      {editing ? (
        <form
          className="flex items-center gap-1"
          onSubmit={(e) => {
            e.preventDefault();
            onUrlChange(draft);
            setEditing(false);
          }}
        >
          <input
            type="text"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            className="px-2 py-0.5 rounded text-[11px]"
            style={{
              background: '#0f172a',
              border: '1px solid #334155',
              color: '#e2e8f0',
              width: 240,
              fontFamily: 'inherit',
            }}
            autoFocus
          />
          <button
            type="submit"
            className="px-2 py-0.5 rounded text-[10px] font-bold"
            style={{ background: '#1d4ed8', color: '#fff' }}
          >
            OK
          </button>
          <button
            type="button"
            onClick={() => setEditing(false)}
            className="px-2 py-0.5 rounded text-[10px]"
            style={{ color: '#64748b' }}
          >
            Cancel
          </button>
        </form>
      ) : (
        <button
          onClick={() => {
            setDraft(wsUrl);
            setEditing(true);
          }}
          className="text-[10px] hover:underline"
          style={{ color: '#475569' }}
        >
          {wsUrl}
        </button>
      )}
    </div>
  );
}
