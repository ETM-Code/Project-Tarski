import { useCallback, useEffect, useRef, useState } from 'react';
import type { ServerMessage, SimFrame, BoardInfo, SimStatus, ClientMessage } from '../types/protocol';

export interface WebSocketState {
  connected: boolean;
  boardInfo: BoardInfo | null;
  frame: SimFrame | null;
  status: SimStatus | null;
  send: (msg: ClientMessage) => void;
  serialLog: SerialEntry[];
}

export interface SerialEntry {
  dir: 'TX' | 'RX';
  bytes: number[];
  time: number;
}

export function useWebSocket(wsUrl: string): WebSocketState {
  const ws = useRef<WebSocket | null>(null);
  const [connected, setConnected] = useState(false);
  const [boardInfo, setBoardInfo] = useState<BoardInfo | null>(null);
  const [frame, setFrame] = useState<SimFrame | null>(null);
  const [status, setStatus] = useState<SimStatus | null>(null);
  const [serialLog, setSerialLog] = useState<SerialEntry[]>([]);

  const send = useCallback((msg: ClientMessage) => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify(msg));
    }
  }, []);

  useEffect(() => {
    let reconnectTimer: ReturnType<typeof setTimeout>;

    const connect = () => {
      try {
        const socket = new WebSocket(wsUrl);

        socket.onopen = () => {
          setConnected(true);
          socket.send(JSON.stringify({ type: 'GetBoardInfo' }));
        };

        socket.onclose = () => {
          setConnected(false);
          reconnectTimer = setTimeout(connect, 2000);
        };

        socket.onerror = () => {
          socket.close();
        };

        socket.onmessage = (event) => {
          try {
            const msg: ServerMessage = JSON.parse(event.data);
            switch (msg.type) {
              case 'BoardInfo':
                setBoardInfo(msg);
                break;
              case 'SimFrame':
                setFrame(msg);
                // Accumulate serial log from frames
                if (msg.uart_tx.length > 0 || msg.uart_rx.length > 0) {
                  setSerialLog((prev) => {
                    const entries: SerialEntry[] = [];
                    if (msg.uart_tx.length > 0) {
                      entries.push({ dir: 'TX', bytes: [...msg.uart_tx], time: msg.time_ms });
                    }
                    if (msg.uart_rx.length > 0) {
                      entries.push({ dir: 'RX', bytes: [...msg.uart_rx], time: msg.time_ms });
                    }
                    return [...prev.slice(-100), ...entries];
                  });
                }
                break;
              case 'Status':
                setStatus(msg);
                break;
            }
          } catch {
            // ignore parse errors
          }
        };

        ws.current = socket;
      } catch {
        reconnectTimer = setTimeout(connect, 2000);
      }
    };

    connect();

    return () => {
      clearTimeout(reconnectTimer);
      ws.current?.close();
    };
  }, [wsUrl]);

  return { connected, boardInfo, frame, status, send, serialLog };
}
