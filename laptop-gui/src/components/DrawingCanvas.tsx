import { useRef, useCallback, useEffect } from 'react';
import type { MutableRefObject } from 'react';

interface DrawingCanvasProps {
  pixels: number[];
  isHiRes: boolean;
  hiResData: MutableRefObject<number[]>;
  isDrawing: MutableRefObject<boolean>;
  onPaintHiRes: (x: number, y: number) => void;
  onPaintLoRes: (gx: number, gy: number, value: number) => void;
  onClear: () => void;
}

const GRID_SIZE = 6;
const CANVAS_PX = 360; // Pixel size of the canvas element

export function DrawingCanvas({
  pixels,
  isHiRes,
  hiResData,
  isDrawing,
  onPaintHiRes,
  onPaintLoRes,
  onClear,
}: DrawingCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Draw the canvas
  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, CANVAS_PX, CANVAS_PX);

    if (isHiRes) {
      // Draw 28x28 hi-res grid
      const cellSize = CANVAS_PX / 28;
      const data = hiResData.current;
      for (let y = 0; y < 28; y++) {
        for (let x = 0; x < 28; x++) {
          const val = data[y * 28 + x];
          if (val > 0.01) {
            const brightness = Math.floor(val * 255);
            ctx.fillStyle = `rgb(${brightness}, ${brightness}, ${brightness})`;
            ctx.fillRect(x * cellSize, y * cellSize, cellSize, cellSize);
          }
        }
      }
      // Draw 6x6 grid overlay
      ctx.strokeStyle = 'rgba(59, 130, 246, 0.3)';
      ctx.lineWidth = 1;
      const bigCell = CANVAS_PX / GRID_SIZE;
      for (let i = 0; i <= GRID_SIZE; i++) {
        ctx.beginPath();
        ctx.moveTo(i * bigCell, 0);
        ctx.lineTo(i * bigCell, CANVAS_PX);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(0, i * bigCell);
        ctx.lineTo(CANVAS_PX, i * bigCell);
        ctx.stroke();
      }
    } else {
      // Draw 6x6 lo-res grid with large blocks
      const cellSize = CANVAS_PX / GRID_SIZE;
      for (let gy = 0; gy < GRID_SIZE; gy++) {
        for (let gx = 0; gx < GRID_SIZE; gx++) {
          const val = pixels[gy * GRID_SIZE + gx];
          const brightness = Math.floor(Math.max(0, Math.min(1, val)) * 255);
          ctx.fillStyle = `rgb(${brightness}, ${brightness}, ${brightness})`;
          ctx.fillRect(gx * cellSize + 1, gy * cellSize + 1, cellSize - 2, cellSize - 2);
        }
      }
      // Grid lines
      ctx.strokeStyle = '#334155';
      ctx.lineWidth = 1;
      for (let i = 0; i <= GRID_SIZE; i++) {
        ctx.beginPath();
        ctx.moveTo(i * cellSize, 0);
        ctx.lineTo(i * cellSize, CANVAS_PX);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(0, i * cellSize);
        ctx.lineTo(CANVAS_PX, i * cellSize);
        ctx.stroke();
      }
    }
  }, [pixels, isHiRes, hiResData]);

  useEffect(() => {
    draw();
  }, [draw]);

  const getCanvasCoords = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      if (!canvas) return null;
      const rect = canvas.getBoundingClientRect();
      const scaleX = CANVAS_PX / rect.width;
      const scaleY = CANVAS_PX / rect.height;
      const x = (e.clientX - rect.left) * scaleX;
      const y = (e.clientY - rect.top) * scaleY;
      return { x, y };
    },
    [],
  );

  const handlePaint = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const coords = getCanvasCoords(e);
      if (!coords) return;

      if (isHiRes) {
        const hx = (coords.x / CANVAS_PX) * 28;
        const hy = (coords.y / CANVAS_PX) * 28;
        onPaintHiRes(hx, hy);
      } else {
        const gx = Math.floor((coords.x / CANVAS_PX) * GRID_SIZE);
        const gy = Math.floor((coords.y / CANVAS_PX) * GRID_SIZE);
        if (gx >= 0 && gx < GRID_SIZE && gy >= 0 && gy < GRID_SIZE) {
          onPaintLoRes(gx, gy, 1.0);
        }
      }
    },
    [isHiRes, getCanvasCoords, onPaintHiRes, onPaintLoRes],
  );

  const handleMouseDown = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      isDrawing.current = true;
      handlePaint(e);
    },
    [isDrawing, handlePaint],
  );

  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (!isDrawing.current) return;
      handlePaint(e);
    },
    [isDrawing, handlePaint],
  );

  const handleMouseUp = useCallback(() => {
    isDrawing.current = false;
  }, [isDrawing]);

  // Right-click to erase
  const handleContextMenu = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      e.preventDefault();
      if (!isHiRes) {
        const coords = getCanvasCoords(e);
        if (!coords) return;
        const gx = Math.floor((coords.x / CANVAS_PX) * GRID_SIZE);
        const gy = Math.floor((coords.y / CANVAS_PX) * GRID_SIZE);
        if (gx >= 0 && gx < GRID_SIZE && gy >= 0 && gy < GRID_SIZE) {
          onPaintLoRes(gx, gy, 0);
        }
      }
    },
    [isHiRes, getCanvasCoords, onPaintLoRes],
  );

  return (
    <div className="flex flex-col items-center gap-3">
      <canvas
        ref={canvasRef}
        width={CANVAS_PX}
        height={CANVAS_PX}
        className="drawing-canvas rounded-lg"
        style={{
          width: CANVAS_PX,
          height: CANVAS_PX,
          background: '#000',
          border: '1px solid #334155',
        }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onContextMenu={handleContextMenu}
      />
      <div className="flex items-center gap-2">
        <button
          onClick={onClear}
          className="px-3 py-1 rounded text-[11px] font-medium transition-colors"
          style={{
            background: '#1e293b',
            border: '1px solid #334155',
            color: '#94a3b8',
          }}
        >
          CLEAR
        </button>
        <span className="text-[10px]" style={{ color: '#475569' }}>
          {isHiRes ? 'Draw freely (28x28 -> 6x6)' : 'Click cells (6x6 direct)'}
          {!isHiRes && ' · Right-click to erase'}
        </span>
      </div>
    </div>
  );
}
