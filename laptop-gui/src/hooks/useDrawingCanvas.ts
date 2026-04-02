import { useCallback, useRef, useState } from 'react';

/** 6x6 pixel grid, values in [-1, 1] range (normalized). */
export type PixelGrid = number[];

const GRID_SIZE = 6;

export function useDrawingCanvas() {
  const [pixels, setPixels] = useState<PixelGrid>(new Array(36).fill(0));
  const [isHiRes, setIsHiRes] = useState(true);
  const hiResRef = useRef<number[]>(new Array(28 * 28).fill(0));
  const isDrawing = useRef(false);

  const clear = useCallback(() => {
    setPixels(new Array(36).fill(0));
    hiResRef.current = new Array(28 * 28).fill(0);
  }, []);

  /** Downscale 28x28 to 6x6 using average pooling. */
  const downscale28to6 = useCallback((src: number[]): number[] => {
    const out = new Array(36).fill(0);
    // Each 6x6 cell covers roughly 4.67 pixels of the 28x28 grid
    // Use simple area averaging
    for (let gy = 0; gy < GRID_SIZE; gy++) {
      for (let gx = 0; gx < GRID_SIZE; gx++) {
        const yStart = Math.floor((gy * 28) / GRID_SIZE);
        const yEnd = Math.floor(((gy + 1) * 28) / GRID_SIZE);
        const xStart = Math.floor((gx * 28) / GRID_SIZE);
        const xEnd = Math.floor(((gx + 1) * 28) / GRID_SIZE);
        let sum = 0;
        let count = 0;
        for (let y = yStart; y < yEnd; y++) {
          for (let x = xStart; x < xEnd; x++) {
            sum += src[y * 28 + x];
            count++;
          }
        }
        out[gy * GRID_SIZE + gx] = count > 0 ? sum / count : 0;
      }
    }
    return out;
  }, []);

  /** Paint on the 28x28 hi-res canvas at (x, y) in [0, 28) space. */
  const paintHiRes = useCallback(
    (x: number, y: number) => {
      const grid = hiResRef.current;
      // Brush: paint a 3x3 area with gaussian-ish falloff
      const cx = Math.floor(x);
      const cy = Math.floor(y);
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const px = cx + dx;
          const py = cy + dy;
          if (px >= 0 && px < 28 && py >= 0 && py < 28) {
            const dist = Math.abs(dx) + Math.abs(dy);
            const intensity = dist === 0 ? 1.0 : dist === 1 ? 0.6 : 0.3;
            const idx = py * 28 + px;
            grid[idx] = Math.min(1.0, grid[idx] + intensity);
          }
        }
      }
      hiResRef.current = grid;
      setPixels(downscale28to6(grid));
    },
    [downscale28to6],
  );

  /** Paint on the 6x6 lo-res canvas directly. */
  const paintLoRes = useCallback((gx: number, gy: number, value: number) => {
    setPixels((prev) => {
      const next = [...prev];
      const idx = gy * GRID_SIZE + gx;
      if (idx >= 0 && idx < 36) {
        next[idx] = Math.min(1.0, Math.max(0, value));
      }
      return next;
    });
  }, []);

  /** Set pixels from an external 6x6 source. */
  const setFromExternal = useCallback((p: number[]) => {
    setPixels(p.slice(0, 36));
    // Also update hi-res to match (upscale for display consistency)
    const grid = new Array(28 * 28).fill(0);
    for (let gy = 0; gy < GRID_SIZE; gy++) {
      for (let gx = 0; gx < GRID_SIZE; gx++) {
        const val = p[gy * GRID_SIZE + gx] ?? 0;
        const yStart = Math.floor((gy * 28) / GRID_SIZE);
        const yEnd = Math.floor(((gy + 1) * 28) / GRID_SIZE);
        const xStart = Math.floor((gx * 28) / GRID_SIZE);
        const xEnd = Math.floor(((gx + 1) * 28) / GRID_SIZE);
        for (let y = yStart; y < yEnd; y++) {
          for (let x = xStart; x < xEnd; x++) {
            grid[y * 28 + x] = val;
          }
        }
      }
    }
    hiResRef.current = grid;
  }, []);

  /** Set pixels from an external 28x28 source (real MNIST hi-res) + matching 6x6. */
  const setFromExternal28 = useCallback((p6x6: number[], p28x28: number[]) => {
    setPixels(p6x6.slice(0, 36));
    hiResRef.current = p28x28.slice(0, 784);
  }, []);

  return {
    pixels,
    isHiRes,
    setIsHiRes,
    hiResData: hiResRef,
    isDrawing,
    clear,
    paintHiRes,
    paintLoRes,
    setFromExternal,
    setFromExternal28,
  };
}
