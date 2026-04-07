import { useEffect, useState, useCallback, useRef } from 'react';
import { useWebSocket } from './hooks/useWebSocket';
import { useDrawingCanvas } from './hooks/useDrawingCanvas';
import { ConnectionStatus } from './components/ConnectionStatus';
import { DrawingCanvas } from './components/DrawingCanvas';
import { PixelPreview } from './components/PixelPreview';
import { MnistSelector } from './components/MnistSelector';
import { PredictionDisplay } from './components/PredictionDisplay';
import { SerialMonitor } from './components/SerialMonitor';
import type { EndpointKind } from './types/protocol';

type InputSource = 'draw' | 'mnist';
const PIXEL_COUNT = 36;
const PIXEL_MIN = 0;
const PIXEL_MAX = 1;
const MNIST_MEAN = 0.1307;
const MNIST_STD = 0.3081;
const TARGET_CONTRAST_MEAN = 0.14;
const TARGET_CONTRAST_STD = 0.28;
const MIN_CONTRAST_STD = 0.05;
const MAX_POST_NORMALIZATION_MEAN = 0.16;
const MIN_POST_NORMALIZATION_SCALE = 0.25;
const TARGET_ACTIVE_PIXELS = 12;
const MAX_ACTIVE_PIXELS_BEFORE_SPARSIFY = 16;
const SPARSIFY_BLEND = 0.9;
const FINAL_FOREGROUND_PIXELS = 10;
const FOREGROUND_MIN_INTENSITY = 0.35;
const DEFAULT_ARDUINO_SERIAL_PORT = '/dev/cu.usbserial-10';
const DEFAULT_SERIAL_BAUD = 9600;
const EMULATOR_WS_PORT = 3001;
const ARDUINO_BRIDGE_WS_PORT = 3012;

function clampPixel(value: number): number {
  return Math.max(PIXEL_MIN, Math.min(PIXEL_MAX, value));
}

function preprocessCustomPixelsForInference(pixels: number[], useContrastNormalization: boolean): number[] {
  const clipped = pixels.slice(0, PIXEL_COUNT).map(clampPixel);
  if (!useContrastNormalization) {
    return clipped;
  }

  const mean = clipped.reduce((sum, value) => sum + value, 0) / PIXEL_COUNT;
  const variance = clipped.reduce((sum, value) => {
    const delta = value - mean;
    return sum + delta * delta;
  }, 0) / PIXEL_COUNT;
  const std = Math.sqrt(variance);
  const safeStd = Math.max(std, MIN_CONTRAST_STD);
  const scale = TARGET_CONTRAST_STD / safeStd;

  const contrastNormalized = clipped.map((value) =>
    clampPixel((value - mean) * scale + TARGET_CONTRAST_MEAN),
  );
  const normalizedMean =
    contrastNormalized.reduce((sum, value) => sum + value, 0) / PIXEL_COUNT;

  // Final guardrail: keep dense drawings from shifting to all-positive normalized values.
  const meanConstrained = normalizedMean <= MAX_POST_NORMALIZATION_MEAN ? contrastNormalized : contrastNormalized.map((value) => {
    const meanScale = Math.max(
      MIN_POST_NORMALIZATION_SCALE,
      MAX_POST_NORMALIZATION_MEAN / Math.max(normalizedMean, Number.EPSILON),
    );
    return clampPixel(value * meanScale);
  });

  const activePixels = meanConstrained.filter((value) => value > MNIST_MEAN).length;
  if (activePixels <= MAX_ACTIVE_PIXELS_BEFORE_SPARSIFY) {
    return meanConstrained;
  }

  const sorted = [...meanConstrained].sort((a, b) => b - a);
  const thresholdIndex = Math.min(TARGET_ACTIVE_PIXELS - 1, sorted.length - 1);
  const threshold = sorted[thresholdIndex];
  const denom = Math.max(PIXEL_MAX - threshold, Number.EPSILON);
  const sparsified = meanConstrained.map((value) => {
    const sparse = clampPixel((value - threshold) / denom);
    // Preserve some grayscale while forcing a sparse active set.
    return clampPixel(sparse * SPARSIFY_BLEND + value * (1 - SPARSIFY_BLEND));
  });

  // Hard foreground extraction: keep only strongest cells so background maps
  // to MNIST-like negative values after normalization.
  const ranked = sparsified
    .map((value, index) => ({ value, index }))
    .sort((a, b) => b.value - a.value);
  const keepCount = Math.min(FINAL_FOREGROUND_PIXELS, ranked.length);
  const keepSet = new Set(ranked.slice(0, keepCount).map((entry) => entry.index));

  return sparsified.map((value, index) => {
    if (!keepSet.has(index)) {
      return 0;
    }
    return clampPixel(Math.max(FOREGROUND_MIN_INTENSITY, value));
  });
}

function App() {
  const host = window.location.hostname || 'localhost';
  const emulatorWsUrl = `ws://${host}:${EMULATOR_WS_PORT}/ws`;
  const arduinoBridgeWsUrl = `ws://${host}:${ARDUINO_BRIDGE_WS_PORT}/ws`;
  const [wsUrl, setWsUrl] = useState(
    emulatorWsUrl,
  );
  const { connected, frame, status, send, serialLog } = useWebSocket(wsUrl);
  const emulatorSocket = useWebSocket(emulatorWsUrl);
  const [backendKind, setBackendKind] = useState<EndpointKind>('local_emulator');
  const [arduinoPort, setArduinoPort] = useState(DEFAULT_ARDUINO_SERIAL_PORT);
  const mnistStatus = emulatorSocket.status ?? status;
  const mnistNavigationDisabled = backendKind === 'serial_arduino' && !emulatorSocket.connected;

  const {
    pixels,
    isHiRes,
    setIsHiRes,
    hiResData,
    isDrawing,
    clear,
    paintHiRes,
    paintLoRes,
    setFromExternal,
    setFromExternal28,
  } = useDrawingCanvas();

  const [inputSource, setInputSource] = useState<InputSource>('draw');
  const isMnistSource = inputSource === 'mnist';
  const [useContrastNormalization, setUseContrastNormalization] = useState(true);

  useEffect(() => {
    if (!connected) {
      return;
    }

    send({
      type: 'SetEndpoint',
      endpoint_kind: backendKind,
      endpoint_address: backendKind === 'serial_arduino' ? arduinoPort : null,
      endpoint_baud: backendKind === 'serial_arduino' ? DEFAULT_SERIAL_BAUD : null,
    });
  }, [arduinoPort, backendKind, connected, send]);

  useEffect(() => {
    setWsUrl(backendKind === 'serial_arduino' ? arduinoBridgeWsUrl : emulatorWsUrl);
  }, [arduinoBridgeWsUrl, backendKind, emulatorWsUrl]);

  useEffect(() => {
    if (status?.endpoint_kind && status.endpoint_kind !== backendKind) {
      setBackendKind(status.endpoint_kind);
    }
    if (
      status?.endpoint_kind === 'serial_arduino' &&
      status.endpoint_address &&
      status.endpoint_address !== arduinoPort
    ) {
      setArduinoPort(status.endpoint_address);
    }
  }, [arduinoPort, backendKind, status?.endpoint_address, status?.endpoint_kind]);

  // When MNIST sample is loaded via server, update pixels from frame data
  const handleLoadSample = useCallback(
    (_index: number) => {
      setInputSource('mnist');
    },
    [],
  );

  // Update canvas when a new MNIST sample is loaded (keyed on sample_index)
  const lastLoadedIndex = useRef<number | null>(null);
  useEffect(() => {
    const idx = mnistStatus?.sample_index ?? null;
    if (
      idx !== null &&
      idx !== lastLoadedIndex.current &&
      mnistStatus?.sample_pixels &&
      mnistStatus.sample_pixels.length === 36
    ) {
      lastLoadedIndex.current = idx;
      // Server pixels are MNIST-normalized: (pixel/255 - 0.1307) / 0.3081
      // Convert back to [0, 1] for canvas display
      const displayPixels = mnistStatus.sample_pixels.map((p) => {
        const raw = p * 0.3081 + 0.1307;
        return Math.max(0, Math.min(1, raw));
      });

      if (mnistStatus.sample_pixels_28x28 && mnistStatus.sample_pixels_28x28.length === 784) {
        // Use real 28x28 MNIST data for hi-res view
        setFromExternal28(displayPixels, mnistStatus.sample_pixels_28x28);
        setIsHiRes(true);
      } else {
        setFromExternal(displayPixels);
      }
    }
  }, [mnistStatus?.sample_index, mnistStatus?.sample_pixels, mnistStatus?.sample_pixels_28x28, setFromExternal, setFromExternal28, setIsHiRes]);

  // Send drawn image to board
  const sendToBoard = useCallback(() => {
    const preprocessedPixels = preprocessCustomPixelsForInference(pixels, useContrastNormalization);
    // Apply MNIST normalization: (pixel - 0.1307) / 0.3081
    // Canvas pixels are in [0, 1] (same as pixel/255), matching the MNIST pipeline
    const normalized = preprocessedPixels.map((v) => (v - MNIST_MEAN) / MNIST_STD);
    send({ type: 'InferCustom', pixels: normalized });
  }, [pixels, send, useContrastNormalization]);

  const markDrawingSource = useCallback(() => {
    setInputSource((prev) => (prev === 'draw' ? prev : 'draw'));
  }, []);

  const handlePaintHiRes = useCallback(
    (x: number, y: number) => {
      markDrawingSource();
      paintHiRes(x, y);
    },
    [markDrawingSource, paintHiRes],
  );

  const handlePaintLoRes = useCallback(
    (gx: number, gy: number, value: number) => {
      markDrawingSource();
      paintLoRes(gx, gy, value);
    },
    [markDrawingSource, paintLoRes],
  );

  // Keyboard shortcuts
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).tagName === 'INPUT') return;
      switch (e.key) {
        case ' ':
          e.preventDefault();
          send(frame?.running ? { type: 'Pause' } : { type: 'Play' });
          break;
        case 'ArrowRight':
          emulatorSocket.send({
            type: 'LoadSample',
            index: (mnistStatus?.sample_index ?? 0) + 1,
          });
          setInputSource('mnist');
          break;
        case 'ArrowLeft':
          emulatorSocket.send({
            type: 'LoadSample',
            index: Math.max(0, (mnistStatus?.sample_index ?? 0) - 1),
          });
          setInputSource('mnist');
          break;
        case 'Enter':
          sendToBoard();
          break;
        case 'c':
          clear();
          setInputSource('draw');
          break;
      }
    };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [frame?.running, mnistStatus?.sample_index, send, sendToBoard, clear, emulatorSocket]);

  return (
    <div className="flex flex-col h-screen" style={{ background: '#020617' }}>
      {/* Header bar */}
      <div
        className="flex items-center justify-between px-4 py-2 shrink-0"
        style={{
          background: '#0f172a',
          borderBottom: '1px solid #1e293b',
        }}
      >
        <div className="flex items-center gap-4">
          <h1
            className="text-sm font-bold tracking-widest"
            style={{ color: '#e2e8f0' }}
          >
            TARSKI
          </h1>
          <span className="text-[10px]" style={{ color: '#475569' }}>
            Neuromorphic Board Interface
          </span>
        </div>
        <ConnectionStatus
          connected={connected}
          wsUrl={wsUrl}
          onUrlChange={setWsUrl}
          backendKind={backendKind}
          backendPort={arduinoPort}
          onBackendKindChange={setBackendKind}
          onBackendPortChange={setArduinoPort}
        />
      </div>

      {/* Main content */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left: Drawing area (primary focus) */}
        <div className="flex-1 flex flex-col items-center justify-center gap-4 p-6 min-w-0">
          {/* Canvas mode toggle */}
          <div className="flex items-center gap-1 p-1 rounded-lg" style={{ background: '#0f172a' }}>
            <button
              onClick={() => { setIsHiRes(true); setInputSource('draw'); }}
              className="px-3 py-1.5 rounded text-[11px] font-medium transition-all"
              style={{
                background: isHiRes && inputSource === 'draw' ? '#1d4ed8' : 'transparent',
                color: isHiRes && inputSource === 'draw' ? '#fff' : '#64748b',
              }}
            >
              HI-RES (28x28)
            </button>
            <button
              onClick={() => { setIsHiRes(false); setInputSource('draw'); }}
              className="px-3 py-1.5 rounded text-[11px] font-medium transition-all"
              style={{
                background: !isHiRes && inputSource === 'draw' ? '#1d4ed8' : 'transparent',
                color: !isHiRes && inputSource === 'draw' ? '#fff' : '#64748b',
              }}
            >
              LO-RES (6x6)
            </button>
            <div style={{ width: 1, height: 16, background: '#334155', margin: '0 4px' }} />
            <button
              onClick={() => setInputSource('mnist')}
              className="px-3 py-1.5 rounded text-[11px] font-medium transition-all"
              style={{
                background: inputSource === 'mnist' ? '#1d4ed8' : 'transparent',
                color: inputSource === 'mnist' ? '#fff' : '#64748b',
              }}
            >
              MNIST
            </button>
          </div>

          {/* Canvas */}
          <DrawingCanvas
            pixels={pixels}
            isHiRes={isHiRes}
            hiResData={hiResData}
            isDrawing={isDrawing}
            onPaintHiRes={handlePaintHiRes}
            onPaintLoRes={handlePaintLoRes}
            onClear={() => { clear(); setInputSource('draw'); }}
          />

          {/* 6x6 preview + Send button */}
          <div className="flex items-center gap-6">
            <div className="flex flex-col items-center gap-1">
              <span className="text-[10px] font-bold tracking-wider" style={{ color: '#64748b' }}>
                6x6 INPUT
              </span>
              <PixelPreview pixels={pixels} size={72} highlighted />
            </div>

            <button
              onClick={sendToBoard}
              disabled={!connected}
              className="px-6 py-3 rounded-lg text-sm font-bold tracking-wider transition-all"
              style={{
                background: connected ? '#1d4ed8' : '#1e293b',
                color: connected ? '#fff' : '#475569',
                border: `1px solid ${connected ? '#2563eb' : '#334155'}`,
                opacity: connected ? 1 : 0.5,
                cursor: connected ? 'pointer' : 'not-allowed',
              }}
            >
              SEND TO BOARD
            </button>

            <div className="flex flex-col items-center gap-1 text-[10px]" style={{ color: '#475569' }}>
              <div>Enter = Send</div>
              <div>Space = Play/Pause</div>
              <div>C = Clear</div>
            </div>
          </div>
          <label
            className="flex items-center gap-2 text-[11px] px-3 py-1.5 rounded-md"
            style={{ background: '#0f172a', border: '1px solid #334155', color: '#94a3b8' }}
          >
            <input
              type="checkbox"
              checked={useContrastNormalization}
              onChange={(e) => setUseContrastNormalization(e.target.checked)}
            />
            Contrast normalize custom input
          </label>
        </div>

        {/* Right panel */}
        <div
          className="flex flex-col gap-3 p-3 overflow-y-auto"
          style={{
            width: 'clamp(340px, 30vw, 440px)',
            minWidth: 340,
            borderLeft: '1px solid #1e293b',
            background: '#0f172a',
          }}
        >
          {/* Prediction */}
          <PredictionDisplay
            prediction={status?.prediction ?? null}
            trueLabel={isMnistSource ? (status?.true_label ?? null) : null}
            correct={isMnistSource ? (status?.correct ?? null) : null}
            spikeCounts={frame?.output_spike_counts ?? []}
            timestep={frame?.timestep ?? 0}
            timeMs={frame?.time_ms ?? 0}
          />

          {/* MNIST selector */}
          <MnistSelector
            send={emulatorSocket.send}
            currentIndex={mnistStatus?.sample_index ?? null}
            totalSamples={mnistStatus?.total_samples ?? 0}
            onLoadSample={handleLoadSample}
            disabled={mnistNavigationDisabled}
          />
          {mnistNavigationDisabled ? (
            <div
              className="p-2.5 rounded-lg text-[10px]"
              style={{ background: '#3f2b12', border: '1px solid #78350f', color: '#fcd34d' }}
            >
              Start emulator backend on localhost:3001 to use MNIST Prev/Next while in Arduino mode.
            </div>
          ) : null}

          {/* Simulation controls */}
          <div
            className="flex flex-col gap-2 p-3 rounded-lg"
            style={{ background: '#1e293b', border: '1px solid #334155' }}
          >
            <span className="text-[11px] font-bold tracking-wider" style={{ color: '#94a3b8' }}>
              SIMULATION
            </span>
            <div className="flex items-center gap-2">
              <button
                onClick={() => send({ type: 'Play' })}
                className="flex-1 py-1.5 rounded text-[11px] font-bold transition-opacity hover:opacity-80"
                style={{ background: '#059669', color: '#fff' }}
              >
                PLAY
              </button>
              <button
                onClick={() => send({ type: 'Pause' })}
                className="flex-1 py-1.5 rounded text-[11px] font-bold transition-opacity hover:opacity-80"
                style={{ background: '#b91c1c', color: '#fff' }}
              >
                PAUSE
              </button>
              <button
                onClick={() => send({ type: 'Step' })}
                className="flex-1 py-1.5 rounded text-[11px] font-bold transition-opacity hover:opacity-80"
                style={{ background: '#1e293b', border: '1px solid #334155', color: '#94a3b8' }}
              >
                STEP
              </button>
            </div>
            <div className="flex items-center gap-2 text-[10px]" style={{ color: '#475569' }}>
              <span>Step: {frame?.timestep ?? 0}</span>
              <span>|</span>
              <span>{(frame?.time_ms ?? 0).toFixed(1)}ms</span>
              <span>|</span>
              <span>{frame?.running ? 'Running' : 'Paused'}</span>
            </div>
          </div>

          {/* Serial Monitor */}
          <SerialMonitor entries={serialLog} />
          {status?.endpoint_error ? (
            <div
              className="p-2.5 rounded-lg text-[10px]"
              style={{ background: '#3f1d1d', border: '1px solid #7f1d1d', color: '#fecaca' }}
            >
              Backend error: {status.endpoint_error}
            </div>
          ) : null}

          {/* Keyboard help */}
          <div
            className="p-2.5 rounded-lg text-[10px]"
            style={{ background: '#1e293b', border: '1px solid #334155', color: '#475569' }}
          >
            <span className="font-bold" style={{ color: '#64748b' }}>
              Keys:{' '}
            </span>
            Space=play/pause | Arrows=samples | Enter=send | C=clear
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
