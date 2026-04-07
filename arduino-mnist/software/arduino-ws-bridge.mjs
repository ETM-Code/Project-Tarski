import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { spawnSync } from 'node:child_process';
import { WebSocketServer } from 'ws';

const HOST = process.env.ARDUINO_BRIDGE_HOST ?? '0.0.0.0';
const WS_PORT = Number(process.env.ARDUINO_BRIDGE_PORT ?? '3012');
const DEFAULT_SERIAL_PORT = process.env.ARDUINO_SERIAL_PORT ?? '/dev/cu.usbserial-10';
const DEFAULT_BAUD = Number(process.env.ARDUINO_SERIAL_BAUD ?? '9600');
const INTERFACE_BIN = process.env.ARDUINO_INTERFACE_BIN ?? './arduino-interface';
const PIXEL_COUNT = 36;
const DUMMY_LABEL_BYTE = 0;
const PORT_TRN_END = 0x04;
const PREDICT_COMMAND = '-p';

/** @typedef {{ endpoint_address?: string | null; endpoint_baud?: number | null; prediction?: number | null; endpoint_error?: string | null; }} ClientState */

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function normalizedToInt8Byte(value) {
  const raw01 = clamp(value * 0.3081 + 0.1307, 0, 1);
  const quantized = clamp(Math.round(raw01 * 255 - 128), -128, 127);
  return (quantized + 256) % 256;
}

function pixelsToSampleBytes(pixels) {
  const out = new Uint8Array(PIXEL_COUNT + 1);
  for (let i = 0; i < PIXEL_COUNT; i += 1) {
    out[i] = normalizedToInt8Byte(Number(pixels?.[i] ?? 0));
  }
  out[PIXEL_COUNT] = DUMMY_LABEL_BYTE;
  return out;
}

function runInterface(args) {
  const proc = spawnSync(INTERFACE_BIN, args, { encoding: 'utf8' });
  const stdout = proc.stdout ?? '';
  const stderr = proc.stderr ?? '';
  const output = `${stdout}\n${stderr}`.trim();
  if (proc.status !== 0) {
    throw new Error(output || `arduino-interface exited with code ${proc.status ?? 'unknown'}`);
  }
  return output;
}

function parsePrediction(outputText) {
  const lines = outputText
    .split('\n')
    .map((line) => line.trim())
    .filter((line) => line.length > 0);

  for (const line of lines) {
    const parts = line.split(',');
    if (parts.length < 2) {
      continue;
    }
    const maybePrediction = Number(parts[1]);
    if (Number.isInteger(maybePrediction) && maybePrediction >= 0 && maybePrediction <= 9) {
      return maybePrediction;
    }
  }
  throw new Error('Could not parse prediction from arduino-interface output');
}

function buildFrame(rxBytes, txBytes) {
  return {
    type: 'SimFrame',
    time_ms: Date.now(),
    timestep: 0,
    speed: 1,
    running: false,
    hidden_membranes: [],
    output_membranes: [],
    hidden_spikes: [],
    output_spikes: [],
    output_spike_counts: [],
    net_voltages: {},
    component_states: {},
    uart_tx: txBytes,
    uart_rx: rxBytes,
    power_by_rail: {},
    energy_joules: 0,
  };
}

function buildStatus(state) {
  return {
    type: 'Status',
    running: false,
    time_ms: 0,
    timestep: 0,
    speed: 1,
    max_steps: 0,
    checkpoint_loaded: false,
    sample_index: null,
    total_samples: 0,
    true_label: null,
    prediction: state.prediction ?? null,
    correct: null,
    sample_pixels: undefined,
    sample_pixels_28x28: undefined,
    endpoint_kind: 'serial_arduino',
    endpoint_address: state.endpoint_address ?? DEFAULT_SERIAL_PORT,
    endpoint_baud: state.endpoint_baud ?? DEFAULT_BAUD,
    endpoint_error: state.endpoint_error ?? null,
  };
}

function runInfer(pixels, serialPort, baud) {
  const tempDir = mkdtempSync(join(tmpdir(), 'tarski-arduino-'));
  const samplePath = join(tempDir, 'sample.bin');
  try {
    const sampleBytes = pixelsToSampleBytes(pixels);
    writeFileSync(samplePath, sampleBytes);
    // Use predict mode so output stays in CSV form:
    // <firmware_message>,<expected_label>,<true|false>
    const inferOutput = runInterface([serialPort, PREDICT_COMMAND, '-i', samplePath, '-o', '/dev/stdout']);
    const prediction = parsePrediction(inferOutput);
    const txBytes = Array.from(new TextEncoder().encode(inferOutput));
    const rxBytes = [...Array.from(sampleBytes), PORT_TRN_END];
    return { prediction, rxBytes, txBytes, baud };
  } finally {
    rmSync(tempDir, { recursive: true, force: true });
  }
}

const wss = new WebSocketServer({ host: HOST, port: WS_PORT, path: '/ws' });

console.log(`Arduino WS bridge listening on ws://${HOST}:${WS_PORT}/ws`);
console.log(`Using arduino-interface binary: ${INTERFACE_BIN}`);
console.log(`Default serial port: ${DEFAULT_SERIAL_PORT} @ ${DEFAULT_BAUD}`);

wss.on('connection', (socket) => {
  /** @type {ClientState} */
  const state = {
    endpoint_address: DEFAULT_SERIAL_PORT,
    endpoint_baud: DEFAULT_BAUD,
    prediction: null,
    endpoint_error: null,
  };

  socket.send(JSON.stringify({ type: 'BoardInfo', num_nets: 0, num_components: 0, arduino_pins: {}, neurons: [], shift_registers: [], pcb_svg: '', firmware_version: 'arduino-bridge-0.1.0' }));
  socket.send(JSON.stringify(buildStatus(state)));

  socket.on('message', (raw) => {
    try {
      const msg = JSON.parse(String(raw));

      if (msg.type === 'SetEndpoint') {
        state.endpoint_address = msg.endpoint_address || DEFAULT_SERIAL_PORT;
        state.endpoint_baud = Number(msg.endpoint_baud ?? DEFAULT_BAUD);
        state.endpoint_error = null;
        socket.send(JSON.stringify(buildStatus(state)));
        return;
      }

      if (msg.type === 'GetBoardInfo') {
        socket.send(JSON.stringify({ type: 'BoardInfo', num_nets: 0, num_components: 0, arduino_pins: {}, neurons: [], shift_registers: [], pcb_svg: '', firmware_version: 'arduino-bridge-0.1.0' }));
        return;
      }

      if (msg.type !== 'InferCustom') {
        return;
      }

      const serialPort = state.endpoint_address ?? DEFAULT_SERIAL_PORT;
      const baud = state.endpoint_baud ?? DEFAULT_BAUD;
      const { prediction, rxBytes, txBytes } = runInfer(msg.pixels, serialPort, baud);
      state.prediction = prediction;
      state.endpoint_error = null;
      socket.send(JSON.stringify(buildFrame(rxBytes, txBytes)));
      socket.send(JSON.stringify(buildStatus(state)));
    } catch (error) {
      state.endpoint_error = error instanceof Error ? error.message : String(error);
      socket.send(JSON.stringify(buildStatus(state)));
    }
  });
});
