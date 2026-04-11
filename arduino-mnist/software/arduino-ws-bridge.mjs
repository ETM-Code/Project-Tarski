import { existsSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { spawnSync } from 'node:child_process';
import { WebSocketServer } from 'ws';

const HOST = process.env.ARDUINO_BRIDGE_HOST ?? '0.0.0.0';
const WS_PORT = Number(process.env.ARDUINO_BRIDGE_PORT ?? '3012');
const DEFAULT_SERIAL_PORT = process.env.ARDUINO_SERIAL_PORT ?? '/dev/cu.usbserial-10';
const DEFAULT_BAUD = Number(process.env.ARDUINO_SERIAL_BAUD ?? '9600');
const INTERFACE_BIN = process.env.ARDUINO_INTERFACE_BIN ?? './arduino-interface';
const SAMPLE_DIR = process.env.ARDUINO_SAMPLE_DIR ?? './data';
const MNIST_RAW_DIR = process.env.ARDUINO_MNIST_RAW_DIR ?? '../data/MNIST/raw';
const PIXEL_COUNT = 36;
const DUMMY_LABEL_BYTE = 0;
const MNIST_IMAGE_MAGIC = 2051;
const MNIST_LABEL_MAGIC = 2049;
const MNIST_MEAN = 0.1307;
const MNIST_STD = 0.3081;
const SOURCE_IMAGE_SIZE = 28;
const TARGET_IMAGE_SIZE = 6;
const PORT_TRN_END = 0x04;
const PREDICT_COMMAND = '-p';

/** @typedef {{ endpoint_address?: string | null; endpoint_baud?: number | null; prediction?: number | null; endpoint_error?: string | null; sample_index?: number | null; total_samples?: number; true_label?: number | null; sample_pixels?: number[] | undefined; sample_files?: string[]; mnistRaw?: MnistRawData | null; }} ClientState */
/** @typedef {{ imageCount: number; rows: number; cols: number; images: Uint8Array; labels: Uint8Array; }} MnistRawData */

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

function readU32BE(buffer, offset) {
  return (
    (buffer[offset] << 24) |
    (buffer[offset + 1] << 16) |
    (buffer[offset + 2] << 8) |
    buffer[offset + 3]
  ) >>> 0;
}

function normalizeU8ToMnist(value) {
  return (value / 255 - MNIST_MEAN) / MNIST_STD;
}

function downscale28To6Normalized(src28x28) {
  const out = new Array(PIXEL_COUNT).fill(0);
  for (let gy = 0; gy < TARGET_IMAGE_SIZE; gy += 1) {
    for (let gx = 0; gx < TARGET_IMAGE_SIZE; gx += 1) {
      const yStart = Math.floor((gy * SOURCE_IMAGE_SIZE) / TARGET_IMAGE_SIZE);
      const yEnd = Math.floor(((gy + 1) * SOURCE_IMAGE_SIZE) / TARGET_IMAGE_SIZE);
      const xStart = Math.floor((gx * SOURCE_IMAGE_SIZE) / TARGET_IMAGE_SIZE);
      const xEnd = Math.floor(((gx + 1) * SOURCE_IMAGE_SIZE) / TARGET_IMAGE_SIZE);
      let sum = 0;
      let count = 0;
      for (let y = yStart; y < yEnd; y += 1) {
        for (let x = xStart; x < xEnd; x += 1) {
          sum += src28x28[y * SOURCE_IMAGE_SIZE + x];
          count += 1;
        }
      }
      const avg = count > 0 ? sum / count : 0;
      out[gy * TARGET_IMAGE_SIZE + gx] = normalizeU8ToMnist(avg);
    }
  }
  return out;
}

function loadMnistRawData() {
  const imagesPath = join(MNIST_RAW_DIR, 't10k-images-idx3-ubyte');
  const labelsPath = join(MNIST_RAW_DIR, 't10k-labels-idx1-ubyte');
  if (!existsSync(imagesPath) || !existsSync(labelsPath)) {
    return null;
  }

  const imagesFile = readFileSync(imagesPath);
  const labelsFile = readFileSync(labelsPath);
  if (imagesFile.length < 16 || labelsFile.length < 8) {
    throw new Error('MNIST raw IDX files are too small.');
  }

  const imageMagic = readU32BE(imagesFile, 0);
  const imageCount = readU32BE(imagesFile, 4);
  const rows = readU32BE(imagesFile, 8);
  const cols = readU32BE(imagesFile, 12);
  const labelMagic = readU32BE(labelsFile, 0);
  const labelCount = readU32BE(labelsFile, 4);

  if (imageMagic !== MNIST_IMAGE_MAGIC || labelMagic !== MNIST_LABEL_MAGIC) {
    throw new Error('MNIST raw IDX magic values are invalid.');
  }
  if (rows !== SOURCE_IMAGE_SIZE || cols !== SOURCE_IMAGE_SIZE) {
    throw new Error(`Expected ${SOURCE_IMAGE_SIZE}x${SOURCE_IMAGE_SIZE} MNIST images, got ${rows}x${cols}.`);
  }
  if (imageCount !== labelCount) {
    throw new Error('MNIST image/label counts do not match.');
  }

  const images = imagesFile.subarray(16);
  const labels = labelsFile.subarray(8);
  const expectedPixels = imageCount * rows * cols;
  if (images.length < expectedPixels || labels.length < imageCount) {
    throw new Error('MNIST raw IDX payload size mismatch.');
  }

  return { imageCount, rows, cols, images, labels };
}

function i8FromByte(byteValue) {
  return byteValue > 127 ? byteValue - 256 : byteValue;
}

function normalizeByteToMnist(byteValue) {
  const int8 = i8FromByte(byteValue);
  const raw01 = (int8 + 128) / 255;
  return (raw01 - 0.1307) / 0.3081;
}

function listSampleFiles() {
  if (!existsSync(SAMPLE_DIR)) {
    return [];
  }
  return readdirSync(SAMPLE_DIR)
    .filter((file) => file.toLowerCase().endsWith('.bin'))
    .sort((a, b) => a.localeCompare(b))
    .map((file) => join(SAMPLE_DIR, file));
}

function loadSampleByIndex(state, index) {
  const files = state.sample_files ?? [];
  if (files.length > 0) {
    const wrappedIndex = ((index % files.length) + files.length) % files.length;
    const data = readFileSync(files[wrappedIndex]);
    if (data.length < PIXEL_COUNT) {
      throw new Error(`Sample '${files[wrappedIndex]}' is too small (${data.length} bytes).`);
    }

    const samplePixels = [];
    for (let i = 0; i < PIXEL_COUNT; i += 1) {
      samplePixels.push(normalizeByteToMnist(data[i]));
    }

    state.sample_index = wrappedIndex;
    state.total_samples = files.length;
    state.true_label = data.length >= PIXEL_COUNT + 1 ? Number(data[PIXEL_COUNT]) : null;
    state.sample_pixels = samplePixels;
    console.log(`[bridge] Loaded .bin sample index=${wrappedIndex} total=${files.length}`);
    return;
  }

  const mnistRaw = state.mnistRaw;
  if (!mnistRaw) {
    state.sample_index = null;
    state.total_samples = 0;
    state.true_label = null;
    state.sample_pixels = undefined;
    throw new Error(
      `No .bin samples found in '${SAMPLE_DIR}', and no MNIST raw set found in '${MNIST_RAW_DIR}'.`,
    );
  }

  const wrappedIndex = ((index % mnistRaw.imageCount) + mnistRaw.imageCount) % mnistRaw.imageCount;
  const imageOffset = wrappedIndex * mnistRaw.rows * mnistRaw.cols;
  const src28 = mnistRaw.images.subarray(imageOffset, imageOffset + mnistRaw.rows * mnistRaw.cols);
  const samplePixels = downscale28To6Normalized(src28);

  state.sample_index = wrappedIndex;
  state.total_samples = mnistRaw.imageCount;
  state.true_label = Number(mnistRaw.labels[wrappedIndex] ?? 0);
  state.sample_pixels = samplePixels;
  console.log(`[bridge] Loaded MNIST raw sample index=${wrappedIndex} total=${mnistRaw.imageCount}`);
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
    sample_index: state.sample_index ?? null,
    total_samples: state.total_samples ?? 0,
    true_label: state.true_label ?? null,
    prediction: state.prediction ?? null,
    correct: null,
    sample_pixels: state.sample_pixels,
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
    const inferOutput = runInterface([serialPort, PREDICT_COMMAND, '-i', samplePath]);
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
  console.log('[bridge] Client connected');
  /** @type {ClientState} */
  const state = {
    endpoint_address: DEFAULT_SERIAL_PORT,
    endpoint_baud: DEFAULT_BAUD,
    prediction: null,
    endpoint_error: null,
    sample_index: null,
    total_samples: 0,
    true_label: null,
    sample_pixels: undefined,
    sample_files: listSampleFiles(),
    mnistRaw: null,
  };

  if (state.sample_files.length === 0) {
    try {
      state.mnistRaw = loadMnistRawData();
    } catch (error) {
      state.endpoint_error = error instanceof Error ? error.message : String(error);
    }
  }

  if (state.sample_files.length > 0) {
    try {
      loadSampleByIndex(state, 0);
    } catch (error) {
      state.endpoint_error = error instanceof Error ? error.message : String(error);
    }
  }

  socket.send(JSON.stringify({ type: 'BoardInfo', num_nets: 0, num_components: 0, arduino_pins: {}, neurons: [], shift_registers: [], pcb_svg: '', firmware_version: 'arduino-bridge-0.1.0' }));
  socket.send(JSON.stringify(buildStatus(state)));

  socket.on('message', (raw) => {
    try {
      const msg = JSON.parse(String(raw));
      console.log(`[bridge] RX message type=${msg?.type ?? 'unknown'}`);

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

      if (msg.type === 'LoadSample') {
        loadSampleByIndex(state, Number(msg.index ?? 0));
        state.endpoint_error = null;
        socket.send(JSON.stringify(buildStatus(state)));
        console.log(`[bridge] TX status after LoadSample index=${state.sample_index}`);
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
      console.log(`[bridge] Inference complete prediction=${prediction}`);
    } catch (error) {
      state.endpoint_error = error instanceof Error ? error.message : String(error);
      console.error(`[bridge] ERROR ${state.endpoint_error}`);
      socket.send(JSON.stringify(buildStatus(state)));
    }
  });
  socket.on('close', () => {
    console.log('[bridge] Client disconnected');
  });
});
