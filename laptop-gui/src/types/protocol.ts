// TypeScript types matching the Rust protocol.rs definitions

// ============================================================
// Server -> Client
// ============================================================

export type ServerMessage =
  | ({ type: 'BoardInfo' } & BoardInfo)
  | ({ type: 'SimFrame' } & SimFrame)
  | ({ type: 'ProbeData' } & ProbeData)
  | ({ type: 'Status' } & SimStatus);

export interface BoardInfo {
  num_nets: number;
  num_components: number;
  arduino_pins: Record<string, string>;
  neurons: NeuronInfo[];
  shift_registers: string[];
  pcb_svg: string;
  firmware_version: string;
}

export interface NeuronInfo {
  id: string;
  layer: number;
  membrane_net: string;
}

export interface SimFrame {
  time_ms: number;
  timestep: number;
  speed: number;
  running: boolean;

  hidden_membranes: number[];
  output_membranes: number[];
  hidden_spikes: number[];
  output_spikes: number[];
  output_spike_counts: number[];

  net_voltages: Record<string, number>;
  component_states: Record<string, ComponentVizState>;

  uart_tx: number[];
  uart_rx: number[];

  power_by_rail: Record<string, number>;
  energy_joules: number;
}

export type ComponentVizState =
  | { kind: 'ShiftRegister'; bits: boolean[]; value: number }
  | { kind: 'Dac'; voltage: number; channel: number }
  | { kind: 'Comparator'; output_high: boolean }
  | { kind: 'Latch'; q: boolean }
  | { kind: 'Switch'; closed: boolean }
  | { kind: 'Neuron'; membrane_mv: number; spiking: boolean };

export interface SimStatus {
  running: boolean;
  time_ms: number;
  timestep: number;
  speed: number;
  max_steps: number;
  checkpoint_loaded: boolean;
  sample_index: number | null;
  total_samples: number;
  true_label: number | null;
  prediction: number | null;
  correct: boolean | null;
  sample_pixels?: number[];
  sample_pixels_28x28?: number[];
}

export interface ProbeData {
  net_name: string;
  times_ms: number[];
  voltages: number[];
}

// ============================================================
// Client -> Server
// ============================================================

export type ClientMessage =
  | { type: 'Play' }
  | { type: 'Pause' }
  | { type: 'Step' }
  | { type: 'SetSpeed'; speed: number }
  | { type: 'SetMaxSteps'; steps: number }
  | { type: 'LoadCheckpoint'; path: string }
  | { type: 'LoadSample'; index: number }
  | { type: 'SerialCommand'; command: number; payload: number[] }
  | { type: 'AddProbe'; net_name: string }
  | { type: 'RemoveProbe'; net_name: string }
  | { type: 'GetBoardInfo' }
  | { type: 'InferCustom'; pixels: number[] };
