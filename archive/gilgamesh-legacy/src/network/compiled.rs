use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompiledNetwork {
    pub version: String,
    pub timebase: String,
    pub neuron_count: usize,
    pub synapse_count: usize,
    pub globals: BTreeMap<String, f64>,
    pub layers: Vec<CompiledLayer>,
    pub neuron: NeuronParameters,
    pub incoming: IncomingConnectivity,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outgoing: Option<OutgoingConnectivity>,
    #[serde(default)]
    pub readouts: Vec<CompiledReadout>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stimuli: Vec<StimulusChannel>,
    pub state0: InitialState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub training_checkpoint: Option<TrainingCheckpoint>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompiledLayer {
    pub id: String,
    pub offset: usize,
    pub size: usize,
    pub layer_type: LayerRuntimeType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuronParameters {
    pub cm: Vec<f64>,
    pub r_leak: Vec<f64>,
    pub tau_m: Vec<f64>,
    pub theta_mode: Vec<u8>,
    pub theta0: Vec<f64>,
    /// Duration of output pulse stretch in seconds (0.0 = no stretching, use single timestep)
    #[serde(default)]
    pub pulse_stretch_duration: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncomingConnectivity {
    pub row_ptr: Vec<usize>,
    pub src: Vec<usize>,
    pub g: Vec<f64>,
    pub synapse_type: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutgoingConnectivity {
    pub col_ptr: Vec<usize>,
    pub dst: Vec<usize>,
    pub synapse_index: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompiledReadout {
    pub id: String,
    #[serde(rename = "type")]
    pub r#type: ReadoutRuntimeType,
    pub indices: Vec<usize>,
    pub signal: String,
    #[serde(default)]
    pub post: Option<String>,
    #[serde(default)]
    pub tau_s: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LayerRuntimeType {
    Input,
    Hidden,
    Readout,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReadoutRuntimeType {
    Analog,
    Spike,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitialState {
    pub t: f64,
    pub u: Vec<f64>,
    pub theta: Vec<f64>,
    #[serde(default)]
    pub comp: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingCheckpoint {
    pub epoch: usize,
    pub learning_rate: f64,
    pub row_spacing_start: f64,
    pub row_spacing_end: f64,
    pub pulse_width: f64,
    pub input_scale: f64,
    pub regularization: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StimulusChannel {
    pub id: String,
    pub target_indices: Vec<usize>,
    pub times: Vec<f64>,
    pub values: Vec<f64>,
}

impl StimulusChannel {
    pub fn is_valid(&self) -> bool {
        self.times.len() == self.values.len() && !self.times.is_empty()
    }
}

impl CompiledNetwork {
    pub fn neuron_count(&self) -> usize {
        self.neuron_count
    }

    pub fn synapse_count(&self) -> usize {
        self.synapse_count
    }
}
