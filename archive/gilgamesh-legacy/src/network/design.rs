use serde::{Deserialize, Deserializer, Serialize};
use std::collections::{BTreeMap, HashMap};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Design {
    pub version: Option<String>,
    #[serde(default)]
    pub units: Option<Units>,
    #[serde(default)]
    pub globals: Globals,
    #[serde(
        default,
        rename = "neuron_templates",
        deserialize_with = "deserialize_templates",
        alias = "neuron_template"
    )]
    pub neuron_templates: HashMap<String, NeuronTemplate>,
    #[serde(default)]
    pub layers: Vec<LayerSpec>,
    #[serde(default)]
    pub connect: Vec<ConnectSpec>,
    #[serde(default)]
    pub overrides: Vec<OverrideSpec>,
    #[serde(default)]
    pub external_inputs: Vec<ExternalInputSpec>,
    #[serde(default)]
    pub readouts: Vec<ReadoutSpec>,
    #[serde(default)]
    pub checkpoint: Option<CheckpointSpec>,
    #[serde(default)]
    pub training: Option<TrainingDefaults>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Units {
    #[serde(default)]
    pub voltage: Option<String>,
    #[serde(default)]
    pub time: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Globals {
    #[serde(flatten)]
    pub values: BTreeMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuronTemplate {
    #[serde(alias = "Cm")]
    pub cm: f64,
    #[serde(alias = "Rleak", alias = "R_leak")]
    pub r_leak: f64,
    #[serde(default)]
    pub theta: ThetaTemplate,
    #[serde(default)]
    pub reset: ResetTemplate,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "lowercase")]
pub enum ThetaTemplate {
    Constant {
        #[serde(alias = "theta0")]
        theta0: f64,
    },
    Adaptive {
        #[serde(alias = "theta0")]
        theta0: f64,
        #[serde(alias = "Rvh", default)]
        rvh: f64,
        #[serde(alias = "Rvl", default)]
        rvl: f64,
        #[serde(alias = "Rf", default)]
        rf: f64,
        #[serde(alias = "Ctheta", default)]
        ctheta: f64,
    },
}

impl Default for ThetaTemplate {
    fn default() -> Self {
        ThetaTemplate::Constant { theta0: 0.0 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResetTemplate {
    #[serde(alias = "Rreset", alias = "R_reset")]
    pub r_reset: f64,
    #[serde(alias = "Ron_switch", alias = "R_on_switch")]
    pub ron_switch: f64,
}

impl Default for ResetTemplate {
    fn default() -> Self {
        Self {
            r_reset: 0.0,
            ron_switch: 0.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerSpec {
    pub id: String,
    #[serde(default)]
    pub r#type: LayerType,
    pub size: usize,
    #[serde(default = "default_layer_template")]
    pub neuron: String,
}

fn default_layer_template() -> String {
    "default".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LayerType {
    Input,
    Hidden,
    Readout,
}

impl Default for LayerType {
    fn default() -> Self {
        LayerType::Hidden
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynapseSpec {
    #[serde(default = "default_synapse_type")]
    pub r#type: SynapseType,
    #[serde(alias = "R", alias = "R_ohm")]
    pub resistance: f64,
}

fn default_synapse_type() -> SynapseType {
    SynapseType::Excitatory
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum SynapseType {
    Excitatory,
    Inhibitory,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectSpec {
    pub from: String,
    pub to: String,
    #[serde(default = "ConnectRule::dense")]
    pub rule: ConnectRule,
    pub synapse: SynapseSpec,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "rule", rename_all = "snake_case")]
pub enum ConnectRule {
    Dense,
    /// Dense connectivity but excludes self-connections (for lateral inhibition).
    DenseNoSelf,
    FixedFanIn {
        #[serde(alias = "k", alias = "fan_in")]
        fan_in: usize,
    },
    Prob {
        #[serde(alias = "p", alias = "probability")]
        probability: f64,
        #[serde(default)]
        seed: Option<u64>,
    },
    Ring {
        #[serde(default = "default_ring_shift")]
        shift: i32,
    },
}

fn default_ring_shift() -> i32 {
    1
}

impl ConnectRule {
    pub fn dense() -> Self {
        ConnectRule::Dense
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverrideSpec {
    pub from: String,
    pub to: String,
    pub synapse: SynapseSpec,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExternalInputSpec {
    pub id: String,
    pub target: String,
    pub wave: String,
    #[serde(default)]
    pub params: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadoutSpec {
    pub id: String,
    #[serde(rename = "type")]
    pub r#type: ReadoutType,
    #[serde(default)]
    pub source_layer: Option<String>,
    #[serde(default = "default_readout_signal")]
    pub signal: String,
    #[serde(default)]
    pub post: Option<String>,
    #[serde(default)]
    pub tau_s: Option<f64>,
}

fn default_readout_signal() -> String {
    "analog_background".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReadoutType {
    Analog,
    Spike,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointSpec {
    pub every: f64,
    #[serde(default)]
    pub fields: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TrainingDefaults {
    #[serde(default)]
    pub dataset: Option<String>,
    #[serde(default)]
    pub test_dataset: Option<String>,
    #[serde(default)]
    pub input_layer: Option<String>,
    #[serde(default)]
    pub target_readout: Option<String>,
    #[serde(default)]
    pub config: Option<TrainingConfigDefaults>,
    #[serde(default)]
    pub checkpoint_dir: Option<String>,
    #[serde(default)]
    pub checkpoint_every: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TrainingConfigDefaults {
    #[serde(default)]
    pub learning_rate: Option<f64>,
    #[serde(default)]
    pub epochs: Option<usize>,
    #[serde(default)]
    pub regularization: Option<f64>,
    #[serde(default)]
    pub row_spacing_start: Option<f64>,
    #[serde(default)]
    pub row_spacing_end: Option<f64>,
    #[serde(default)]
    pub pulse_width: Option<f64>,
    #[serde(default)]
    pub input_scale: Option<f64>,
    #[serde(default)]
    pub sample_limit: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct IndexRef {
    pub layer: String,
    pub selector: IndexSelector,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum IndexSelector {
    All,
    Single(usize),
    Range { start: usize, end: usize },
}

impl IndexRef {
    pub fn parse(value: &str) -> anyhow::Result<Self> {
        let trimmed = value.trim();
        if let Some(idx) = trimmed.find('[') {
            let layer = trimmed[..idx].trim().to_string();
            let inside = trimmed[idx + 1..]
                .strip_suffix(']')
                .ok_or_else(|| anyhow::anyhow!("missing closing bracket in index"))?;
            if inside == "*" {
                Ok(IndexRef {
                    layer,
                    selector: IndexSelector::All,
                })
            } else if let Some(colon) = inside.find(':') {
                let start = inside[..colon].trim().parse()?;
                let end = inside[colon + 1..].trim().parse()?;
                if end < start {
                    return Err(anyhow::anyhow!("range end before start"));
                }
                Ok(IndexRef {
                    layer,
                    selector: IndexSelector::Range { start, end },
                })
            } else {
                let index = inside.trim().parse()?;
                Ok(IndexRef {
                    layer,
                    selector: IndexSelector::Single(index),
                })
            }
        } else {
            Ok(IndexRef {
                layer: trimmed.to_string(),
                selector: IndexSelector::All,
            })
        }
    }
}

fn deserialize_templates<'de, D>(
    deserializer: D,
) -> Result<HashMap<String, NeuronTemplate>, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum TemplateInput {
        Map(HashMap<String, NeuronTemplate>),
        Single(NeuronTemplate),
    }

    let parsed = TemplateInput::deserialize(deserializer)?;
    Ok(match parsed {
        TemplateInput::Map(map) => map,
        TemplateInput::Single(template) => {
            let mut map = HashMap::new();
            map.insert("default".to_string(), template);
            map
        }
    })
}

impl Globals {
    pub fn get(&self, key: &str) -> Option<f64> {
        self.values.get(key).copied()
    }
}
