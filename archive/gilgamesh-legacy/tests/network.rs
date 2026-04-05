use std::collections::{BTreeMap, HashMap};

use gilgamesh::network::compiler::compile_design;
use gilgamesh::network::design::{
    CheckpointSpec, ConnectRule, ConnectSpec, Design, ExternalInputSpec, Globals, LayerSpec,
    LayerType, NeuronTemplate, ReadoutSpec, ReadoutType, ResetTemplate, SynapseSpec, SynapseType,
    ThetaTemplate, Units,
};
use gilgamesh::network::runtime::{simulate_network, SimulationOptions};
use gilgamesh::network::training::{train_network, TrainerConfig, TrainingExample};
use serde_json::json;

fn simple_design() -> Design {
    let mut neuron_templates = HashMap::new();
    neuron_templates.insert(
        "default".to_string(),
        NeuronTemplate {
            cm: 1e-9,
            r_leak: 1e6,
            theta: ThetaTemplate::Constant { theta0: 1.0 },
            reset: ResetTemplate {
                r_reset: 1e9,
                ron_switch: 1e9,
            },
        },
    );

    let mut globals = Globals::default();
    globals.values.insert("alpha_out".to_string(), 1.0);

    let layers = vec![
        LayerSpec {
            id: "L0".to_string(),
            r#type: LayerType::Input,
            size: 2,
            neuron: "default".to_string(),
        },
        LayerSpec {
            id: "L1".to_string(),
            r#type: LayerType::Readout,
            size: 1,
            neuron: "default".to_string(),
        },
    ];

    let connect = vec![ConnectSpec {
        from: "L0".to_string(),
        to: "L1".to_string(),
        rule: ConnectRule::Dense,
        synapse: SynapseSpec {
            r#type: SynapseType::Excitatory,
            resistance: 2.5e3,
        },
    }];

    let mut params = BTreeMap::new();
    params.insert("time".to_string(), json!(0.0));
    params.insert("amp".to_string(), json!(1.0));

    Design {
        version: Some("0.2".to_string()),
        units: Some(Units {
            voltage: Some("V".to_string()),
            time: Some("s".to_string()),
        }),
        globals,
        neuron_templates,
        layers,
        connect,
        overrides: Vec::new(),
        external_inputs: vec![ExternalInputSpec {
            id: "stim".to_string(),
            target: "L0[*]".to_string(),
            wave: "step".to_string(),
            params,
        }],
        readouts: vec![ReadoutSpec {
            id: "out".to_string(),
            r#type: ReadoutType::Analog,
            source_layer: Some("L1".to_string()),
            signal: "analog_background".to_string(),
            post: None,
            tau_s: Some(1e-3),
        }],
        checkpoint: Some(CheckpointSpec {
            every: 0.01,
            fields: vec!["t".to_string()],
        }),
        training: None,
    }
}

#[test]
fn compile_produces_expected_dimensions() {
    let design = simple_design();
    let compiled = compile_design(&design).expect("compile design");
    assert_eq!(compiled.neuron_count(), 3);
    assert_eq!(compiled.synapse_count(), 2); // 2 inputs -> 1 output dense
    assert_eq!(compiled.stimuli.len(), 1);
}

#[test]
fn simulation_generates_positive_readout() {
    let design = simple_design();
    let compiled = compile_design(&design).expect("compile design");
    let mut opts = SimulationOptions::default();
    opts.dt = 1e-4;
    opts.t_end = 5e-3;
    let result = simulate_network(&compiled, &opts);
    let final_values = result
        .readouts
        .get("out")
        .and_then(|samples| samples.last())
        .expect("readout samples");
    assert!(final_values[0] > 0.0);
}

#[test]
fn training_reduces_loss() {
    let design = simple_design();
    let mut compiled = compile_design(&design).expect("compile design");
    let mut opts = SimulationOptions::default();
    opts.dt = 1e-4;
    opts.t_end = 5e-3;

    let mut targets = HashMap::new();
    targets.insert("out".to_string(), vec![0.2]);
    let example = TrainingExample::new(targets, opts.clone());
    let config = TrainerConfig {
        learning_rate: 1e-1,
        epochs: 10,
        regularization: 0.0,
        ..TrainerConfig::default()
    };

    let logs = train_network(&mut compiled, &[example], &config).expect("train network");
    assert!(!logs.is_empty());
    let initial_loss = logs.first().unwrap().loss;
    let final_loss = logs.last().unwrap().loss;
    assert!(final_loss <= initial_loss);
}
