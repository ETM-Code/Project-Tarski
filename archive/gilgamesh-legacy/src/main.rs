use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use clap::{Parser, Subcommand};
use serde::Deserialize;
use serde_json;

use gilgamesh::network::compiled::{CompiledLayer, CompiledNetwork, LayerRuntimeType};
use gilgamesh::network::compiler::{compile_design, randomize_weights};
use gilgamesh::network::design::{Design, TrainingConfigDefaults};
use gilgamesh::network::runtime::{simulate_network, SimulationOptions, SimulationResult};
use gilgamesh::network::training::{
    reduce_readout_samples, train_network_with_callback, RateEncoder, TemporalEncoder,
    TrainerConfig, TrainingExample,
};
use gilgamesh::network::visualization::write_visualization_outputs;
use gilgamesh::spice_comparator::{run_comparison, ComparisonConfig};

const DEFAULT_DESIGN_PATH: &str = "configs/default_design.json";
const DEFAULT_DATASET_PATH: &str = "src/inputs/training_set/converted/train_7x7.bin";
const DEFAULT_TEST_DATASET_PATH: &str = "src/inputs/testing_set/converted/test_7x7.bin";

#[derive(Parser, Debug)]
#[command(name = "gilgamesh", version, about = "Neuron SPICE comparator toolkit")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    Compare {
        #[arg(long)]
        network: PathBuf,
        #[arg(long)]
        spice_csv: PathBuf,
        #[arg(long)]
        neuron: Option<PathBuf>,
        #[arg(long)]
        json_out: Option<PathBuf>,
        #[arg(long)]
        equivalent_csv: Option<PathBuf>,
        #[arg(long)]
        no_output: bool,
    },
    Simulate {
        #[arg(long, default_value = DEFAULT_DESIGN_PATH)]
        design: PathBuf,
        #[arg(long)]
        output: Option<PathBuf>,
        #[arg(long)]
        compiled_out: Option<PathBuf>,
        #[arg(long)]
        dt: Option<f64>,
        #[arg(long)]
        t_end: Option<f64>,
    },
    Train {
        #[arg(long, default_value = DEFAULT_DESIGN_PATH)]
        design: PathBuf,
        #[arg(long)]
        dataset: Option<PathBuf>,
        #[arg(long)]
        test_dataset: Option<PathBuf>,
        #[arg(long)]
        resume: Option<PathBuf>,
        #[arg(long)]
        checkpoint_dir: Option<PathBuf>,
        #[arg(long)]
        checkpoint_every: Option<usize>,
        #[arg(long)]
        compiled_out: Option<PathBuf>,
        #[arg(long)]
        log: Option<PathBuf>,
        #[arg(long)]
        learning_rate: Option<f64>,
        #[arg(long)]
        epochs: Option<usize>,
        #[arg(long)]
        regularization: Option<f64>,
        #[arg(long)]
        row_spacing_start: Option<f64>,
        #[arg(long)]
        row_spacing_end: Option<f64>,
        #[arg(long)]
        pulse_width: Option<f64>,
        #[arg(long)]
        input_scale: Option<f64>,
        #[arg(long)]
        sample_limit: Option<usize>,
        #[arg(long)]
        target_readout: Option<String>,
        #[arg(long)]
        input_layer: Option<String>,
        /// Random seed for weight initialization (omit to use random entropy)
        #[arg(long)]
        seed: Option<u64>,
        /// Skip random weight initialization (use uniform weights)
        #[arg(long)]
        no_randomize: bool,
    },
    Visualize {
        #[arg(long, default_value = DEFAULT_DESIGN_PATH)]
        design: PathBuf,
        #[arg(long)]
        dataset: Option<PathBuf>,
        #[arg(long)]
        sample: Option<usize>,
        #[arg(long)]
        layer: Option<String>,
        #[arg(long, default_value_t = 8)]
        neurons: usize,
        #[arg(long, default_value = "output/visualizations")]
        output_dir: PathBuf,
        #[arg(long)]
        test: bool,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Compare {
            network,
            spice_csv,
            neuron,
            json_out,
            equivalent_csv,
            no_output,
        } => {
            let config = ComparisonConfig {
                network_config: network,
                neuron_config: neuron,
                spice_csv,
            };

            let mut result = run_comparison(&config)?;

            let mut serialization_ns: u128 = 0;

            if !no_output {
                if let Some(path) = equivalent_csv {
                    let start = Instant::now();
                    result.equivalent.to_csv(&path)?;
                    serialization_ns += start.elapsed().as_nanos();
                }

                if let Some(path) = json_out {
                    result.timings.rust_serialization_ns = serialization_ns;
                    let start = Instant::now();
                    // Serialize once; the timing in the file reflects work up to this point,
                    // avoiding the extra write we previously performed to update the timings.
                    result.write_json(&path)?;
                    serialization_ns += start.elapsed().as_nanos();
                }
            }

            result.timings.rust_serialization_ns = serialization_ns;

            if no_output {
                println!(
                    "[timings] pre={}ns sim={}ns analysis={}ns serialize={}ns",
                    result.timings.rust_preprocess_ns,
                    result.timings.rust_simulate_ns,
                    result.timings.rust_analysis_ns,
                    result.timings.rust_serialization_ns
                );
            }

            for metric in &result.metrics {
                println!(
                    "{}: rms={:.6} mean_abs={:.6} max_abs={:.6} @ {:.6}s ({} samples)",
                    metric.signal,
                    metric.rms_error,
                    metric.mean_abs_error,
                    metric.max_abs_error,
                    metric.max_abs_error_time,
                    metric.sample_count
                );
            }
        }
        Commands::Simulate {
            design,
            output,
            compiled_out,
            dt,
            t_end,
        } => {
            let design = load_design(&design)?;
            let compiled = compile_design(&design)?;

            if let Some(path) = compiled_out {
                write_json_file(&compiled, &path)?;
            }

            let mut opts = SimulationOptions::default();
            if let Some(value) = dt {
                opts.dt = value;
            }
            if let Some(value) = t_end {
                opts.t_end = value;
            }
            opts.record_readout = true;
            opts.return_average = true;

            let result = simulate_network(&compiled, &opts);
            display_simulation(&compiled, &result);

            if let Some(path) = output {
                write_json_file(&result, &path)?;
            }
        }
        Commands::Train {
            design,
            dataset,
            test_dataset,
            resume,
            checkpoint_dir,
            checkpoint_every,
            compiled_out,
            log,
            learning_rate,
            epochs,
            regularization,
            row_spacing_start,
            row_spacing_end,
            pulse_width,
            input_scale,
            sample_limit,
            target_readout,
            input_layer,
            seed,
            no_randomize,
        } => {
            let design_path = design;
            let design = load_design(&design_path)?;
            let mut compiled = compile_design(&design)?;

            let design_dir = design_path.parent().unwrap_or_else(|| Path::new("."));
            let training_defaults = design.training.as_ref();

            let dataset_path = resolve_dataset_path(
                dataset,
                training_defaults.and_then(|defaults| defaults.dataset.as_ref()),
                design_dir,
                DEFAULT_DATASET_PATH,
            );
            let test_dataset_path = resolve_dataset_path(
                test_dataset,
                training_defaults.and_then(|defaults| defaults.test_dataset.as_ref()),
                design_dir,
                DEFAULT_TEST_DATASET_PATH,
            );

            let target_readout = target_readout
                .or_else(|| training_defaults.and_then(|defaults| defaults.target_readout.clone()));
            let input_layer = input_layer
                .or_else(|| training_defaults.and_then(|defaults| defaults.input_layer.clone()));
            let input_layer_for_eval = input_layer.clone();
            let sample_limit = sample_limit.or_else(|| {
                training_defaults
                    .and_then(|defaults| defaults.config.as_ref())
                    .and_then(|cfg| cfg.sample_limit)
            });

            let checkpoint_dir = checkpoint_dir.or_else(|| {
                training_defaults
                    .and_then(|d| d.checkpoint_dir.clone())
                    .map(PathBuf::from)
            });
            let checkpoint_every = checkpoint_every
                .or_else(|| training_defaults.and_then(|d| d.checkpoint_every))
                .unwrap_or(0);
            let checkpoint_dir_resolved = checkpoint_dir.as_ref().map(|dir| {
                if dir.is_relative() {
                    design_dir.join(dir)
                } else {
                    dir.clone()
                }
            });

            let resume_path = if let Some(cli_resume) = resume {
                if cli_resume.is_relative() {
                    Some(design_dir.join(cli_resume))
                } else {
                    Some(cli_resume)
                }
            } else {
                checkpoint_dir_resolved
                    .as_ref()
                    .map(|d| d.join("latest.json"))
            };
            let mut resumed = false;
            if let Some(resume_path) = resume_path {
                if resume_path.exists() {
                    compiled = load_compiled_network(&resume_path)?;
                    println!("resumed weights from {}", resume_path.display());
                    resumed = true;
                } else {
                    println!(
                        "warning: resume checkpoint {} not found; starting from fresh compile",
                        resume_path.display()
                    );
                }
            }

            // Randomize weights if not resuming and not explicitly disabled
            if !resumed && !no_randomize {
                randomize_weights(&mut compiled, seed);
                match seed {
                    Some(s) => println!("initialized weights with seed {}", s),
                    None => println!("initialized weights with random seed"),
                }
            }

            let dataset_bundle = load_training_dataset(
                &dataset_path,
                &compiled,
                target_readout.as_deref(),
                input_layer.as_deref(),
                sample_limit,
            )?;
            let DatasetBundle {
                examples,
                labels: training_labels,
                config: mut trainer_cfg,
                readout_id: training_readout_id,
            } = dataset_bundle;

            let readout_for_training = training_readout_id
                .or_else(|| target_readout.clone())
                .or_else(|| compiled.readouts.first().map(|r| r.id.clone()));

            if let Some(config_defaults) =
                training_defaults.and_then(|defaults| defaults.config.as_ref())
            {
                apply_training_defaults(config_defaults, &mut trainer_cfg);
            }

            if let Some(value) = learning_rate {
                trainer_cfg.learning_rate = value;
            }
            if let Some(value) = epochs {
                trainer_cfg.epochs = value;
            }
            if let Some(value) = regularization {
                trainer_cfg.regularization = value;
            }
            if let Some(value) = row_spacing_start {
                trainer_cfg.row_spacing_start = value;
            }
            if let Some(value) = row_spacing_end {
                trainer_cfg.row_spacing_end = value;
            }
            if let Some(value) = pulse_width {
                trainer_cfg.pulse_width = value;
            }
            if let Some(value) = input_scale {
                trainer_cfg.input_scale = value;
            }
            if let Some(value) = sample_limit {
                trainer_cfg.sample_limit = Some(value);
            }

            let cp_every = checkpoint_every;
            let checkpoint_dir_capture = checkpoint_dir_resolved.clone();
            let logs = train_network_with_callback(
                &mut compiled,
                &examples,
                &trainer_cfg,
                move |net, cfg, log_entry| {
                    println!("epoch {:>3} loss {:.6}", log_entry.epoch, log_entry.loss);
                    if cp_every > 0 && log_entry.epoch % cp_every == 0 {
                        if let Some(dir) = &checkpoint_dir_capture {
                            save_checkpoint(net, dir, log_entry.epoch, cfg)?;
                        }
                    }
                    Ok(())
                },
            )?;

            if let Some(readout_id) = readout_for_training {
                if !training_labels.is_empty() {
                    let train_acc = evaluate_accuracy(
                        &compiled,
                        &examples,
                        &training_labels,
                        &readout_id,
                        &trainer_cfg,
                    )?;
                    println!("train accuracy {:6.2}%", train_acc * 100.0);
                }

                if test_dataset_path.exists() {
                    let test_bundle = load_training_dataset(
                        &test_dataset_path,
                        &compiled,
                        Some(readout_id.as_str()),
                        input_layer_for_eval.as_deref(),
                        None,
                    )?;
                    if !test_bundle.labels.is_empty() {
                        let test_acc = evaluate_accuracy(
                            &compiled,
                            &test_bundle.examples,
                            &test_bundle.labels,
                            &readout_id,
                            &trainer_cfg,
                        )?;
                        println!("test accuracy  {:6.2}%", test_acc * 100.0);
                    } else {
                        println!(
                            "test dataset at {} lacks labels; skipping accuracy",
                            test_dataset_path.display()
                        );
                    }
                } else {
                    println!(
                        "warning: test dataset {} not found; skipping accuracy",
                        test_dataset_path.display()
                    );
                }
            }

            if let Some(dir) = &checkpoint_dir_resolved {
                save_checkpoint(&compiled, dir, trainer_cfg.epochs, &trainer_cfg)?;
            }

            if let Some(path) = log {
                write_json_file(&logs, &path)?;
            }

            if let Some(path) = compiled_out {
                write_json_file(&compiled, &path)?;
            }
        }
        Commands::Visualize {
            design,
            dataset,
            sample,
            layer,
            neurons,
            output_dir,
            test,
        } => {
            let design_path = design;
            let design = load_design(&design_path)?;
            let compiled = compile_design(&design)?;

            let design_dir = design_path.parent().unwrap_or_else(|| Path::new("."));
            let defaults = design.training.as_ref();
            let dataset_default = if test {
                defaults.and_then(|d| d.test_dataset.as_ref())
            } else {
                defaults.and_then(|d| d.dataset.as_ref())
            };
            let fallback = if test {
                DEFAULT_TEST_DATASET_PATH
            } else {
                DEFAULT_DATASET_PATH
            };

            let dataset_path = resolve_dataset_path(dataset, dataset_default, design_dir, fallback);

            let default_readout = defaults
                .and_then(|d| d.target_readout.clone())
                .or_else(|| compiled.readouts.first().map(|r| r.id.clone()))
                .ok_or_else(|| anyhow!("compiled network defines no readouts"))?;
            let default_input_layer = defaults.and_then(|d| d.input_layer.clone());

            let bundle = load_training_dataset(
                &dataset_path,
                &compiled,
                Some(default_readout.as_str()),
                default_input_layer.as_deref(),
                None,
            )?;

            if bundle.examples.is_empty() {
                return Err(anyhow!(
                    "dataset {} contains no samples",
                    dataset_path.display()
                ));
            }

            let sample_idx = sample.unwrap_or(0);
            if sample_idx >= bundle.examples.len() {
                return Err(anyhow!(
                    "sample index {} out of range ({} samples)",
                    sample_idx,
                    bundle.examples.len()
                ));
            }

            let DatasetBundle {
                examples,
                labels,
                config,
                readout_id,
            } = bundle;

            let mut example = examples[sample_idx].clone();
            let label = labels.get(sample_idx).copied();
            let readout_id = readout_id.unwrap_or(default_readout.clone());

            let spacing = config.spacing_for_epoch(config.epochs.saturating_sub(1));
            let pulse_width = config.pulse_width_for_spacing(spacing);
            let scale = config.input_scale;

            eprintln!("DEBUG: spacing={}, pulse_width={}, scale={}", spacing, pulse_width, scale);
            eprintln!("DEBUG: has_encoder={}, has_rate_encoder={}", example.encoder.is_some(), example.rate_encoder.is_some());

            if let Some(encoder) = example.encoder.as_ref() {
                let stim = encoder.build_stimuli(spacing, pulse_width, scale);
                for s in &stim {
                    let max_val = s.values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    eprintln!("DEBUG: stim {} targets {:?}, max_value={}", s.id, s.target_indices, max_val);
                }
            }

            if let Some(encoder) = example.rate_encoder.as_ref() {
                // Rate-coded: all pixels at once
                example.simulation.stimuli_override =
                    Some(encoder.build_stimuli(encoder.suggested_duration(), scale));
                example.simulation.t_end = encoder.suggested_duration();
            } else if let Some(encoder) = example.encoder.as_ref() {
                // Temporal: row-by-row
                example.simulation.stimuli_override =
                    Some(encoder.build_stimuli(spacing, pulse_width, scale));
                let duration = encoder.suggested_duration(spacing);
                if example.simulation.t_end < duration {
                    example.simulation.t_end = duration;
                }
            }

            example.simulation.record_readout = true;
            example.simulation.return_average = false;
            if spacing > 0.0 {
                let suggested_dt = (spacing / 25.0).max(1e-6);
                if example.simulation.dt > suggested_dt {
                    example.simulation.dt = suggested_dt;
                }
            }

            let (trace_indices, traced_layer_id) = select_trace_neurons(
                &compiled,
                layer.as_deref(),
                default_input_layer.as_deref(),
                neurons,
            );
            example.simulation.trace_neurons = trace_indices.clone();

            let result = simulate_network(&compiled, &example.simulation);

            let mut run_dir = output_dir.clone();
            run_dir.push(if test { "test" } else { "train" });
            run_dir.push(format!("sample_{:04}", sample_idx));

            write_visualization_outputs(
                &run_dir,
                &result.times,
                &result.readouts,
                &result.final_readouts,
                &result.neuron_traces,
            )?;

            println!(
                "visualization written to {} (label: {})",
                run_dir.display(),
                label
                    .map(|l| l.to_string())
                    .unwrap_or_else(|| "unknown".to_string())
            );
            println!("focused readout '{}'", readout_id);
            if let Some(layer_id) = traced_layer_id {
                println!(
                    "traced {} neurons from layer '{}'",
                    trace_indices.len(),
                    layer_id
                );
            }
        }
    }

    Ok(())
}

fn load_design(path: &Path) -> Result<Design> {
    let data = fs::read_to_string(path)
        .with_context(|| format!("failed to read design file {}", path.display()))?;
    let design: Design = serde_json::from_str(&data)
        .with_context(|| format!("failed to parse design file {}", path.display()))?;
    Ok(design)
}

fn write_json_file<T>(value: &T, path: &Path) -> Result<()>
where
    T: serde::Serialize,
{
    let mut file =
        fs::File::create(path).with_context(|| format!("failed to create {}", path.display()))?;
    serde_json::to_writer_pretty(&mut file, value)
        .with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}

fn load_compiled_network(path: &Path) -> Result<CompiledNetwork> {
    let data = fs::read_to_string(path)
        .with_context(|| format!("failed to read checkpoint {}", path.display()))?;
    let network: CompiledNetwork = serde_json::from_str(&data)
        .with_context(|| format!("failed to parse checkpoint {}", path.display()))?;
    Ok(network)
}

fn save_checkpoint(
    network: &CompiledNetwork,
    dir: &Path,
    epoch: usize,
    config: &TrainerConfig,
) -> Result<()> {
    fs::create_dir_all(dir)
        .with_context(|| format!("failed to create checkpoint directory {}", dir.display()))?;

    let mut network_with_checkpoint = network.clone();
    network_with_checkpoint.training_checkpoint =
        Some(gilgamesh::network::compiled::TrainingCheckpoint {
            epoch,
            learning_rate: config.learning_rate,
            row_spacing_start: config.row_spacing_start,
            row_spacing_end: config.row_spacing_end,
            pulse_width: config.pulse_width,
            input_scale: config.input_scale,
            regularization: config.regularization,
        });

    let filename = format!("checkpoint_epoch_{:04}.json", epoch);
    let path = dir.join(&filename);
    write_json_file(&network_with_checkpoint, &path)?;
    let latest = dir.join("latest.json");
    write_json_file(&network_with_checkpoint, &latest)?;
    Ok(())
}

fn display_simulation(compiled: &CompiledNetwork, result: &SimulationResult) {
    println!("simulation completed at t={:.6}s", result.final_state.time);
    for readout in &compiled.readouts {
        match result
            .readouts
            .get(&readout.id)
            .and_then(|samples| samples.last())
        {
            Some(values) => {
                print!("{}:", readout.id);
                for (idx, value) in values.iter().enumerate() {
                    print!(" [{}]={:.6}", idx, value);
                }
                println!();
            }
            None => {
                println!("{}: no samples recorded", readout.id);
            }
        }
    }

    if let Some(avg) = &result.average_activity {
        let total: f64 = avg.iter().map(|v| v.abs()).sum();
        if total > 0.0 {
            println!("average membrane activity (L1 norm): {:.6}", total);
        }
    }
}

struct DatasetBundle {
    examples: Vec<TrainingExample>,
    labels: Vec<usize>,
    config: TrainerConfig,
    readout_id: Option<String>,
}

fn load_training_dataset(
    path: &Path,
    network: &CompiledNetwork,
    target_readout: Option<&str>,
    input_layer: Option<&str>,
    sample_limit: Option<usize>,
) -> Result<DatasetBundle> {
    let resolved = if path.is_dir() {
        let candidate = path.join("train_7x7.bin");
        if candidate.exists() {
            candidate
        } else {
            return Err(anyhow!(
                "dataset directory {} missing train_7x7.bin",
                path.display()
            ));
        }
    } else {
        path.to_path_buf()
    };

    let ext = resolved
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    match ext.as_str() {
        "json" => load_training_dataset_json(&resolved),
        "bin" => load_mnist7x7_dataset(
            &resolved,
            network,
            target_readout,
            input_layer,
            sample_limit,
        ),
        other => Err(anyhow!("unsupported training dataset format '{}'", other)),
    }
}

fn load_training_dataset_json(path: &Path) -> Result<DatasetBundle> {
    let data = fs::read_to_string(path)
        .with_context(|| format!("failed to read training dataset {}", path.display()))?;
    let raw: TrainingDatasetFile = serde_json::from_str(&data)
        .with_context(|| format!("failed to parse training dataset {}", path.display()))?;
    raw.into_dataset()
}

fn load_mnist7x7_dataset(
    path: &Path,
    network: &CompiledNetwork,
    target_readout: Option<&str>,
    input_layer: Option<&str>,
    sample_limit: Option<usize>,
) -> Result<DatasetBundle> {
    let input_indices = resolve_input_indices(network, input_layer)?;
    let readout = resolve_target_readout(network, target_readout)?;

    let mut file = BufReader::new(
        File::open(path)
            .with_context(|| format!("failed to open mnist dataset {}", path.display()))?,
    );

    let mut magic = [0u8; 4];
    file.read_exact(&mut magic)
        .context("failed reading dataset magic header")?;
    if &magic != b"M7x7" {
        return Err(anyhow!("unexpected dataset magic {:?}", magic));
    }

    let mut buf4 = [0u8; 4];
    file.read_exact(&mut buf4)?;
    let count = u32::from_le_bytes(buf4) as usize;

    let mut single = [0u8; 1];
    file.read_exact(&mut single)?;
    let rows = single[0] as usize;
    file.read_exact(&mut single)?;
    let cols = single[0] as usize;

    if rows == 0 || cols == 0 {
        return Err(anyhow!("dataset rows/cols must be > 0"));
    }

    let mut reserved = [0u8; 2];
    file.read_exact(&mut reserved)?;

    let limit = sample_limit.unwrap_or(count).min(count);
    let mut examples = Vec::with_capacity(limit);
    let mut labels = Vec::with_capacity(limit);

    for sample_idx in 0..limit {
        let mut pixels = vec![0u8; rows * cols];
        file.read_exact(&mut pixels)
            .with_context(|| format!("failed reading pixels for sample {}", sample_idx))?;
        file.read_exact(&mut single)
            .with_context(|| format!("failed reading label for sample {}", sample_idx))?;
        let label = single[0] as usize;

        let mut columns = vec![vec![0.0f64; rows]; cols];
        for row in 0..rows {
            for col in 0..cols {
                let value = pixels[row * cols + col] as f64 / 255.0;
                columns[col][row] = value;
            }
        }

        let mut targets = HashMap::new();
        let mut target_vec = vec![0.0; readout.indices.len()];
        if label < target_vec.len() {
            target_vec[label] = 1.0;
        }
        targets.insert(readout.id.clone(), target_vec);

        let mut sim_opts = SimulationOptions::default();
        let default_spacing = TrainerConfig::default().row_spacing_start;
        sim_opts.t_end = (rows as f64 + 1.0) * default_spacing;
        sim_opts.dt = 1e-4;

        // Choose encoder based on input layer size:
        // - 49 neurons: rate-coded (all pixels at once)
        // - 7 neurons: temporal (row-by-row)
        let example = if input_indices.len() >= 49 {
            // Rate-coded: flatten all pixels and present simultaneously
            let mut pixel_values = Vec::with_capacity(rows * cols);
            for row in 0..rows {
                for col in 0..cols {
                    pixel_values.push(columns[col][row]);
                }
            }
            let rate_encoder = RateEncoder::new(pixel_values, input_indices[..49].to_vec());

            // For rate coding, use fixed duration (no row spacing)
            sim_opts.t_end = rate_encoder.suggested_duration();

            TrainingExample::new(targets, sim_opts).with_rate_encoder(rate_encoder)
        } else if input_indices.len() >= cols {
            // Temporal: present columns row-by-row
            let encoder = TemporalEncoder::new(cols, rows, columns, input_indices[..cols].to_vec());
            TrainingExample::new(targets, sim_opts).with_encoder(encoder)
        } else {
            return Err(anyhow!(
                "input layer has {} neurons but requires at least {} columns",
                input_indices.len(),
                cols
            ));
        };

        examples.push(example);
        labels.push(label);
    }

    if examples.is_empty() {
        return Err(anyhow!("no samples loaded from {}", path.display()));
    }

    let mut config = TrainerConfig::default();
    config.sample_limit = sample_limit;
    Ok(DatasetBundle {
        examples,
        labels,
        config,
        readout_id: Some(readout.id.clone()),
    })
}

fn resolve_input_indices(network: &CompiledNetwork, layer_id: Option<&str>) -> Result<Vec<usize>> {
    let layer = if let Some(id) = layer_id {
        network
            .layers
            .iter()
            .find(|layer| layer.id == id)
            .ok_or_else(|| anyhow!("input layer '{}' not found", id))?
    } else {
        network
            .layers
            .iter()
            .find(|layer| matches!(layer.layer_type, LayerRuntimeType::Input))
            .ok_or_else(|| anyhow!("compiled network missing input layer"))?
    };

    Ok((layer.offset..layer.offset + layer.size).collect())
}

fn resolve_target_readout<'a>(
    network: &'a CompiledNetwork,
    readout_id: Option<&str>,
) -> Result<&'a gilgamesh::network::compiled::CompiledReadout> {
    let readout = if let Some(id) = readout_id {
        network
            .readouts
            .iter()
            .find(|readout| readout.id == id)
            .ok_or_else(|| anyhow!("readout '{}' not found", id))?
    } else {
        network
            .readouts
            .first()
            .ok_or_else(|| anyhow!("compiled network defines no readouts"))?
    };
    Ok(readout)
}

#[derive(Debug, Deserialize)]
struct TrainingDatasetFile {
    examples: Vec<TrainingExampleFile>,
    #[serde(default)]
    config: Option<TrainerConfigFile>,
}

impl TrainingDatasetFile {
    fn into_dataset(self) -> Result<DatasetBundle> {
        if self.examples.is_empty() {
            return Err(anyhow!(
                "training dataset must contain at least one example"
            ));
        }

        let mut examples = Vec::with_capacity(self.examples.len());
        let labels = Vec::new();
        for (idx, example) in self.examples.into_iter().enumerate() {
            if example.targets.is_empty() {
                return Err(anyhow!("training example {} missing targets", idx));
            }
            let mut opts = SimulationOptions::default();
            if let Some(sim) = example.simulation {
                if let Some(value) = sim.dt {
                    opts.dt = value;
                }
                if let Some(value) = sim.t_end {
                    opts.t_end = value;
                }
            }
            let mut training_example = TrainingExample::new(example.targets, opts);
            if let Some(weight) = example.weight {
                training_example = training_example.with_weight(weight);
            }
            examples.push(training_example);
        }

        let config = self.config.unwrap_or_default().into_config();
        Ok(DatasetBundle {
            examples,
            labels,
            config,
            readout_id: None,
        })
    }
}

#[derive(Debug, Deserialize)]
struct TrainingExampleFile {
    targets: HashMap<String, Vec<f64>>,
    #[serde(default)]
    weight: Option<f64>,
    #[serde(default)]
    simulation: Option<SimulationOptionsFile>,
}

#[derive(Debug, Deserialize, Default)]
struct SimulationOptionsFile {
    #[serde(default)]
    t_end: Option<f64>,
    #[serde(default)]
    dt: Option<f64>,
}

#[derive(Debug, Deserialize, Default)]
struct TrainerConfigFile {
    #[serde(default)]
    learning_rate: Option<f64>,
    #[serde(default)]
    epochs: Option<usize>,
    #[serde(default)]
    regularization: Option<f64>,
    #[serde(default)]
    row_spacing_start: Option<f64>,
    #[serde(default)]
    row_spacing_end: Option<f64>,
    #[serde(default)]
    pulse_width: Option<f64>,
    #[serde(default)]
    input_scale: Option<f64>,
    #[serde(default)]
    sample_limit: Option<usize>,
}

impl TrainerConfigFile {
    fn into_config(self) -> TrainerConfig {
        let mut cfg = TrainerConfig::default();
        if let Some(value) = self.learning_rate {
            cfg.learning_rate = value;
        }
        if let Some(value) = self.epochs {
            cfg.epochs = value;
        }
        if let Some(value) = self.regularization {
            cfg.regularization = value;
        }
        if let Some(value) = self.row_spacing_start {
            cfg.row_spacing_start = value;
        }
        if let Some(value) = self.row_spacing_end {
            cfg.row_spacing_end = value;
        }
        if let Some(value) = self.pulse_width {
            cfg.pulse_width = value;
        }
        if let Some(value) = self.input_scale {
            cfg.input_scale = value;
        }
        if let Some(value) = self.sample_limit {
            cfg.sample_limit = Some(value);
        }
        cfg
    }
}

fn resolve_dataset_path(
    cli_value: Option<PathBuf>,
    design_default: Option<&String>,
    design_dir: &Path,
    fallback: &str,
) -> PathBuf {
    if let Some(path) = cli_value {
        return path;
    }

    if let Some(design_path) = design_default {
        let candidate = PathBuf::from(design_path);
        if candidate.is_relative() {
            design_dir.join(candidate)
        } else {
            candidate
        }
    } else {
        PathBuf::from(fallback)
    }
}

fn apply_training_defaults(defaults: &TrainingConfigDefaults, cfg: &mut TrainerConfig) {
    if let Some(value) = defaults.learning_rate {
        cfg.learning_rate = value;
    }
    if let Some(value) = defaults.epochs {
        cfg.epochs = value;
    }
    if let Some(value) = defaults.regularization {
        cfg.regularization = value;
    }
    if let Some(value) = defaults.row_spacing_start {
        cfg.row_spacing_start = value;
    }
    if let Some(value) = defaults.row_spacing_end {
        cfg.row_spacing_end = value;
    }
    if let Some(value) = defaults.pulse_width {
        cfg.pulse_width = value;
    }
    if let Some(value) = defaults.input_scale {
        cfg.input_scale = value;
    }
    if let Some(value) = defaults.sample_limit {
        cfg.sample_limit = Some(value);
    }
}

fn evaluate_accuracy(
    network: &CompiledNetwork,
    examples: &[TrainingExample],
    labels: &[usize],
    readout_id: &str,
    cfg: &TrainerConfig,
) -> Result<f64> {
    if examples.len() != labels.len() || examples.is_empty() {
        return Err(anyhow!(
            "dataset for evaluation must contain equal numbers of examples and labels"
        ));
    }

    let readout = network
        .readouts
        .iter()
        .find(|r| r.id == readout_id)
        .ok_or_else(|| anyhow!("readout '{}' not found in compiled network", readout_id))?;

    let mut correct = 0usize;
    let spacing = cfg.spacing_for_epoch(cfg.epochs.saturating_sub(1));
    let pulse_width = cfg.pulse_width_for_spacing(spacing);
    let scale = cfg.input_scale;

    for (example, &label) in examples.iter().zip(labels.iter()) {
        let mut sim_opts = example.simulation.clone();
        sim_opts.record_readout = true;
        sim_opts.return_average = false;

        if let Some(encoder) = example.rate_encoder.as_ref() {
            // Rate-coded: all pixels at once
            sim_opts.stimuli_override = Some(encoder.build_stimuli(encoder.suggested_duration(), scale));
            sim_opts.t_end = encoder.suggested_duration();
        } else if let Some(encoder) = example.encoder.as_ref() {
            // Temporal: row-by-row
            sim_opts.stimuli_override = Some(encoder.build_stimuli(spacing, pulse_width, scale));
            let duration = encoder.suggested_duration(spacing);
            if sim_opts.t_end < duration {
                sim_opts.t_end = duration;
            }
        }

        let result = simulate_network(network, &sim_opts);
        let samples = result
            .readouts
            .get(readout_id)
            .ok_or_else(|| anyhow!("readout '{}' missing from simulation output", readout_id))?;
        if samples.is_empty() {
            return Err(anyhow!(
                "readout '{}' produced no samples during simulation",
                readout_id
            ));
        }

        let num_outputs = samples.first().map(|s| s.len()).unwrap_or(0);
        let logits = reduce_readout_samples(readout.r#type.clone(), samples, num_outputs);
        let predicted = argmax_index(&logits);
        if predicted == label {
            correct += 1;
        }
    }

    Ok(correct as f64 / examples.len() as f64)
}

fn argmax_index(values: &[f64]) -> usize {
    let mut best_idx = 0usize;
    let mut best_val = f64::NEG_INFINITY;
    for (idx, &value) in values.iter().enumerate() {
        if value > best_val {
            best_val = value;
            best_idx = idx;
        }
    }
    best_idx
}

fn select_trace_neurons(
    network: &CompiledNetwork,
    requested_layer: Option<&str>,
    fallback_layer: Option<&str>,
    count: usize,
) -> (Vec<usize>, Option<String>) {
    let candidate = requested_layer
        .and_then(|id| find_layer(network, id))
        .or_else(|| fallback_layer.and_then(|id| find_layer(network, id)))
        .or_else(|| {
            network
                .layers
                .iter()
                .find(|layer| matches!(layer.layer_type, LayerRuntimeType::Hidden))
        })
        .or_else(|| network.layers.first());

    if let Some(layer) = candidate {
        let take = if count == 0 {
            layer.size
        } else {
            count.min(layer.size)
        };
        let indices = (layer.offset..layer.offset + take).collect();
        return (indices, Some(layer.id.clone()));
    }

    (Vec::new(), None)
}

fn find_layer<'a>(network: &'a CompiledNetwork, id: &str) -> Option<&'a CompiledLayer> {
    network.layers.iter().find(|layer| layer.id == id)
}
