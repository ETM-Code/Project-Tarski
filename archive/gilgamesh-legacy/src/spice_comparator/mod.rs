use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::simulators::config::{NetworkConfig, NeuronConfig};
use crate::simulators::lif_equivalent::{EquivalentNeuron, SimulationResult, SimulationSample};

#[derive(Debug, Clone)]
pub struct ComparisonConfig {
    pub network_config: PathBuf,
    pub neuron_config: Option<PathBuf>,
    pub spice_csv: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalComparison {
    pub signal: String,
    pub rms_error: f64,
    pub mean_abs_error: f64,
    pub max_abs_error: f64,
    pub max_abs_error_time: f64,
    pub sample_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SignalSamplePair {
    pub time_s: f64,
    pub equivalent: f64,
    pub spice: f64,
    pub delta: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonResult {
    pub equivalent: SimulationResult,
    pub metrics: Vec<SignalComparison>,
    pub signal_series: HashMap<String, Vec<SignalSamplePair>>,
    #[serde(default)]
    pub timings: TimingBreakdown,
}

impl ComparisonResult {
    pub fn write_json(&self, path: &Path) -> Result<()> {
        let json = serde_json::to_vec_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TimingBreakdown {
    pub rust_preprocess_ns: u128,
    pub rust_simulate_ns: u128,
    pub rust_analysis_ns: u128,
    pub rust_serialization_ns: u128,
    pub spice_parse_ns: u128,
    #[serde(default)]
    pub spice_preprocess_ns: Option<u128>,
    #[serde(default)]
    pub spice_simulate_ns: Option<u128>,
    #[serde(default)]
    pub spice_post_ns: Option<u128>,
}

pub fn run_comparison(cfg: &ComparisonConfig) -> Result<ComparisonResult> {
    let network_cfg = NetworkConfig::load(&cfg.network_config).with_context(|| {
        format!(
            "failed to load network config at {}",
            cfg.network_config.display()
        )
    })?;

    let neuron_path = if let Some(path) = &cfg.neuron_config {
        path.clone()
    } else {
        let base = cfg
            .network_config
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        base.join(&network_cfg.neuron_json)
    };

    let neuron_cfg = NeuronConfig::load(&neuron_path)
        .with_context(|| format!("failed to load neuron config at {}", neuron_path.display()))?;

    let build_start = Instant::now();
    let equivalent_model = EquivalentNeuron::from_configs(&neuron_cfg, &network_cfg);
    let rust_preprocess_ns = build_start.elapsed().as_nanos();

    let sim_start = Instant::now();
    let equivalent = equivalent_model.run();
    let rust_simulate_ns = sim_start.elapsed().as_nanos();

    let spice_start = Instant::now();
    let spice_trace = SpiceTrace::from_ascii(&cfg.spice_csv)
        .with_context(|| format!("failed to load SPICE CSV at {}", cfg.spice_csv.display()))?;
    let spice_parse_ns = spice_start.elapsed().as_nanos();

    let analysis_start = Instant::now();
    let (metrics, signal_series) = compare_against_spice(&equivalent, &spice_trace)?;
    let rust_analysis_ns = analysis_start.elapsed().as_nanos();

    Ok(ComparisonResult {
        equivalent,
        metrics,
        signal_series,
        timings: TimingBreakdown {
            rust_preprocess_ns,
            rust_simulate_ns,
            rust_analysis_ns,
            rust_serialization_ns: 0,
            spice_parse_ns,
            ..TimingBreakdown::default()
        },
    })
}

struct SpiceTrace {
    times: Vec<f64>,
    signals: HashMap<String, Vec<f64>>,
}

impl SpiceTrace {
    /// Parse the SPICE ASCII CSV, retaining only the time column and the signals we compare.
    fn from_ascii(path: &Path) -> Result<Self> {
        let file = File::open(path)
            .with_context(|| format!("unable to open SPICE output file {}", path.display()))?;
        let reader = BufReader::new(file);

        let mut header: Option<Vec<String>> = None;
        // Column indexes for the signals we care about.
        let mut time_idx: Option<usize> = None;
        let mut signal_indexes: Vec<(usize, String)> = Vec::new();
        let mut index_lookup: HashMap<usize, usize> = HashMap::new();

        let mut times: Vec<f64> = Vec::new();
        let mut signals: Vec<Vec<f64>> = Vec::new();
        let mut names: Vec<String> = Vec::new();

        for line in reader.lines() {
            let line = line?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            if header.is_none() {
                let cols: Vec<String> = trimmed.split_whitespace().map(|s| s.to_string()).collect();
                if cols.is_empty() {
                    bail!("SPICE header row is empty in {}", path.display());
                }

                time_idx = cols.iter().position(|c| c == "time");
                if time_idx.is_none() {
                    bail!("missing 'time' column in {}", path.display());
                }

                for spec in SIGNAL_SPECS {
                    if let Some(pos) = cols.iter().position(|c| c == spec.spice_column) {
                        signal_indexes.push((pos, spec.spice_column.to_string()));
                    }
                }

                if signal_indexes.is_empty() {
                    bail!(
                        "no expected signals ({:?}) found in {}",
                        SIGNAL_SPECS
                            .iter()
                            .map(|s| s.spice_column)
                            .collect::<Vec<_>>(),
                        path.display()
                    );
                }

                signal_indexes.sort_by_key(|(idx, _)| *idx);
                for (vec_idx, (col_idx, name)) in signal_indexes.iter().enumerate() {
                    index_lookup.insert(*col_idx, vec_idx);
                    names.push(name.clone());
                    signals.push(Vec::new());
                }

                header = Some(cols);
                continue;
            }

            let mut time_val: Option<f64> = None;

            for (col_idx, field) in trimmed.split_whitespace().enumerate() {
                if Some(col_idx) == time_idx {
                    time_val = Some(field.parse::<f64>().with_context(|| {
                        format!("failed to parse time value in {}", path.display())
                    })?);
                    continue;
                }

                if let Some(&vec_idx) = index_lookup.get(&col_idx) {
                    let value = field.parse::<f64>().with_context(|| {
                        format!("failed to parse numeric data in {}", path.display())
                    })?;
                    signals[vec_idx].push(value);
                }
            }

            if let Some(t) = time_val {
                times.push(t);
            }
        }

        if times.is_empty() {
            bail!("SPICE trace is missing time samples in {}", path.display());
        }

        for (idx, values) in signals.iter().enumerate() {
            if values.len() != times.len() {
                bail!(
                    "signal '{}' has {} samples but time has {} in {}",
                    names[idx],
                    values.len(),
                    times.len(),
                    path.display()
                );
            }
        }

        let mut signal_map = HashMap::new();
        for (name, values) in names.into_iter().zip(signals.into_iter()) {
            signal_map.insert(name, values);
        }

        Ok(SpiceTrace {
            times,
            signals: signal_map,
        })
    }
}

struct SignalSpec {
    id: &'static str,
    spice_column: &'static str,
    extractor: fn(&SimulationSample) -> f64,
}

/// Upper bound on how many comparison samples we keep per signal for plotting.
/// Metrics are always computed on the full-resolution data.
// Keep more points for plotting when enabled; metrics still use full data.
const MAX_SERIES_SAMPLES: usize = 10_000;

const SIGNAL_SPECS: &[SignalSpec] = &[
    SignalSpec {
        id: "v_mem",
        spice_column: "v(mem)",
        extractor: |s: &SimulationSample| s.v_mem,
    },
    SignalSpec {
        id: "v_comp",
        spice_column: "v(comp)",
        extractor: |s: &SimulationSample| s.v_comp,
    },
    SignalSpec {
        id: "v_outmix",
        spice_column: "v(n_outmix)",
        extractor: |s: &SimulationSample| s.v_outmix,
    },
    SignalSpec {
        id: "v_ana",
        spice_column: "v(ana)",
        extractor: |s: &SimulationSample| s.v_ana,
    },
];

fn compare_against_spice(
    equivalent: &SimulationResult,
    spice: &SpiceTrace,
) -> Result<(
    Vec<SignalComparison>,
    HashMap<String, Vec<SignalSamplePair>>,
)> {
    let mut metrics = Vec::new();
    let mut series_map: HashMap<String, Vec<SignalSamplePair>> = HashMap::new();

    for spec in SIGNAL_SPECS {
        let mut sum_sq = 0.0;
        let mut sum_abs = 0.0;
        let mut max_abs = 0.0;
        let mut max_time = 0.0;
        let mut count = 0usize;
        let mut series = Vec::new();

        // Walk the SPICE samples once per signal using a forward tooling to avoid
        // repeated binary searches.
        let spice_values = match spice.signals.get(spec.spice_column) {
            Some(v) => v,
            None => continue,
        };
        let mut tooling = 0usize;
        let last_idx = spice.times.len().saturating_sub(1);

        for sample in &equivalent.samples {
            let eq_val = (spec.extractor)(sample);
            let time = sample.time_s;
            if time < spice.times.first().copied().unwrap_or_default()
                || time > spice.times.last().copied().unwrap_or_default()
            {
                continue;
            }

            while tooling + 1 < last_idx && spice.times[tooling + 1] < time {
                tooling += 1;
            }

            let t0 = spice.times[tooling];
            let t1 = spice.times[(tooling + 1).min(last_idx)];
            let v0 = spice_values[tooling];
            let v1 = spice_values[(tooling + 1).min(last_idx)];

            let spice_val = if (t1 - t0).abs() < f64::EPSILON {
                v0
            } else {
                let alpha = ((time - t0) / (t1 - t0)).clamp(0.0, 1.0);
                v0 + (v1 - v0) * alpha
            };

            let delta = eq_val - spice_val;
            let abs_delta = delta.abs();
            sum_sq += delta * delta;
            sum_abs += abs_delta;
            if abs_delta > max_abs {
                max_abs = abs_delta;
                max_time = time;
            }
            count += 1;
            series.push(SignalSamplePair {
                time_s: time,
                equivalent: eq_val,
                spice: spice_val,
                delta,
            });
        }

        if count == 0 {
            continue;
        }

        let rms = (sum_sq / count as f64).sqrt();
        let mean_abs = sum_abs / count as f64;
        metrics.push(SignalComparison {
            signal: spec.id.to_string(),
            rms_error: rms,
            mean_abs_error: mean_abs,
            max_abs_error: max_abs,
            max_abs_error_time: max_time,
            sample_count: count,
        });
        series_map.insert(
            spec.id.to_string(),
            downsample_series(series, MAX_SERIES_SAMPLES),
        );
    }

    Ok((metrics, series_map))
}

fn downsample_series(series: Vec<SignalSamplePair>, max_samples: usize) -> Vec<SignalSamplePair> {
    if series.len() <= max_samples || max_samples == 0 {
        return series;
    }

    let stride = ((series.len() as f64) / (max_samples as f64)).ceil() as usize;
    if stride <= 1 {
        return series;
    }

    let mut reduced = Vec::with_capacity(series.len() / stride + 1);
    let mut idx = 0usize;
    while idx < series.len() {
        reduced.push(series[idx].clone());
        idx += stride;
    }

    if let Some(last) = series.last() {
        if reduced.last() != Some(last) {
            reduced.push(last.clone());
        }
    }

    reduced
}
