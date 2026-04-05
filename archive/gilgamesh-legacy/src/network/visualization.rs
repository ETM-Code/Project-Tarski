use anyhow::{Context, Result};
use plotters::prelude::{BitMapBackend, ChartBuilder, IntoDrawingArea, LineSeries};
use plotters::style::{
    FontDesc, FontFamily, FontStyle, Palette, Palette99, ShapeStyle, BLUE, WHITE,
};
use serde::Serialize;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

#[derive(Serialize)]
pub struct VisualizationBundle<'a> {
    pub times: &'a [f64],
    pub readouts: &'a HashMap<String, Vec<Vec<f64>>>,
    pub final_readouts: &'a HashMap<String, Vec<f64>>,
    pub neuron_traces: &'a HashMap<usize, Vec<f64>>,
}

pub fn write_visualization_outputs(
    output_dir: &Path,
    times: &[f64],
    readouts: &HashMap<String, Vec<Vec<f64>>>,
    final_readouts: &HashMap<String, Vec<f64>>,
    neuron_traces: &HashMap<usize, Vec<f64>>,
) -> Result<()> {
    fs::create_dir_all(output_dir)
        .with_context(|| format!("failed to create output directory {}", output_dir.display()))?;

    let json_path = output_dir.join("visualization.json");
    let bundle = VisualizationBundle {
        times,
        readouts,
        final_readouts,
        neuron_traces,
    };
    let json_data = serde_json::to_vec_pretty(&bundle)?;
    fs::write(&json_path, json_data)
        .with_context(|| format!("failed to write {}", json_path.display()))?;

    if !readouts.is_empty() {
        for (readout_id, series) in readouts {
            if series.is_empty() || times.is_empty() {
                continue;
            }
            let transposed = transpose_series(series);
            if transposed.is_empty() {
                continue;
            }
            let path = output_dir.join(format!("readout_{}.png", readout_id));
            plot_series(
                &path,
                times,
                &transposed,
                &format!("Readout {}", readout_id),
                "time (s)",
                "value",
            )?;
        }
    }

    if !neuron_traces.is_empty() {
        for (neuron, trace) in neuron_traces {
            let path = output_dir.join(format!("neuron_{}.png", neuron));
            plot_single_series(
                &path,
                times,
                trace,
                &format!("Neuron {} membrane", neuron),
                "time (s)",
                "deflection (V)",
            )?;
        }
    }

    Ok(())
}

fn plot_series(
    path: &Path,
    times: &[f64],
    series: &[Vec<f64>],
    title: &str,
    x_label: &str,
    y_label: &str,
) -> Result<()> {
    if times.is_empty() {
        return Ok(());
    }
    let root = BitMapBackend::new(path, (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    let y_min = series
        .iter()
        .flat_map(|values| values.iter())
        .fold(f64::INFINITY, |a, &v| a.min(v));
    let y_max = series
        .iter()
        .flat_map(|values| values.iter())
        .fold(f64::NEG_INFINITY, |a, &v| a.max(v));
    let y_min = if y_min.is_finite() { y_min } else { 0.0 };
    let y_max = if y_max.is_finite() { y_max } else { 1.0 };

    let x_start = *times.first().unwrap();
    let x_end = *times.last().unwrap_or(&x_start);

    let mut chart = ChartBuilder::on(&root)
        .margin(20)
        .caption(
            title,
            FontDesc::new(FontFamily::SansSerif, 32.0, FontStyle::Normal),
        )
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(x_start..x_end, y_min..y_max)?;

    chart
        .configure_mesh()
        .label_style(FontDesc::new(
            FontFamily::SansSerif,
            18.0,
            FontStyle::Normal,
        ))
        .x_desc(x_label)
        .y_desc(y_label)
        .draw()?;

    for (idx, values) in series.iter().enumerate() {
        let iter = times.iter().zip(values.iter()).map(|(&x, &y)| (x, y));
        let style = ShapeStyle::from(&Palette99::pick(idx)).stroke_width(2);
        chart.draw_series(LineSeries::new(iter, style))?;
    }

    root.present()?;
    Ok(())
}

fn plot_single_series(
    path: &Path,
    times: &[f64],
    values: &[f64],
    title: &str,
    x_label: &str,
    y_label: &str,
) -> Result<()> {
    if times.is_empty() || values.is_empty() {
        return Ok(());
    }
    let root = BitMapBackend::new(path, (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    let y_min = values.iter().fold(f64::INFINITY, |a, &v| a.min(v));
    let y_max = values.iter().fold(f64::NEG_INFINITY, |a, &v| a.max(v));
    let y_min = if y_min.is_finite() { y_min } else { 0.0 };
    let y_max = if y_max.is_finite() { y_max } else { 1.0 };

    let x_start = *times.first().unwrap();
    let x_end = *times.last().unwrap_or(&x_start);

    let mut chart = ChartBuilder::on(&root)
        .margin(20)
        .caption(
            title,
            FontDesc::new(FontFamily::SansSerif, 32.0, FontStyle::Normal),
        )
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(x_start..x_end, y_min..y_max)?;

    chart
        .configure_mesh()
        .label_style(FontDesc::new(
            FontFamily::SansSerif,
            18.0,
            FontStyle::Normal,
        ))
        .x_desc(x_label)
        .y_desc(y_label)
        .draw()?;
    let iter = times.iter().zip(values.iter()).map(|(&x, &y)| (x, y));
    let style = ShapeStyle::from(&BLUE).stroke_width(2);
    chart.draw_series(LineSeries::new(iter, style))?;
    root.present()?;
    Ok(())
}

fn transpose_series(series: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if series.is_empty() {
        return Vec::new();
    }
    let dims = series[0].len();
    if dims == 0 {
        return Vec::new();
    }
    let mut output = vec![Vec::with_capacity(series.len()); dims];
    for time_slice in series {
        for (idx, value) in time_slice.iter().enumerate() {
            output[idx].push(*value);
        }
    }
    output
}
