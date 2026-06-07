#!/usr/bin/env python3
"""Tarski Board — Interactive Terminal UI.

Usage:
    uv run tarski_tui.py --port /dev/tty.usbserial-XXX
    uv run tarski_tui.py --port /dev/tty.usbserial-XXX --non-interactive --calibrate --infer
"""

import argparse
import json
import sys

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, ScrollableContainer
from textual.widgets import (
    Header, Footer, Static, Button, Label, RichLog
)
from textual.reactive import reactive
from textual import work

from tarski_board import TarskiBoard, load_mnist, normalize_mnist

import numpy as np


def downsample_array_split(img: np.ndarray) -> np.ndarray:
    """Downsample a 2-D image to 6x6 via np.array_split block means.

    NOTE: this DIVERGES from the board's int(ty*scale) floor-block downsample.
    For a non-divisible axis length (e.g. 28) np.array_split yields uneven
    blocks, so the nested np.array is inhomogeneous and raises ValueError —
    behavior the characterization suite pins deliberately.
    """
    h_blocks = np.array_split(img, 6, axis=0)
    small = np.array([np.array_split(hb, 6, axis=1) for hb in h_blocks])
    return np.array([[block.mean() for block in row] for row in small])


class StatusPanel(Static):
    """Connection status and firmware info."""

    connected = reactive(False)
    version = reactive("—")

    def render(self) -> str:
        dot = "[green]●[/]" if self.connected else "[red]●[/]"
        return f"{dot} {'Connected' if self.connected else 'Disconnected'}  Firmware: v{self.version}"


class PredictionDisplay(Static):
    """Large prediction digit display."""

    prediction = reactive(-1)
    true_label = reactive(-1)
    spike_counts = reactive([0]*10)

    def render(self) -> str:
        if self.prediction < 0:
            return "[dim]No prediction yet[/]"

        correct = self.prediction == self.true_label if self.true_label >= 0 else None
        color = "green" if correct else ("red" if correct is False else "blue")
        label_str = f"  (label: {self.true_label})" if self.true_label >= 0 else ""

        bars = ""
        max_c = max(self.spike_counts) if any(self.spike_counts) else 1
        for i, c in enumerate(self.spike_counts):
            bar_len = int(c / max_c * 15) if max_c > 0 else 0
            marker = "[bold green]" if i == self.prediction else "[dim]"
            bars += f"  {marker}{i}: {'█' * bar_len}{'░' * (15-bar_len)} {c}[/]\n"

        return f"[bold {color}]  Prediction: {self.prediction}[/]{label_str}\n\n{bars}"


class TarskiTUI(App):
    """Interactive terminal UI for the Tarski neuromorphic board."""

    CSS = """
    Screen {
        layout: grid;
        grid-size: 2 3;
        grid-rows: 3 1fr 1fr;
        grid-columns: 1fr 1fr;
    }
    #status { column-span: 2; height: 3; padding: 0 1; }
    #actions { height: 100%; padding: 1; border: solid $accent; }
    #prediction { height: 100%; padding: 1; border: solid $success; }
    #serial-log { column-span: 2; height: 100%; padding: 1; border: solid $primary; }
    Button { margin: 0 1 1 0; }
    .action-row { height: auto; }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("c", "calibrate", "Calibrate"),
        ("i", "infer_sample", "Infer Sample"),
        ("n", "next_sample", "Next Sample"),
        ("p", "prev_sample", "Prev Sample"),
    ]

    def __init__(self, port: str, checkpoint: str = None, data_dir: str = None):
        super().__init__()
        self.port = port
        self.checkpoint_path = checkpoint
        self.data_dir = data_dir or "../gilgamesh/data"
        self.board: TarskiBoard | None = None
        self.calib: dict | None = None
        self.fc1_weights = None
        self.fc2_quantized = None
        self.images = None
        self.labels = None
        self.sample_idx = 0
        self.total_correct = 0
        self.total_tested = 0

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield StatusPanel(id="status")

        with Vertical(id="actions"):
            yield Label("[bold]Actions[/]")
            with Horizontal(classes="action-row"):
                yield Button("Connect", id="btn-connect", variant="primary")
                yield Button("Setup DACs", id="btn-setup-dacs")
                yield Button("Calibrate", id="btn-calibrate", variant="warning")
            with Horizontal(classes="action-row"):
                yield Button("Load Model", id="btn-load-model")
                yield Button("Infer", id="btn-infer", variant="success")
                yield Button("Run 100", id="btn-run-100")
            with Horizontal(classes="action-row"):
                yield Button("← Prev", id="btn-prev")
                yield Button("Next →", id="btn-next")
                yield Label(f"Sample: {self.sample_idx}", id="sample-label")

        yield PredictionDisplay(id="prediction")

        with ScrollableContainer(id="serial-log"):
            yield Label("[bold]Serial Log[/]")
            yield RichLog(id="log", highlight=True, markup=True)

        yield Footer()

    def on_mount(self) -> None:
        self.log_msg("[bold]Tarski Board Interface[/]")
        self.log_msg(f"Port: {self.port}")
        if self.checkpoint_path:
            self.log_msg(f"Checkpoint: {self.checkpoint_path}")

    def log_msg(self, msg: str) -> None:
        log = self.query_one("#log", RichLog)
        log.write(msg)

    # ── Button handlers ──

    _BUTTON_ACTIONS = {
        "btn-connect": "action_connect",
        "btn-setup-dacs": "action_setup_dacs",
        "btn-calibrate": "action_calibrate",
        "btn-load-model": "action_load_model",
        "btn-infer": "action_infer_sample",
        "btn-run-100": "action_run_batch",
        "btn-prev": "action_prev_sample",
        "btn-next": "action_next_sample",
    }

    def on_button_pressed(self, event: Button.Pressed) -> None:
        handler = self._BUTTON_ACTIONS.get(event.button.id)
        if handler is not None:
            getattr(self, handler)()

    # ── Actions ──

    @work(thread=True)
    def action_connect(self) -> None:
        self.log_msg(f"\n[bold]Connecting to {self.port}...[/]")
        try:
            self.board = TarskiBoard(self.port)
            version = self.board.get_signature()
            status = self.query_one("#status", StatusPanel)
            status.connected = True
            status.version = version
            self.log_msg(f"[green]Connected! Firmware v{version}[/]")
        except Exception as e:
            self.log_msg(f"[red]Connection failed: {e}[/]")

    @work(thread=True)
    def action_setup_dacs(self) -> None:
        if not self.board:
            self.log_msg("[red]Not connected[/]")
            return
        self.log_msg("\n[bold yellow]DAC Address Programming[/]")
        self.log_msg("Ensure only ONE DAC is connected via jumpers!")
        for i, addr in enumerate([0x60, 0x61, 0x62]):
            self.log_msg(f"\n[bold]Programming DAC #{i+1} to 0x{addr:02X}...[/]")
            self.log_msg("[yellow]>>> Isolate DAC #{} via jumpers, then press Enter in terminal <<<[/]".format(i+1))
            # In TUI mode, we can't easily wait for input, so just try it
            try:
                self.board.program_dac_address(0x60, addr)
                self.log_msg(f"[green]  DAC #{i+1} → 0x{addr:02X} OK[/]")
            except Exception as e:
                self.log_msg(f"[red]  Failed: {e}[/]")

    @work(thread=True)
    def action_calibrate(self) -> None:
        if not self.board:
            self.log_msg("[red]Not connected[/]")
            return
        self.log_msg("\n[bold yellow]Running full calibration...[/]")
        try:
            self.calib = self.board.full_calibration()
            with open('calibration_results.json', 'w') as f:
                json.dump(self.calib, f, indent=2, default=str)
            self.log_msg("[green]Calibration complete! Saved to calibration_results.json[/]")
            scales = self.calib.get('dac_scales', [])
            self.log_msg(f"Per-channel scales: {[f'{s:.4f}' for s in scales]}")
        except Exception as e:
            self.log_msg(f"[red]Calibration failed: {e}[/]")

    @work(thread=True)
    def action_load_model(self) -> None:
        if not self.board:
            self.log_msg("[red]Not connected[/]")
            return
        if not self.checkpoint_path:
            self.log_msg("[red]No checkpoint specified (use --checkpoint)[/]")
            return

        self.log_msg(f"\n[bold]Loading {self.checkpoint_path}...[/]")
        try:
            cp = self.board.load_checkpoint(self.checkpoint_path)
            self.fc1_weights = np.array(cp['weights']['fc1_weight'], dtype=np.float32)
            self.fc2_quantized = cp['quantized']['fc2_weight']
            self.log_msg(f"[green]Model loaded. fc1: {self.fc1_weights.shape}[/]")

            # Load MNIST data
            self._load_mnist()

            # Try loading calibration
            try:
                with open('calibration_results.json') as f:
                    self.calib = json.load(f)
                self.log_msg("[green]Loaded calibration from calibration_results.json[/]")
            except FileNotFoundError:
                self.log_msg("[yellow]No calibration file — using default mapping[/]")
        except Exception as e:
            self.log_msg(f"[red]Failed: {e}[/]")

    def _load_mnist(self) -> None:
        try:
            self.images, self.labels = load_mnist(self.data_dir)
            self.log_msg(f"[green]Loaded {len(self.labels)} MNIST test samples[/]")
        except Exception as e:
            self.log_msg(f"[red]MNIST load failed: {e}[/]")

    def _prepare_sample(self, idx: int) -> np.ndarray | None:
        if self.images is None:
            return None
        img = self.images[idx].astype(np.float32)
        return normalize_mnist(downsample_array_split(img))

    @work(thread=True)
    def action_infer_sample(self) -> None:
        if not self.board or self.fc1_weights is None:
            self.log_msg("[red]Load model first[/]")
            return

        pixels = self._prepare_sample(self.sample_idx)
        if pixels is None:
            self.log_msg("[red]No MNIST data[/]")
            return

        label = int(self.labels[self.sample_idx])

        prediction, spike_counts = self.board.infer(
            pixels, self.fc1_weights, self.calib,
            num_samples=50, interval_us=500,
        )

        correct = prediction == label
        if correct:
            self.total_correct += 1
        self.total_tested += 1

        # Update prediction display
        pred_panel = self.query_one("#prediction", PredictionDisplay)
        pred_panel.prediction = prediction
        pred_panel.true_label = label
        pred_panel.spike_counts = spike_counts

        color = "green" if correct else "red"
        self.log_msg(
            f"[{color}]#{self.sample_idx}: label={label} pred={prediction} "
            f"{'✓' if correct else '✗'}  "
            f"({self.total_correct}/{self.total_tested} = "
            f"{self.total_correct/self.total_tested*100:.1f}%)[/]"
        )

    @work(thread=True)
    def action_run_batch(self) -> None:
        if not self.board or self.fc1_weights is None:
            self.log_msg("[red]Load model first[/]")
            return
        self.log_msg("\n[bold]Running 100 samples...[/]")
        correct = 0
        for i in range(100):
            idx = self.sample_idx + i
            if idx >= len(self.labels):
                break
            pixels = self._prepare_sample(idx)
            label = int(self.labels[idx])
            prediction, _ = self.board.infer(
                pixels, self.fc1_weights, self.calib,
                num_samples=50, interval_us=500,
            )
            if prediction == label:
                correct += 1
            if (i + 1) % 10 == 0:
                self.log_msg(f"  {i+1}/100: {correct}/{i+1} ({correct/(i+1)*100:.1f}%)")

        self.log_msg(f"[bold green]Batch result: {correct}/100 ({correct}%)[/]")

    def _move_sample(self, delta: int) -> None:
        self.sample_idx = max(0, min(self.sample_idx + delta, 9999))
        lbl = self.query_one("#sample-label", Label)
        lbl.update(f"Sample: {self.sample_idx}")

    def action_next_sample(self) -> None:
        self._move_sample(1)

    def action_prev_sample(self) -> None:
        self._move_sample(-1)


def main():
    parser = argparse.ArgumentParser(description='Tarski Board — Interactive TUI')
    parser.add_argument('--port', required=True, help='Serial port')
    parser.add_argument('--checkpoint', help='Gilgamesh checkpoint path')
    parser.add_argument('--data-dir', default='../gilgamesh/data')
    parser.add_argument('--non-interactive', action='store_true',
                        help='Run non-interactively (use with --calibrate/--infer)')
    parser.add_argument('--calibrate', action='store_true')
    parser.add_argument('--infer', action='store_true')
    parser.add_argument('--setup-dacs', action='store_true')
    parser.add_argument('--samples', type=int, default=100)
    args = parser.parse_args()

    if args.non_interactive:
        # Fall back to the CLI interface in tarski_board.py
        import subprocess
        cmd = [sys.executable, 'tarski_board.py', '--port', args.port]
        if args.setup_dacs:
            cmd.append('--setup-dacs')
        if args.calibrate:
            cmd.append('--calibrate')
        if args.infer and args.checkpoint:
            cmd.extend(['--infer', '--checkpoint', args.checkpoint])
        if args.data_dir:
            cmd.extend(['--data-dir', args.data_dir])
        cmd.extend(['--samples', str(args.samples)])
        subprocess.run(cmd)
        return

    app = TarskiTUI(
        port=args.port,
        checkpoint=args.checkpoint,
        data_dir=args.data_dir,
    )
    app.run()


if __name__ == '__main__':
    main()
