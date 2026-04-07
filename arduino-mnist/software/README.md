# Arduino MNIST Host Tooling (`software/`)

This directory contains the host-side tools used to communicate with the Arduino MNIST firmware over serial, send quantized test images, trigger inference, and collect results.

This is a **subproject** of the parent `arduino-mnist` repository. It is designed to work with the firmware/runtime in the repository root (`src/`, `include/`) and assumes the firmware implements the serial protocol described below.

## What Is In This Folder

- `arduino-interface` (built from `src/*.cpp`): main CLI tool for serial communication.
- `png_to_bin.py`: converts images into model-ready binary samples.
- `bin_to_png.py`: converts binary samples back into 6x6 PNG images.
- `summary.py`: computes average inference time and accuracy from `results.csv`.
- `run_all.sh`: convenience wrapper for batch prediction + summary.
- `data/`: example `.bin` dataset (labeled 37-byte samples).
- `test_data/test_6x6.bin`: minimal 36-byte sample (unlabeled).

## Supported Platforms

### `arduino-interface`

- Implemented using POSIX serial APIs (`termios`, `ioctl`, `unistd`, `fcntl`).
- Tested build path is Linux (`make linux`).
- macOS may compile with adjustments but is not explicitly validated here.
- Native Windows is **not** supported by current source (`src/serial.cpp` is POSIX-specific).

### Python utilities

- Python 3.10+ recommended.
- Requires `Pillow` (`pip install pillow`).

## Build

From `software/`:

```bash
make linux
```

This produces:

- `./arduino-interface`

Notes:

- `make` default target builds both `linux` and `win`. The `win` target requires a MinGW cross-compiler (`x86_64-w64-mingw32-gcc-win32`) and may fail if that toolchain is missing.
- A local build check of `make linux` succeeds in this directory.

## CLI Tool: `arduino-interface`

### Usage

```bash
./arduino-interface <port> <-l|-r|-p> [options]
```

- `<port>`: serial device path (for example `/dev/ttyUSB0` or `/dev/ttyACM0`)
- Modes:
  - `-l`: load binary sample into device memory
  - `-r`: run inference on already-loaded sample
  - `-p`: load sample then run inference
- Options:
  - `-i <path>`: input file path (or input directory with `-p -b`)
  - `-o <path>`: output file path (default: stdout)
  - `-b`: batch mode (valid only with `-p`)
  - `-f`: repeat mode for fixed duration (valid only with `-l` or `-r`)
  - `-t <seconds>`: duration for repeat mode (must be used together with `-f`)

### Argument Validation Rules

- Exactly one mode must be provided (`-l`, `-r`, or `-p`).
- `-l` requires `-i <file>` and disallows `-b`.
- `-r` disallows `-b` and does not require `-i`.
- `-p` requires `-i`:
  - single-file mode: `-i` is a `.bin` file
  - batch mode (`-b`): `-i` is a directory; all `.bin` files are processed
- `-f` requires `-t <seconds>` and vice versa.
- `-f/-t` are not allowed in `-p` mode.

### Common Examples

Load one sample:

```bash
./arduino-interface /dev/ttyUSB0 -l -i test_data/test_6x6.bin
```

Run inference on previously loaded sample and print device message to stdout:

```bash
./arduino-interface /dev/ttyUSB0 -r
```

Load and infer one labeled sample, save output:

```bash
./arduino-interface /dev/ttyUSB0 -p -i data/digit_7_0.bin -o results.csv
```

Batch prediction over all `.bin` files in `data/`:

```bash
./arduino-interface /dev/ttyUSB0 -p -b -i data -o results.csv
```

Repeated load for 10 seconds (throughput/stability testing):

```bash
./arduino-interface /dev/ttyUSB0 -l -i test_data/test_6x6.bin -f -t 10
```

Repeated inference for 30 seconds:

```bash
./arduino-interface /dev/ttyUSB0 -r -f -t 30 -o run_log.txt
```

## Data Contract

### Input binary format

The firmware-facing sample payload is **36 bytes** (`Arduino::DATA_SIZE`), interpreted as:

- 6x6 grayscale image
- signed int8 per pixel
- expected value range `[-128, 127]`

For prediction mode (`-p`), this tool expects at least **37 bytes**:

- bytes `[0..35]`: input sample (sent to device)
- byte `[36]`: expected class label (`0..9`) used only for host-side correctness annotation

Important behavior:

- If a file is larger than 36/37 bytes, only the first 36 bytes are sent to the board.
- For `-p`, the label is always read from byte index 36.
- There is no host-side label range validation before output annotation.

### Output format

`arduino-interface` writes raw message payload from firmware to output sink (`stdout` or file via `-o`).

In `-p` mode, it appends host-side fields:

```text
,<expected_label>,<true|false>
```

So each line is typically:

```text
<firmware_message>,<expected_label>,<true|false>
```

`summary.py` expects `results.csv` rows where:

- column 0 is a numeric prediction time (float)
- column 3 is `true`/`false` correctness

If your firmware message schema differs, update `summary.py` accordingly.

## Image Conversion Utilities

### `png_to_bin.py`

Converts images (`png/jpg/jpeg/bmp/gif`) to 6x6 int8 binary input.

```bash
python3 png_to_bin.py <input_image_or_dir> <output_file_or_dir> [--batch] [--header] [--preview]
```

Behavior:

- Converts image to grayscale.
- If image is exactly 6x6, uses pixels directly.
- If image is square and dimensions are multiples of 6, performs block averaging.
- Otherwise resizes to 6x6 using bicubic interpolation.
- Quantization: `int8_value = pixel - 128`, clamped to `[-128, 127]`.
- If filename matches `digit_<N>_*.png` (`N` in `0..9`), appends label byte to binary output.

Examples:

```bash
python3 png_to_bin.py images/png/digit_3_0.png data/digit_3_0.bin
python3 png_to_bin.py images/png data --batch
python3 png_to_bin.py images/png/digit_3_0.png out.h --header
```

### `bin_to_png.py`

Converts `.bin` back to 6x6 grayscale PNG.

```bash
python3 bin_to_png.py <input_bin_or_dir> <output_png_or_dir> [--batch]
```

Behavior and constraint:

- Reads exactly 37 bytes per file (36 pixel bytes + 1 extra byte).
- Converts int8 pixels back with `pixel = int8 + 128` (clamped to `[0, 255]`).
- Creates a 6x6 grayscale PNG.

Because it requires 37 bytes, unlabeled 36-byte files (for example `test_data/test_6x6.bin`) will fail in this script unless adjusted.

## Serial Protocol Expectations

The host tool expects firmware support for the following control bytes:

- `PORT_SIG (0x05)`: request firmware version
- `PORT_LOAD ('L')`: load sample request
- `PORT_INFER ('I')`: run inference request
- `PORT_ACK (0x06)`: acknowledge
- `PORT_NAK (0x15)`: negative acknowledge
- `PORT_MSG (0x02)`: begin printable message
- `PORT_MSG_END (0x03)`: end printable message
- `PORT_TRN_END (0x04)`: end transmission

Session behavior:

1. On startup, host requests firmware signature/version.
2. For load mode, host sends 36-byte sample and verifies echoed payload.
3. For inference mode, host expects prediction + message framing + EOT.

If firmware protocol diverges, host operations may timeout or fail handshake checks.

## Known Caveats and Compatibility Notes

- Serial config is fixed at **9600 baud** in current CLI implementation.
- Serial open includes a fixed ~2 second delay to allow port/device readiness.
- `run_all.sh` is hardcoded to `/dev/ttyUSB0`; adjust for your system.
- Batch mode iterates `.bin` files by directory iterator order; do not assume strict lexical ordering across platforms/filesystems.
- Existing low-level I/O paths currently assume successful full read/write calls; partial I/O handling is limited.
- `arduino-interface` only checks that input file size is at least required minimum (36 or 37), not exact size.
- `summary.py` assumes `results.csv` exists in current working directory and follows expected column layout.

## Typical End-to-End Workflow

From `software/`:

1. Build host tool:

```bash
make linux
```

2. (Optional) Generate `.bin` samples from PNG inputs:

```bash
python3 png_to_bin.py images/png data --batch
```

3. Run batch prediction and save CSV output:

```bash
./arduino-interface /dev/ttyUSB0 -p -b -i ./data -o results.csv
```

4. Summarize performance/accuracy:

```bash
python3 summary.py
```

Or use wrapper:

```bash
./run_all.sh
```

## Troubleshooting

- `ERROR: Unable to open serial port path.`
  - Check correct device path and permissions (`dialout`/`uucp` group as applicable).
- `ERROR: Unable to set the requested configuration.`
  - Port might be busy or not a compatible TTY device.
- `ERROR: Device did not respond.`
  - Firmware not running, wrong baud/protocol, or incorrect port.
- `ERROR: Binary image file is too small.`
  - Provide at least 36 bytes (`-l`) or 37 bytes (`-p`).

## Repository Context

This folder is intended to be documented and versioned as the host-tooling component of the larger `arduino-mnist` project. Keep firmware-specific assumptions (protocol bytes, message format, expected sample layout) aligned with parent-repo firmware changes.