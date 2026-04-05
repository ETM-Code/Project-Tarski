This folder contains converted training data.

Files:
- train_7x7.csv : CSV with header "label,p00,p01,...,p66", values are u8 in 0..255
- train_7x7.bin : Binary with header + interleaved pixels and labels as described below
- *_preview.png : Visual grid of the first samples for quick inspection

Binary format (little endian):
  offset  size  type    meaning
  0       4     bytes   magic = "M7x7"
  4       4     u32     num_samples
  8       1     u8      rows = 7
  9       1     u8      cols = 7
  10      2     u16     reserved = 0
  12      ...           for each sample: 49 bytes u8 pixels (row-major), then 1 byte u8 label

Minimal Rust reader:

use std::fs::File;
use std::io::{Read, BufReader};
use byteorder::{LittleEndian, ReadBytesExt};

fn read_m7x7(path: &str) -> std::io::Result<(usize, Vec<[u8; 49]>, Vec<u8>)> {
    let f = File::open(path)?;
    let mut r = BufReader::new(f);

    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    assert_eq!(&magic, b"M7x7");

    let n = r.read_u32::<LittleEndian>()? as usize;
    let rows = r.read_u8()? as usize; assert_eq!(rows, 7);
    let cols = r.read_u8()? as usize; assert_eq!(cols, 7);
    let _reserved = r.read_u16::<LittleEndian>()?;

    let mut pixels = vec![[0u8; 49]; n];
    let mut labels = vec![0u8; n];

    for i in 0..n {
        r.read_exact(&mut pixels[i])?;
        r.read_exact(&mut labels[i..i+1])?;
    }

    Ok((n, pixels, labels))
}
