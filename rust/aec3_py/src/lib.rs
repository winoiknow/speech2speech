// PyO3 wrapper around the pure-Rust WebRTC AEC3 port (crate `aec3`).
// Exposes a tiny class the s2s EchoCanceller drives: feed far-end (render) and
// near-end (capture) as whole 10 ms int16 mono frames; AEC3 aligns the delay
// internally (no manual FIFO needed on our side).
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use aec3::AudioFormat;
use aec3::pipelines::linear::{self, LinearPipeline};

#[pyclass]
struct Aec3 {
    pipeline: LinearPipeline,
    frame: usize,        // samples per 10 ms frame (sample_rate/100)
    cap_f: Vec<f32>,
    ren_f: Vec<f32>,
    out_f: Vec<f32>,
}

#[pymethods]
impl Aec3 {
    #[new]
    fn new(sample_rate: u32) -> PyResult<Self> {
        let fmt = AudioFormat::ten_ms(sample_rate, 1);
        let frame = fmt.sample_count();
        let pipeline = linear::builder(fmt, fmt)
            .build()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("aec3 build: {e:?}")))?;
        Ok(Self { pipeline, frame, cap_f: vec![0.0; frame], ren_f: vec![0.0; frame], out_f: vec![0.0; frame] })
    }

    #[getter]
    fn frame_samples(&self) -> usize { self.frame }

    /// Far-end (render): whole 10 ms frames of int16-LE mono bytes.
    fn process_render(&mut self, pcm: &[u8]) -> PyResult<()> {
        let fb = self.frame * 2;
        let mut off = 0;
        while off + fb <= pcm.len() {
            for (i, b) in pcm[off..off + fb].chunks_exact(2).enumerate() {
                self.ren_f[i] = i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0;
            }
            self.pipeline
                .handle_render_frame(&self.ren_f)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("aec3 render: {e:?}")))?;
            off += fb;
        }
        Ok(())
    }

    /// Near-end (capture): whole 10 ms frames of int16-LE mono → echo-cancelled int16 bytes.
    fn process_capture<'py>(&mut self, py: Python<'py>, pcm: &[u8]) -> PyResult<Bound<'py, PyBytes>> {
        let fb = self.frame * 2;
        let mut out = Vec::with_capacity(pcm.len());
        let mut off = 0;
        while off + fb <= pcm.len() {
            for (i, b) in pcm[off..off + fb].chunks_exact(2).enumerate() {
                self.cap_f[i] = i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0;
            }
            self.pipeline
                .process_capture_frame(&self.cap_f, &mut self.out_f)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("aec3 capture: {e:?}")))?;
            for &s in &self.out_f {
                let v = (s.clamp(-1.0, 1.0) * 32767.0).round() as i16;
                out.extend_from_slice(&v.to_le_bytes());
            }
            off += fb;
        }
        Ok(PyBytes::new_bound(py, &out))
    }
}

#[pymodule]
fn aec3_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Aec3>()?;
    Ok(())
}
