//! Compute device selection.
//!
//! The engine runs on the Metal GPU and falls back to the CPU when no Metal
//! device can be created. Candle's `Device` stays inside the crate (ADR-03).

use candle_core::Device;

/// Returns the first Metal device, or the CPU if Metal is unavailable.
///
/// The fallback is logged at `warn`: it keeps the engine usable, but CPU
/// inference is far slower.
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "first caller is weight loading (#8)")
)]
pub(crate) fn select() -> Device {
    match Device::new_metal(0) {
        Ok(device) => {
            tracing::debug!(location = ?device.location(), "using Metal device");
            device
        }
        Err(error) => {
            tracing::warn!(%error, "Metal device unavailable, falling back to CPU");
            Device::Cpu
        }
    }
}

#[cfg(test)]
mod tests {
    mod tier3 {
        use candle_core::{DType, Device, Result, Tensor};

        use crate::device::select;

        /// f32 matmul, Metal vs CPU: `|metal - cpu| <= ATOL + RTOL * |cpu|`.
        /// See the tolerance table in docs/TESTING.md.
        const ATOL: f32 = 1e-4;
        const RTOL: f32 = 1e-4;

        /// Deterministic values in [-1, 1], shaped `[batch, rows, cols]`.
        fn input(len: u32, shape: (usize, usize, usize), scale: f64) -> Result<Tensor> {
            Tensor::arange(0u32, len, &Device::Cpu)?
                .to_dtype(DType::F32)?
                .affine(scale, 0.0)?
                .sin()?
                .reshape(shape)
        }

        #[test]
        fn metal_matmul_matches_cpu() -> Result<()> {
            let metal = select();
            assert!(
                metal.is_metal(),
                "expected Metal, got {:?}",
                metal.location()
            );

            // [2, 64, 256] x [2, 256, 96]: batched, with a realistic inner dimension.
            let a = input(2 * 64 * 256, (2, 64, 256), 0.618_034)?;
            let b = input(2 * 256 * 96, (2, 256, 96), 0.414_214)?;

            let expected = a.matmul(&b)?.flatten_all()?.to_vec1::<f32>()?;
            let product = a.to_device(&metal)?.matmul(&b.to_device(&metal)?)?;
            assert!(product.device().is_metal(), "matmul did not run on Metal");
            let actual = product
                .to_device(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<f32>()?;

            let worst = expected
                .iter()
                .zip(&actual)
                .map(|(e, a)| ((a - e).abs(), e.abs()))
                .fold((0.0_f32, 0.0_f32), |w, d| if d.0 > w.0 { d } else { w });
            assert!(
                expected
                    .iter()
                    .zip(&actual)
                    .all(|(e, a)| (a - e).abs() <= ATOL + RTOL * e.abs()),
                "max |diff| {} at |expected| {}",
                worst.0,
                worst.1
            );
            Ok(())
        }
    }
}
