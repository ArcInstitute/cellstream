//! Lossless integer-routing preflight: validate that f4/f8/i4/i8 source values
//! fit uint32 losslessly, reproducing narrow.lossless_cast's fail-loud guarantee.
//! No copy.

/// Why an integer-routing scan failed — carried back to Python to build a
/// LossyCastError.
#[derive(Debug)]
pub struct NarrowError {
    pub min: f64,
    pub max: f64,
    pub reason: &'static str,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnDiskDtype {
    U16,
    U32,
    F32,
    F64,
}

#[allow(dead_code)]
impl OnDiskDtype {
    pub fn name(self) -> &'static str {
        match self {
            Self::U16 => "uint16",
            Self::U32 => "uint32",
            Self::F32 => "float32",
            Self::F64 => "float64",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "uint16" => Some(Self::U16),
            "uint32" => Some(Self::U32),
            "float32" => Some(Self::F32),
            "float64" => Some(Self::F64),
            _ => None,
        }
    }

    pub fn itemsize(self) -> usize {
        match self {
            Self::U16 => 2,
            Self::U32 | Self::F32 => 4,
            Self::F64 => 8,
        }
    }
}

#[allow(dead_code)]
#[derive(Clone, Copy)]
pub enum SrcKind {
    U16,
    U32,
    F32,
    F64,
}

/// Source element types that may be routed to integer uint32 storage.
pub trait ToIntScan: Copy {
    /// True iff this value round-trips through uint32 losslessly.
    fn fits_u32(self) -> bool;
    /// True iff this value is negative (for the signed-to-unsigned sign-loss rule).
    fn is_negative(self) -> bool;
    fn as_f64(self) -> f64;
    fn as_u64(self) -> u64;
}

macro_rules! impl_signed_int_scan {
    ($t:ty, $max:expr) => {
        impl ToIntScan for $t {
            #[inline]
            fn fits_u32(self) -> bool {
                (0..=$max).contains(&self)
            }
            #[inline]
            fn is_negative(self) -> bool {
                self < 0
            }
            #[inline]
            fn as_f64(self) -> f64 {
                self as f64
            }
            #[inline]
            fn as_u64(self) -> u64 {
                self as u64
            }
        }
    };
}
impl_signed_int_scan!(i32, i32::MAX);
impl_signed_int_scan!(i64, u32::MAX as i64);

impl ToIntScan for u16 {
    #[inline]
    fn fits_u32(self) -> bool {
        true
    }
    #[inline]
    fn is_negative(self) -> bool {
        false
    }
    #[inline]
    fn as_f64(self) -> f64 {
        self as f64
    }
    #[inline]
    fn as_u64(self) -> u64 {
        self as u64
    }
}

impl ToIntScan for u32 {
    #[inline]
    fn fits_u32(self) -> bool {
        true
    }
    #[inline]
    fn is_negative(self) -> bool {
        false
    }
    #[inline]
    fn as_f64(self) -> f64 {
        self as f64
    }
    #[inline]
    fn as_u64(self) -> u64 {
        self as u64
    }
}

macro_rules! impl_float_scan {
    ($t:ty) => {
        impl ToIntScan for $t {
            #[inline]
            fn fits_u32(self) -> bool {
                // EXCLUSIVE upper bound, and the literal is 2^32 rather than u32::MAX.
                // u32::MAX (4294967295) is NOT representable in f32 -- it rounds UP to
                // 4294967296.0 -- so `..=4294967295.0` silently admitted 2^32, which does
                // not fit u32, and encode then saturated it to 4294967295. 2^32 IS exactly
                // representable in both f32 and f64, so the exclusive bound is exact for
                // both instantiations, and fract()==0.0 already excludes the open interval
                // below it. (#253)
                self.is_finite() && self.fract() == 0.0 && (0.0..4294967296.0).contains(&self)
            }
            #[inline]
            fn is_negative(self) -> bool {
                self < 0.0
            }
            #[inline]
            fn as_f64(self) -> f64 {
                self as f64
            }
            #[inline]
            fn as_u64(self) -> u64 {
                self as u64
            }
        }
    };
}
impl_float_scan!(f32);
impl_float_scan!(f64);

/// Return the maximum integer value when every element can route to uint32.
/// Returns None for non-finite, non-integral, negative, or >uint32 values.
pub fn scan_int_max<T: ToIntScan + Sync>(data: &[T], n_threads: usize) -> Option<u64> {
    if data.is_empty() {
        return Some(0);
    }
    let chunk = data.len().div_ceil(n_threads.max(1));
    let all_fit = std::sync::atomic::AtomicBool::new(true);
    std::thread::scope(|s| {
        let handles: Vec<_> = data
            .chunks(chunk)
            .map(|c| {
                let all_fit = &all_fit;
                s.spawn(move || {
                    let mut max = 0u64;
                    for &v in c {
                        if !all_fit.load(std::sync::atomic::Ordering::Relaxed) {
                            return None;
                        }
                        if !v.fits_u32() {
                            all_fit.store(false, std::sync::atomic::Ordering::Relaxed);
                            return None;
                        }
                        max = max.max(v.as_u64());
                    }
                    Some(max)
                })
            })
            .collect();
        let mut max = 0u64;
        for h in handles {
            max = max.max(h.join().unwrap()?);
        }
        Some(max)
    })
}

pub fn min_uint_name(max: u64) -> &'static str {
    if max <= u8::MAX as u64 {
        "uint8"
    } else if max <= u16::MAX as u64 {
        "uint16"
    } else {
        "uint32"
    }
}

/// (on_disk dtype, min dtype name). Integer-routable data -> on_disk U32; min
/// name = narrowest of "uint8"/"uint16"/"uint32" by max. Otherwise
/// (non-finite, non-integral, negative, or > u32::MAX) -> native float for both;
/// min name = on_disk.name().
pub fn decide_data_dtypes<T: ToIntScan + Sync>(
    data: &[T],
    src: SrcKind,
    n_threads: usize,
) -> (OnDiskDtype, &'static str) {
    match scan_int_max(data, n_threads) {
        Some(max) => (OnDiskDtype::U32, min_uint_name(max)),
        None => {
            let native = match src {
                SrcKind::F32 => OnDiskDtype::F32,
                SrcKind::F64 => OnDiskDtype::F64,
                SrcKind::U16 => OnDiskDtype::U32,
                SrcKind::U32 => OnDiskDtype::U32,
            };
            (native, native.name())
        }
    }
}

/// Scan a slice; Ok(()) if every element routes to uint32 losslessly, else
/// NarrowError with the min/max over the slice (matching lossless_cast's
/// diagnostic shape).
pub fn scan_u32<T: ToIntScan>(arr: &[T]) -> Result<(), NarrowError> {
    if arr.is_empty() {
        return Ok(());
    }
    let mut min = arr[0].as_f64();
    let mut max = arr[0].as_f64();
    let mut any_negative = false;
    let mut all_fit = true;
    for &v in arr {
        let f = v.as_f64();
        // Propagate NaN into min/max to match numpy's .min()/.max() diagnostics.
        if f.is_nan() {
            min = f64::NAN;
            max = f64::NAN;
        } else if !min.is_nan() {
            if f < min {
                min = f;
            }
            if f > max {
                max = f;
            }
        }
        if v.is_negative() {
            any_negative = true;
        }
        if !v.fits_u32() {
            all_fit = false;
        }
    }
    if any_negative {
        return Err(NarrowError {
            min,
            max,
            reason: "negative values cannot be stored in an unsigned target",
        });
    }
    if !all_fit {
        return Err(NarrowError {
            min,
            max,
            reason: "value out of range or non-integral for uint32",
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn passes_valid_u32_data() {
        assert!(scan_u32::<f32>(&[0.0, 1.0, 65535.0, 70000.0]).is_ok());
        assert!(scan_u32::<i32>(&[0, 7, i32::MAX]).is_ok());
        assert!(scan_u32::<i64>(&[]).is_ok());
    }

    #[test]
    fn rejects_negative_int() {
        let e = scan_u32::<i32>(&[1, -3, 5]).unwrap_err();
        assert!(e.reason.contains("negative"));
        assert_eq!(e.min, -3.0);
    }

    #[test]
    fn rejects_out_of_range() {
        let e = scan_u32::<f64>(&[1.0, 4294967296.0]).unwrap_err();
        assert!(e.reason.contains("range"));
        assert_eq!(e.max, 4294967296.0);
    }

    #[test]
    fn rejects_non_integral_and_nan() {
        assert!(scan_u32::<f32>(&[1.0, 2.5]).is_err());
        assert!(scan_u32::<f32>(&[f32::NAN]).is_err());
    }

    #[test]
    fn scan_int_max_returns_max_or_none() {
        assert_eq!(scan_int_max::<f32>(&[1.0, 2.5], 1), None);
        assert_eq!(scan_int_max::<i64>(&[0, 70000], 1), Some(70000));
        assert_eq!(scan_int_max::<u16>(&[], 4), Some(0));
    }

    #[test]
    fn data_dtypes_route_integers_to_u32_with_min_name() {
        use SrcKind::*;
        assert_eq!(OnDiskDtype::U16.name(), "uint16");
        assert_eq!(OnDiskDtype::U32.itemsize(), 4);
        assert_eq!(
            decide_data_dtypes::<u16>(&[1, 2, 3], U16, 2),
            (OnDiskDtype::U32, "uint8")
        );
        assert_eq!(
            decide_data_dtypes::<f32>(&[1.0, 70000.0], F32, 2),
            (OnDiskDtype::U32, "uint32")
        );
        assert_eq!(
            decide_data_dtypes::<f64>(&[1.0, 65535.0], F64, 2),
            (OnDiskDtype::U32, "uint16")
        );
    }

    #[test]
    fn from_name_roundtrips() {
        for dt in [
            OnDiskDtype::U16,
            OnDiskDtype::U32,
            OnDiskDtype::F32,
            OnDiskDtype::F64,
        ] {
            assert_eq!(OnDiskDtype::from_name(dt.name()), Some(dt));
        }
        assert_eq!(OnDiskDtype::from_name("uint8"), None);
    }

    #[test]
    fn data_dtypes_preserve_native_float_when_not_integer_routable() {
        use SrcKind::*;
        assert_eq!(
            decide_data_dtypes::<f32>(&[1.0, 2.5], F32, 2),
            (OnDiskDtype::F32, "float32")
        );
        assert_eq!(
            decide_data_dtypes::<f64>(&[-1.0, 2.0], F64, 2),
            (OnDiskDtype::F64, "float64")
        );
        assert_eq!(
            decide_data_dtypes::<f32>(&[f32::NAN], F32, 2),
            (OnDiskDtype::F32, "float32")
        );
        assert_eq!(
            decide_data_dtypes::<u32>(&[1, 70000], U32, 2),
            (OnDiskDtype::U32, "uint32")
        );
    }

    #[test]
    fn rejects_2_pow_32_for_f32_range_literal() {
        // 4294967295.0 (u32::MAX) is NOT representable in f32 -- it rounds UP to 4294967296.0
        // (2^32). So an INCLUSIVE bound of ..=4294967295.0 silently admitted 2^32, which does
        // not fit u32, and encode then saturated/wrapped it. (#253)
        let two_pow_32 = 4294967296.0f32;
        // The heart of the bug: the OLD bound's literal is indistinguishable from 2^32 in
        // f32, so `..=u32::MAX` was really `..=2^32`.
        assert_eq!(4294967295.0f32, 4294967296.0f32);
        assert_eq!(two_pow_32 as f64, 4294967296.0);
        let e = scan_u32::<f32>(&[1.0, two_pow_32]).unwrap_err();
        assert!(e.reason.contains("range"), "reason was {}", e.reason);

        // The largest f32 that IS a valid u32 must still pass: 2^32 - 2^8, exactly representable.
        assert!(scan_u32::<f32>(&[4294967040.0]).is_ok());
        // f64 is unaffected: u32::MAX is exactly representable and must still pass.
        assert!(scan_u32::<f64>(&[4294967295.0]).is_ok());
        assert!(scan_u32::<f64>(&[4294967296.0]).is_err());
    }
}
