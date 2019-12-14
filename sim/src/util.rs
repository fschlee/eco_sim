#[inline]
pub(crate) fn clip(val: f32, min: f32, max: f32) -> f32 {
    max.min(min.max(val))
}

#[inline]
pub(crate) fn f32_cmp(f1: &f32, f2: &f32) -> std::cmp::Ordering {
    f32::partial_cmp(f1,f2).expect("NaN")
}