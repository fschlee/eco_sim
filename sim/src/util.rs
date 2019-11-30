pub fn clip(val: f32, min: f32, max: f32) -> f32 {
    max.min(min.max(val))
}