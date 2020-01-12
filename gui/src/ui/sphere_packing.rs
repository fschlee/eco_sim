#[derive(Debug, Clone)]
pub struct SpherePacking {
    min: [f64; 3],
    max: [f64; 3],
    r: f64,
    current: [f64; 3],
    current_offset: f64,
}

impl SpherePacking {
    pub fn fit_n_in_box(n: u64, min: [f64; 3], max: [f64; 3]) -> Self {
        let range = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
        let mut approx_r = (range[0] * range[1] + range[2]).powf(1.0 / 3.0);

        let epsilon = 1e-5 * approx_r;
        let mut min_r = epsilon;
        let mut max_r = 2.0 * approx_r;
        assert!(fit(range, min_r) >= n);
        while max_r - min_r > epsilon {
            if fit(range, approx_r) >= n {
                min_r = approx_r;
            } else {
                max_r = approx_r;
            }
            approx_r = 0.5 * (min_r + max_r);
        }
        let mut current = min;
        current[0] -= 2.0 * min_r;
        Self {
            min,
            max,
            r: min_r * 2.0,
            current,
            current_offset: 0.0,
        }
    }
}
impl Iterator for SpherePacking {
    type Item = [f64; 3];

    fn next(&mut self) -> Option<[f64; 3]> {
        if self.current[0] + self.r <= self.max[0] {
            self.current[0] += self.r;
            return Some(self.current);
        }
        if self.current[1] + self.r <= self.max[1] {
            self.current[0] = self.min[0] + self.current_offset;
            self.current[1] += self.r;
            return Some(self.current);
        }
        if self.current[2] + pitch(self.r) <= self.max[2] {
            self.current_offset = if self.current_offset > 0.0 {
                0.0
            } else {
                0.5 * self.r
            };
            self.current[2] += pitch(self.r);
            self.current[0] = self.min[0] + self.current_offset;
            self.current[1] = self.min[1] + self.current_offset;
            return Some(self.current);
        }
        None
    }
}

pub fn pitch(r: f64) -> f64 {
    6.0f64.sqrt() * r / 3.0
}

fn fit(range: [f64; 3], r: f64) -> u64 {
    let p = pitch(r);
    (1 + (range[0] / (2.0 * r)).floor() as u64)
        * (1 + (range[1] / (2.0 * r)).floor() as u64)
        * (1 + (range[2] / (4.0 * p)) as u64)
        + ((range[0] - r) / (2.0 * r)).ceil() as u64
            * ((range[1] - r) / (2.0 * r)).ceil() as u64
            * ((range[2] - 2.0 * p) / (4.0 * p)).ceil() as u64
}
