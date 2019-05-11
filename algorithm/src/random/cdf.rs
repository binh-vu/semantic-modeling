/// Represent Cumulative Distribution Function
#[derive(Clone)]
pub struct CDF {
    cdf: Vec<f64>,
    value_index: Vec<usize>
}

impl CDF {
    pub fn new(weights: &[f64]) -> CDF {
        let mut cdf = Vec::with_capacity(weights.len() + 1);
        let mut value_index = (0..weights.len()).collect::<Vec<_>>();

        cdf.push(0.0);
        for i in 0..weights.len() {
            let val = weights[i] + cdf[i];
            cdf.push(val);
            assert!(weights[i] > 0.0, "Invalid CDF, all value should be greater or equal zero.")
        }

        CDF {
            cdf, value_index
        }
    }

    #[inline]
    pub fn draw_sample(&self, prob: &f64) -> usize {
        let score = self.cdf.last().unwrap() * prob;
        // doing binary search to find the match
        let val_idx = match self.cdf.binary_search_by(|x| x.partial_cmp(&score).unwrap()) {
            Ok(idx) => idx,
            Err(idx) => idx - 1
        };

        self.value_index[val_idx]
    }

    #[inline]
    pub fn remove_sample(&mut self, prob: &f64) -> usize {
        let score = self.cdf.last().unwrap() * prob;
        // doing binary search to find the match
        let val_idx = match self.cdf.binary_search_by(|x| x.partial_cmp(&score).unwrap()) {
            Ok(idx) => idx,
            Err(idx) => idx - 1
        };

        // TODO: can do better by swap_remove, cdf need to update according to the change
        let value = self.value_index.remove(val_idx);
        let item_prob = self.cdf[val_idx+1] - self.cdf[val_idx];
        for i in val_idx..self.cdf.len() {
            self.cdf[i] -= item_prob;
        }
        self.cdf.remove(val_idx);

        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_cdf() {
        let cdf = CDF::new(&[2.0, 3.0, 5.0]);
        assert_eq!(cdf.cdf, vec![0.0, 2.0, 5.0, 10.0]);
        assert_eq!(cdf.value_index, vec![0, 1, 2]);
    }

    #[test]
    fn test_draw_sample() {
        let cdf = CDF::new(&[2.0, 3.0, 5.0]);

        for p in [0.0, 0.1, 0.1999].iter() {
            assert_eq!(cdf.draw_sample(p), 0);
        }
        for p in [0.2, 0.35, 0.4999].iter() {
            assert_eq!(cdf.draw_sample(p), 1);
        }
        for p in [0.5, 0.51, 0.7999, 0.999].iter() {
            assert_eq!(cdf.draw_sample(p), 2);
        }
    }

    #[test]
    fn test_remove_sample() {
        for p in [0.0, 0.1, 0.1999].iter() {
            let mut cdf = CDF::new(&[2.0, 3.0, 5.0]);
            assert_eq!(cdf.remove_sample(p), 0);
            assert_eq!(cdf.cdf, vec![0.0, 3.0, 8.0]);
            assert_eq!(cdf.value_index, vec![1, 2]);

            for p in [0.0, 0.27, 0.2726].iter() {
                let mut cdf2 = cdf.clone();
                assert_eq!(cdf2.remove_sample(p), 1);
                assert_eq!(cdf2.cdf, vec![0.0, 5.0]);
                assert_eq!(cdf2.value_index, vec![2]);
            }

            cdf.remove_sample(&0.1);
            for p in [0.0, 0.55, 0.999].iter() {
                let mut cdf2 = cdf.clone();
                assert_eq!(cdf2.remove_sample(p), 2);
                assert_eq!(cdf2.cdf, vec![0.0]);
                assert_eq!(cdf2.value_index.len(), 0);
            }
        }

        for p in [0.2, 0.35, 0.4999].iter() {
            let mut cdf = CDF::new(&[2.0, 3.0, 5.0]);
            assert_eq!(cdf.remove_sample(p), 1);
            assert_eq!(cdf.cdf, vec![0.0, 2.0, 7.0]);
            assert_eq!(cdf.value_index, vec![0, 2]);

            for p in [0.0, 0.11, 0.285].iter() {
                let mut cdf2 = cdf.clone();
                assert_eq!(cdf2.remove_sample(p), 0);
                assert_eq!(cdf2.cdf, vec![0.0, 5.0]);
                assert_eq!(cdf2.value_index, vec![2]);
            }

            for p in [0.286, 0.88, 0.99].iter() {
                let mut cdf2 = cdf.clone();
                assert_eq!(cdf2.remove_sample(p), 2);
                assert_eq!(cdf2.cdf, vec![0.0, 2.0]);
                assert_eq!(cdf2.value_index, vec![0]);
            }
        }
    }
}