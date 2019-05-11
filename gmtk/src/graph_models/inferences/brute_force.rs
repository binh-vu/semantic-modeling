use graph_models::traits::*;
use tensors::*;
use fnv::FnvHashMap;
use graph_models::inferences::Inference;
use std::collections::HashSet;
use std::f64;
use graph_models::utils::misc::iter_assignment;
use num_traits::FromPrimitive;

pub struct BruteForce<'a: 'a2, 'a2, V: 'a + Variable, T: 'static + TensorType=TDouble> {
    variables: &'a [V],
    factors: &'a2 [Box<Factor<'a, V, T> + 'a>],
    log_z: Option<f64>
}

impl<'a: 'a2, 'a2, V: 'a + Variable, T: 'static + TensorType> BruteForce<'a, 'a2, V, T> {
    pub fn new(variables: &'a [V], factors: &'a2 [Box<Factor<'a, V, T> + 'a>]) -> BruteForce<'a, 'a2, V, T> {
        BruteForce {
            variables,
            factors,
            log_z: None
        }
    }

    pub fn all_map(&self) -> Vec<FnvHashMap<usize, V::Value>> {
        let mut max_score = -f64::INFINITY;
        let mut map_sols = Vec::new();

        iter_assignment(&self.variables.iter().collect::<Vec<&V>>(), |current_idx, assignment| {
            let score: f64 = self.factors.iter()
                .map(|f| f.score_assignment(assignment))
                .sum();

            if score > max_score {
                map_sols = vec![current_idx.clone()];
                max_score = score;
            } else if score == max_score {
                map_sols.push(current_idx.clone());
            }
        });

        map_sols.iter().map(|current_idx| {
            current_idx.iter().enumerate()
                .map(|(i, &idx)| (self.variables[i].get_id(), self.variables[i].get_domain().get_value(idx as usize)))
                .collect::<FnvHashMap<usize, V::Value>>()
        }).collect()
    }

    pub fn marginal_log_prob(&self, vars: &[&V]) -> DenseTensor<T> {
        let _factor_var_ids: HashSet<usize> = vars.iter().map(|v| v.get_id()).collect();
        let mut scores = DenseTensor::<T>::create(
            &vars.iter().map(|v| v.get_domain_size()).collect::<Vec<i64>>());
        let remained_variables: Vec<&V> = self.variables.iter()
            .filter(|v| !_factor_var_ids.contains(&v.get_id()))
            .collect();

        iter_assignment(vars, |current_index, assignment| {
            let mut score = Vec::new();
            if remained_variables.len() > 0 {
                iter_assignment(&remained_variables, |_temp, r_ass| {
                    let mut new_assignment: FnvHashMap<usize, V::Value> = Default::default();
                    for (&k, v) in r_ass.iter() {
                        new_assignment.insert(k, v.clone());
                    }
                    for (&k, v) in assignment.iter() {
                        new_assignment.insert(k, v.clone());
                    }
                    // it's weird that rust compiler doesn't support this
//                    r_ass.extend(assignment.iter());
                    score.push(self.factors.iter()
                        .map(|f| T::PrimitiveType::from_f64(f.score_assignment(&new_assignment)).unwrap())
                        .sum());
                });
            } else {
                score.push(self.factors.iter()
                    .map(|f| T::PrimitiveType::from_f64(f.score_assignment(assignment)).unwrap())
                    .sum());
            }

            scores.assign(current_index, DenseTensor::<T>::borrow_from_array(&score).log_sum_exp() - self.log_z.unwrap());
        });

        scores
    }
}

impl<'a: 'a2, 'a2, V: 'a + Variable, T: 'static + TensorType> Inference<'a, V, T> for BruteForce<'a, 'a2, V, T> {
    fn reset_value(&mut self) {}

    fn infer(&mut self) {
        let mut scores: Vec<<T as TensorType>::PrimitiveType> = Vec::with_capacity(self.variables.len());
        let vars: Vec<&V> = self.variables.iter().collect();
        iter_assignment(&vars, |_, assignment| {
            let score = self.factors.iter()
                .map(|f| T::PrimitiveType::from_f64(f.score_assignment(assignment)).unwrap())
                .sum();
            scores.push(score);
        });

        self.log_z = Some(DenseTensor::<T>::borrow_from_array(&scores).log_sum_exp().get_f64());
    }

    fn map(&self) -> FnvHashMap<usize, <V as Variable>::Value> {
        unimplemented!()
    }

    fn log_z(&self) -> f64 {
        self.log_z.unwrap()
    }

    fn log_prob_var(&self, _var: &V) -> DenseTensor<T> {
        unimplemented!()
    }

    fn log_prob_factor(&self, factor: &Factor<'a, V, T>) -> DenseTensor<T> {
        self.marginal_log_prob(factor.get_variables())
    }
}

