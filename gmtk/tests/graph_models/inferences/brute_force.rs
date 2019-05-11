use gmtk::graph_models::*;
use gmtk::tensors::*;
use fnv::FnvHashMap;

pub struct CustomIntFactor<'a> {
    variables: Vec<&'a IntVariable>,
    weights: Weights<TDouble>,
    vars_dims: Vec<i64>,
    features_tensor: DenseTensor<TDouble>
}

impl<'a> CustomIntFactor<'a> {
    pub fn new(variables: Vec<&'a IntVariable>, weights: Weights<TDouble>) -> CustomIntFactor<'a> {
        let vars_dims = variables.iter().map(|v| v.get_domain_size()).collect();
        let mut factor = CustomIntFactor {
            variables, weights, vars_dims,
            features_tensor: DenseTensor::<TDouble>::default()
        };

        factor.update_features_tensor();
        factor
    }

    fn update_features_tensor(&mut self) {
        let mut features_tensor = DenseTensor::<TDouble>::create(
            &[
                self.vars_dims.iter().fold(1, |a, b| a * b),
                self.weights.get_value().numel()
            ]
        );

        iter_values(&self.variables, |idx, _unravel_idx, values| {
            features_tensor.assign(idx, values.iter().map(|&v| v as f64).collect::<Vec<f64>>());
        });

        self.features_tensor = features_tensor;
    }
}



impl<'a> Factor<'a, IntVariable, TDouble> for CustomIntFactor<'a> {
    fn get_variables(&self) -> &[&'a IntVariable] {
        &self.variables
    }

    fn score_assignment(&self, assignment: &FnvHashMap<usize, i32>) -> f64 {
        self.impl_score_assignment(assignment)
    }

    fn get_scores_tensor(&self) -> DenseTensor<TDouble> {
        self.impl_get_scores_tensor()
    }

    fn compute_gradients(&self, target_assignment: &FnvHashMap<usize, i32>, inference: &Inference<'a, IntVariable, TDouble>) -> Vec<(i64, DenseTensor<TDouble>)> {
        self.impl_compute_gradients(target_assignment, inference)
    }

    fn touch(&self, var: &IntVariable) -> bool {
        for v in &self.variables {
            if v.get_id() == var.get_id() {
                return true;
            }
        }

        return false;
    }
}

impl<'a> DotTensor1Factor<'a, IntVariable, TDouble> for CustomIntFactor<'a> {
    fn get_weights(&self) -> &Weights<TDouble> {
        &self.weights
    }

    fn get_features_tensor(&self) -> &DenseTensor<TDouble> {
        &self.features_tensor
    }

    fn get_vars_dims(&self) -> &[i64] {
        &self.vars_dims
    }
}

#[test]
pub fn test_simple_model() {
    // Test a prob. distribution P(x1, x2) = 1 / Z f(x1, x2)
    // f(x1, x2) = exp{x1 + x2}
    let x1 = IntVariable::new(IntDomain::new(0, 3), 1);
    let x2 = IntVariable::new(IntDomain::new(5, 7), 6);

    let vars = vec![x1, x2];
    let weights = Weights::new(DenseTensor::<TDouble>::from_array(&[1.0, 1.0]));
    let f1 = Box::new(CustomIntFactor::new(vars.iter().collect(), weights));
    let factors: Vec<Box<Factor<IntVariable, TDouble>>> = vec![f1];

    let mut brute_force = BruteForce::new(&vars, &factors);
    brute_force.infer();

    let log_z = brute_force.log_z();
    assert_eq!(log_z, 10.847795663005575);

    let prob_x1 = brute_force.marginal_log_prob(&[&vars[0]]);
    assert_eq!(prob_x1.size(), &[vars[0].get_domain_size()]);
    // must be a valid distribution
    assert!((prob_x1.log_sum_exp().get_f64() - 0.0).abs() < 1e-9);
    assert_eq!(prob_x1.to_1darray(), [-3.440189698561195, -2.440189698561195, -1.4401896985611948, -0.44018969856119483]);

    let prob_x2 = brute_force.marginal_log_prob(&[&vars[1]]);
    assert_eq!(prob_x2.size(), &[vars[1].get_domain_size()]);
    // must be a valid distribution
    assert!((prob_x2.log_sum_exp().get_f64() - 0.0).abs() < 1e-9);
    assert_eq!(prob_x2.to_1darray(), [-2.4076059644443806, -1.4076059644443806, -0.4076059644443806]);

    assert_eq!(brute_force.all_map(), [[(vars[0].get_id(), 3), (vars[1].get_id(), 7)]
        .iter().cloned()
        .collect::<FnvHashMap<usize, i32>>()]);
}

#[test]
pub fn test_long_chain_model() {
    // Test a prob. distribution P(x1, x2, x3, x4) = 1 / Z f(x1, x2) f(x1, x3) f(x3, x4)
    // f(x, y) = exp(x1 + x2)
    let x1 = IntVariable::new(IntDomain::new(0, 2), 1);
    let x2 = IntVariable::new(IntDomain::new(5, 8), 6);
    let x3 = IntVariable::new(IntDomain::new(9, 10), 9);
    let x4 = IntVariable::new(IntDomain::new(11, 15), 11);
    
    let vars = [x1, x2, x3, x4];
    let weights = Weights::new(DenseTensor::<TDouble>::from_array(&[1.0, 1.0]));

    let f1 = Box::new(CustomIntFactor::new(vec![&vars[0], &vars[1]], weights.clone()));
    let f2 = Box::new(CustomIntFactor::new(vec![&vars[0], &vars[2]], weights.clone()));
    let f3 = Box::new(CustomIntFactor::new(vec![&vars[2], &vars[3]], weights.clone()));
    let factors: [Box<Factor<IntVariable, TDouble>>; 3] = [f1, f2, f3];

    let mut brute_force = BruteForce::new(&vars, &factors);
    brute_force.infer();

    let log_z = brute_force.log_z();
    assert_eq!(log_z, 48.16196373404166);

    let prob_x1 = brute_force.marginal_log_prob(&[&vars[0]]);
    assert_eq!(prob_x1.size(), &[vars[0].get_domain_size()]);
    // must be a valid distribution
    assert!((prob_x1.log_sum_exp().get_f64() - 0.0).abs() < 1e-7);
    assert_eq!(prob_x1.to_1darray(), [-4.142931628499902, -2.142931628499902, -0.14293162849990182]);

    let prob_x2 = brute_force.marginal_log_prob(&[&vars[1]]);
    assert_eq!(prob_x2.size(), &[vars[1].get_domain_size()]);
    // must be a valid distribution
    assert!((prob_x2.log_sum_exp().get_f64() - 0.0).abs() < 1e-7);
    assert_eq!(prob_x2.to_1darray(), [-3.4401896985611984, -2.4401896985611984, -1.4401896985611984, -0.4401896985611984]);

    assert_eq!(brute_force.all_map(), [[
            (vars[0].get_id(), 2), (vars[1].get_id(), 8),
            (vars[2].get_id(), 10), (vars[3].get_id(), 15)
        ].iter().cloned()
        .collect::<FnvHashMap<usize, i32>>()]);
}

pub fn test_simple_cycle_model() {
    // Test a prob. distribution P(x1, x2) = 1 / Z f(x1, x2) f(x2, x3, x4) f(x1, x4)
    //  f(x1, x2) = exp{x1 + x2}
    //  f(x2, x3, x4) = exp{x2 + x3 + x4 / 5}
    //  f(x1, x4) = exp{2*x1 - x4}
    
    let x1 = IntVariable::new(IntDomain::new(0, 3), 2);
    let x2 = IntVariable::new(IntDomain::new(5, 7), 7);
    let x3 = IntVariable::new(IntDomain::new(3, 5), 3);
    let x4 = IntVariable::new(IntDomain::new(9, 13), 10);

    let vars = [x1, x2, x3, x4];

    let f1 = Box::new(CustomIntFactor::new(vec![&vars[0], &vars[1]], Weights::new(DenseTensor::<TDouble>::from_array(&[1.0, 1.0]))));
    let f2 = Box::new(CustomIntFactor::new(vec![&vars[1], &vars[2], &vars[3]], Weights::new(DenseTensor::<TDouble>::from_array(&[1.0, 1.0, 1.0 / 5.0]))));
    let f3 = Box::new(CustomIntFactor::new(vec![&vars[0], &vars[3]], Weights::new(DenseTensor::<TDouble>::from_array(&[2.0, -1.0]))));
    let factors: [Box<Factor<IntVariable, TDouble>>; 3] = [f1, f2, f3];

    let brute_force = BruteForce::new(&vars, &factors);
    let log_z = brute_force.log_z();
    assert_eq!(log_z, 21.979732862);

    for var in &vars {
        // margin with one variable
        let prob_x = brute_force.marginal_log_prob(&[var]);
        assert_eq!(prob_x.size(), &[var.get_domain_size()]);
        assert!((prob_x.log_sum_exp().get_f64() - 0.0).abs() < 1e-7);
    }

    let prob_x = brute_force.marginal_log_prob(&[&vars[0], &vars[1]]);
    assert_eq!(prob_x.size(), &[vars[0].get_domain_size(), vars[1].get_domain_size()]);
    assert!((prob_x.log_sum_exp().get_f64() - 0.0).abs() < 1e-7);

    brute_force.all_map();
}