use gmtk::graph_models::*;
use gmtk::tensors::*;
use fnv::FnvHashMap;
use graph_models::inferences::brute_force::CustomIntFactor;
use super::mock;

pub fn get_factors_and_vars<F>(func: F)
    where F: for <'a> Fn(&'a [IntVariable], &[Box<Factor<'a, IntVariable, TDouble> + 'a>]) -> ()
{
    // Return list of factors (exp class: exp{w.dot(x)}) and its variable (int variable)
    let examples = [
    	// simplest: P(x1, x2) = 1 / Z f(x1, x2)
        (
        	vec![(0, 3, 2), (5, 7, 5)],
        	vec![
	        	(vec![0, 1], vec![1.0, 1.0]),
	    	]
	    ),
	    // long chain
	    (
	    	vec![(0, 2, 1), (5, 8, 6), (9, 10, 9), (11, 15, 11)],
	    	vec![
	        	(vec![0, 1], vec![1.0, 1.0]),
	        	(vec![1, 2], vec![1.0, 1.0]),
	        	(vec![1, 3], vec![1.0, 1.0])
	    	]
	    ),
	    // disconnect: P(x1, x2, x3, x4) = 1 / Z f(x1, x2) f(x3, x4)
	    (
	    	vec![(0, 3, 3), (5, 7, 7), (3, 5, 4), (9, 13, 10)],
	    	vec![
	        	(vec![0, 1], vec![1.0, 1.0]),
	        	(vec![2, 3], vec![2.0, 1.0]),
	    	]
	    ),
	    // factor of one variable: P(x1, x2) = 1 / Z f(x1) f(x1, x2)
	    (
	    	vec![(0, 3, 2), (5, 7, 5)],
	    	vec![
	        	(vec![0], vec![2.0]),
	        	(vec![0, 1], vec![1.0, 1.0]),
	    	]
	    ),
	    // just have more factors: P(x1, x2, x3, x4) = 1 / Z f(x1, x2) f(x3, x4) f(x1, x4)
	    (
	    	vec![(0, 3, 1), (5, 7, 6), (3, 5, 5), (9, 13, 12)],
	    	vec![
	        	(vec![0, 1], vec![1.0, 1.0]),
	        	(vec![2, 3], vec![1.0, 1.0 / 5.0]),
	        	(vec![0, 3], vec![2.0, -1.0])
	    	]
	    ),
	    // test factors have more than 2 variables: P(x1, x2, x3, x4, x5, x6) = 1 / Z f(x1, x2, x3, x4, x5, x6)
	    (
	    	vec![(-1, 1, 0), (-1, 1, -1), (-1, 1, 1), (-1, 1, 0), (-1, 1, -1), (-1, 1, 1)],
	    	vec![
	        	(vec![0, 1, 2, 3, 4, 5], vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
	    	]
	    ),
	    // factors have more than 2 variables + some factors:
	    // P(x1, x2, x3, x4, x5, x6) = 1 / Z f(x1, x2, x3, x4, x5, x6) f(x5, x7) f(x7, x8)
	    (
	    	vec![(-1, 1, 1), (-1, 1, 0), (-1, 1, -1), (-1, 1, 1), (-1, 1, 0), (-1, 1, 1), (2, 3, 2), (1, 2, 2)],
	    	vec![
	        	(vec![0, 1, 2, 3, 4, 5], vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
	        	(vec![4, 6], vec![2.0, 1.0]),
	        	(vec![6, 7], vec![1.5, 3.2]),
	    	]
	    )
    ];

    for example in &examples {
        let vars = example.0.iter().map(|&(min, max, val)| {
            IntVariable::new(IntDomain::new(min, max), val)
        }).collect::<Vec<IntVariable>>();
        let ref_vars = &vars;

        let factors = example.1.iter().map(|&(ref var_idxs, ref weights)| {
            let vars: Vec<&IntVariable> = var_idxs.iter().map(|&i| { &ref_vars[i] }).collect();
			let x: Box<Factor<IntVariable, TDouble>> = Box::new(CustomIntFactor::new(
                vars,
                Weights::new(DenseTensor::<TDouble>::from_array(&weights))
            ));
            x
		}).collect::<Vec<Box<Factor<IntVariable, TDouble>>>>();

        func(ref_vars, &factors);
    }
}

pub fn get_assignment_score<'a>(assignment: &FnvHashMap<usize, i32>, factors: &[Box<Factor<'a, IntVariable, TDouble> + 'a>]) -> f64 {
    factors.iter().map(|f| f.score_assignment(assignment)).sum()
}


#[test]
pub fn test_simple_acyclic_model() {
    get_factors_and_vars(|vars, factors| {
        let mut brute_force = BruteForce::new(vars, factors);
        let mut bp = BeliefPropagation::new(InferProb::MARGINAL, vars, factors, 120);
        let mut map_bp = BeliefPropagation::new(InferProb::MAP, vars, factors, 120);

        brute_force.infer();
        bp.infer();
        map_bp.infer();

        let map_sols = brute_force.all_map();
        let bp_map_sol = map_bp.map();

        let map_score = get_assignment_score(&map_sols[0], factors);
        assert!(brute_force.log_z() - bp.log_z() < 1e-9);
        assert_eq!(map_score, get_assignment_score(&bp_map_sol, factors));
    });
}

#[test]
pub fn test_misc() {
	get_factors_and_vars(|vars, factors| {
        let mut bp = BeliefPropagation::new(InferProb::MARGINAL, vars, factors, 120);
        bp.infer();
		assert_eq!(bp.log_z(), bp.log_z(), "Repeated log_z will produce same result");
		assert_eq!(bp.log_z(), bp.log_z(), "Repeated log_z will produce same result");
	});
}

#[test]
pub fn test_acyclic_model_from_data() {
	mock::get_factors_and_vars(|vars, factors| {
		let mut brute_force = BruteForce::new(vars, factors);
        let mut bp = BeliefPropagation::new(InferProb::MARGINAL, vars, factors, 120);

		brute_force.infer();
        bp.infer();

       	assert!(brute_force.log_z() - bp.log_z() < 1e-9);
	});
}