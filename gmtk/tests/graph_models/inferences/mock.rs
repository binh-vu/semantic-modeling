use gmtk::prelude::*;
use fnv::FnvHashMap;
use std::path::PathBuf;
use std::fs::File;
use serde_json;
use std::io::*;

pub type MockVar<'a> = BooleanVectorVariable<'a, TDouble>;
pub type MockVarValue = BinaryVectorValue<TDouble>;

pub struct FixedScoredFactor<'a> {
    scores_tensor: DenseTensor<TDouble>,
    variables: Vec<&'a MockVar<'a>>,
    vars_dims: Vec<i64>
}

impl<'a> FixedScoredFactor<'a> {
    pub fn new(scores_tensor: DenseTensor<TDouble>, variables: Vec<&'a MockVar<'a>>) -> FixedScoredFactor<'a> {
        FixedScoredFactor {
            scores_tensor,
            vars_dims: variables.iter().map(|v| v.get_domain().numel() as i64).collect::<Vec<_>>(),
            variables,
        }
    }

    fn val2feature_idx(&self, values: &[&MockVarValue]) -> i64 {
        // Note: you may want to override this function to improve speed
        let vars = self.get_variables();
        ravel_index(&(0..vars.len())
            .map(|i| vars[i].get_domain().get_index(values[i]) as i64)
            .collect(),
            &self.vars_dims
        )
    }
}

impl<'a> Factor<'a, MockVar<'a>, TDouble> for FixedScoredFactor<'a> {
    fn get_id(&self) -> usize {
        ((self as *const _) as *const usize) as usize
    }

    fn get_variables(&self) -> &[&'a MockVar<'a>] {
        &self.variables
    }

    fn score_assignment(&self, assignment: &FnvHashMap<usize, MockVarValue>) -> f64 {
        let values: Vec<_> = self.get_variables().iter().map(|&v| &assignment[&v.get_id()]).collect();
        self.scores_tensor
            .at(self.val2feature_idx(&values) as i64)
            .get_f64()
    }

    fn get_scores_tensor(&self) -> DenseTensor<TDouble> {
        self.scores_tensor.clone_reference()
    }
    
    #[allow(unused_variables)]
    fn compute_gradients(&self, target_assignment: &FnvHashMap<usize, MockVarValue>, inference: &Inference<'a, MockVar<'a>, TDouble>) -> Vec<(i64, DenseTensor<TDouble>)> {
        unimplemented!()
    }

    fn touch(&self, var: &MockVar<'a>)-> bool {
        for v in &self.variables {
            if var.get_id() == v.get_id() {
                return true;
            }
        }
        return false;
    }
}

pub fn get_assignment_score<'a>(assignment: &FnvHashMap<usize, MockVarValue>, factors: &[Box<Factor<'a, MockVar<'a>, TDouble> + 'a>]) -> f64 {
    factors.iter().map(|f| f.score_assignment(assignment)).sum()
}

pub fn get_factors_and_vars<F>(func: F)
    where F: for <'a> Fn(&'a [MockVar<'a>], &[Box<Factor<'a, MockVar<'a>, TDouble> + 'a>]) -> () {
    let inputs = [
        "tests/graph_models/inferences/data/factor_graphs.0.json",
        "tests/graph_models/inferences/data/factor_graphs.1.json"
    ];

    for input in &inputs {
        let mut finput = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        finput.push(input);

        let result: serde_json::Value = serde_json::from_reader(BufReader::new(File::open(finput).unwrap())).unwrap();

        let domain = BooleanVectorDomain::new();
        let variables = result["variables"].as_array().unwrap().iter().map(|ref v| {
            BooleanVectorVariable::new(&domain, domain.get_value(v.as_u64().unwrap() as usize))
        }).collect::<Vec<_>>();
        let ref_vars = &variables;

        let factor_vars = result["factor_variables"].as_array().unwrap();
        let factors = result["factors"].as_array().unwrap().iter().enumerate().map(|(i, f)| {
            let res: Box<Factor<BooleanVectorVariable<TDouble>, TDouble>> = Box::new(FixedScoredFactor::new(
                DenseTensor::<TDouble>::from_array(&f.as_array().unwrap().iter().map(|c| c.as_f64().unwrap()).collect::<Vec<_>>()),
                factor_vars[i].as_array().unwrap().iter().map(|j| &ref_vars[j.as_u64().unwrap() as usize]).collect::<Vec<_>>()));
            res
        }).collect::<Vec<_>>();

        func(&variables, &factors);
    }
}