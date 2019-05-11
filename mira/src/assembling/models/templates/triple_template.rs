use gmtk::graph_models::*;
use gmtk::tensors::*;
use assembling::models::variable::*;
use assembling::models::example::MRRExample;
use fnv::FnvHashMap;
use super::super::factors::TripleFactor;
use settings::conf_mrf::TemplatesConf;

#[derive(Serialize, Deserialize)]
pub struct TripleTemplate {
    weights: Vec<Weights>,
    domain_tensor: DenseTensor
}

impl TripleTemplate {
    pub fn new(weights: Vec<Weights>) -> TripleTemplate {
        let domain_tensor = BooleanVectorDomain::new().get_domain_tensor().clone().view(&[2, 2, -1]);

        TripleTemplate {
            weights,
            domain_tensor
        }
    }

    pub fn default(templates_config: &TemplatesConf, tf_domain: &TFDomain) -> TripleTemplate {
        let default_weight = Weights::new(DenseTensor::zeros(&[2 * tf_domain.numel() as i64]));
        TripleTemplate::new(vec![default_weight])
    }
}

impl FactorTemplate<MRRExample, TripleVar> for TripleTemplate {
    fn get_weights(&self) -> &[Weights] {
        &self.weights
    }

    fn unroll<'a: 'a1, 'a1>(&'a self, example: &'a1 MRRExample) -> Vec<Box<Factor<'a1, TripleVar> + 'a1>> {
        let mut factors: Vec<Box<Factor<'a1, TripleVar> + 'a1>> = Vec::with_capacity(example.variables.len());

        for i in 0..example.graph.n_edges {
            factors.push(Box::new(TripleFactor::new(
                &example.variables[i],
                &example.observed_edge_features[i],
                &self.weights[0],
                &self.domain_tensor
            )));
        }

        factors
    }
}