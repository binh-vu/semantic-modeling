use graph_models::traits::FactorTemplate;
use tensors::*;
use graph_models::traits::*;
use graph_models::weights::Weights;
use serde::de::Deserialize;
use serde::ser::Serialize;

#[derive(Serialize, Deserialize)]
pub struct LogLinearModel<E, V: Variable, T: 'static + TensorType=TDefault> {
    #[serde(bound(deserialize = "Box<FactorTemplate<E, V, T>>: Deserialize<'de>", serialize = "Box<FactorTemplate<E, V, T>>: Serialize"))]
    pub templates: Vec<Box<FactorTemplate<E, V, T>>>,
}

impl<E, V: Variable, T: 'static + TensorType> LogLinearModel<E, V, T> {
    pub fn new(templates: Vec<Box<FactorTemplate<E, V, T>>>) -> LogLinearModel<E, V, T> {
        LogLinearModel { templates }
    }

    pub fn add_template(&mut self, template: Box<FactorTemplate<E, V, T>>) {
        self.templates.push(template);
    }

    pub fn get_factors<'a: 'a1, 'a1>(&'a self, example: &'a1 E) -> Vec<Box<Factor<'a1, V, T> + 'a1>> {
        let mut factors = Vec::with_capacity(8);
        for template in &self.templates {
            factors.append(&mut template.unroll(&example));
        }
        return factors;
    }

    pub fn get_parameters(&self) -> Vec<&Weights<T>> {
        let mut parameters = Vec::with_capacity(self.templates.len());
        for template in &self.templates {
            for weight in template.get_weights() {
                parameters.push(weight);
            }
        }
        return parameters;
    }

    pub fn clone_parameters(&self) -> Vec<Weights<T>> {
        let mut parameters = Vec::with_capacity(self.templates.len());
        for template in &self.templates {
            for weight in template.get_weights() {
                parameters.push(weight.clone());
            }
        }
        return parameters;
    }

    pub fn set_parameters(&self, new_parameters: &[Weights<T>]) {
        let mut i = 0;
        for template in &self.templates {
            for weights in template.get_weights() {
                assert_eq!(weights.id, new_parameters[i].id);
                weights.copy_(&new_parameters[i]);
                i += 1;
            }
        }
    }

    pub fn cuda_(&self) {
        for template in &self.templates {
            for weight in template.get_weights() {
                weight.cuda_();
            }
        }
    }

    pub fn cpu_(&self) {
        for template in &self.templates {
            for weight in template.get_weights() {
                weight.cpu_();
            }
        }
    }
}