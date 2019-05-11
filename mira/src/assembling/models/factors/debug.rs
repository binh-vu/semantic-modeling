use assembling::models::templates::substructure_template::*;
use gmtk::graph_models::*;
use gmtk::tensors::*;
use assembling::models::variable::*;
use fnv::FnvHashMap;
use super::*;
use std::any::Any;
use algorithm::prelude::*;
use assembling::models::example::*;
use std::rc::Rc;
use std::ops::Deref;

pub struct MRRDebugContainer {
    pub target_assignment: Rc<FnvHashMap<usize, TripleVarValue>>,
    pub true_assignment: Rc<FnvHashMap<usize, TripleVarValue>>,
    pub false_assignment: Rc<FnvHashMap<usize, TripleVarValue>>,
    pub var: TripleVar,
    pub example: Rc<MRRExample>,
    pub tf_domain: Rc<TFDomain>,
    pub pk_pairwise_domain: Rc<PkPairwiseDomain>,
    pub dup_pairwise_domain: Rc<DuplicationPairwiseDomain>,
    pub cooccur_domain: Rc<CooccurrenceDomain>
}

impl DebugContainer for MRRDebugContainer {
    fn as_any(&self) -> &Any {
        self
    }
}

fn filter_non_empty_domain_value(domain: &BinaryVectorDomain<String>, features: &Vec<f32>, weights: &Vec<f32>) -> Vec<(String, f32, f32)> {
    (0..domain.numel())
        .filter(|&i| features[i] != 0.0)
        .map(|i| (domain.get_category(i).clone(), features[i], weights[i]))
        .collect::<Vec<_>>()
}

impl<'a> TripleFactor<'a> {
    pub(super) fn debug_(&self, _con: &DebugContainer) {
        let con: &MRRDebugContainer = _con.as_any().downcast_ref::<MRRDebugContainer>().unwrap();
        let true_fscore = self.score_assignment(&con.true_assignment);
        let false_fscore = self.score_assignment(&con.false_assignment);

        let false_features = self.features_tensor.at(0).view(&[2, -1]);
        let true_features = self.features_tensor.at(1).view(&[2, -1]);
        let weights = self.weights.get_value().view(&[2, -1]);

        println!("\t+ true score={:.4} - false score={:.4}. TripleFactor features:", true_fscore, false_fscore);

        let tmp = filter_non_empty_domain_value(
                &con.tf_domain, 
                &false_features.at(slice![0, ;]).to_1darray(), 
                &weights.at(slice![0, ;]).to_1darray());
        println!("\t\t> when this triple is false: total_score = {:8.5}", tmp.iter().map(|r| r.1 * r.2).sum::<f32>());
        let rows = tmp.iter().map(|r| {
                vec![
                    format!("\t\t . f={}", r.0),
                    format!("= {:.5} * w={:8.5} = {:8.5}", r.1, r.2, r.1 * r.2)
                ]
            })
            .collect::<Vec<_>>();
        for row in align_table(&rows, &["left"; 3]) {
            println!("{}", row.join(" "));
        }

        let tmp = filter_non_empty_domain_value(
                    &con.tf_domain, 
                    &true_features.at(slice![1, ;]).to_1darray(), 
                    &weights.at(slice![1, ;]).to_1darray());
        println!("\t\t> when this triple is true: total_score = {:8.5}", tmp.iter().map(|r| r.1 * r.2).sum::<f32>());
        let rows = tmp.iter().map(|r| {
                vec![
                    format!("\t\t . f={}", r.0),
                    format!("= {:.5} * w={:8.5} = {:8.5}", r.1, r.2, r.1 * r.2)
                ]
            })
            .collect::<Vec<_>>();
        for row in align_table(&rows, &["left"; 3]) {
            println!("{}", row.join(" "));
        }
    }
}

impl<'a> SufficientSubFactor<'a> {
    pub(super) fn debug_(&self, _con: &DebugContainer) {
        let con: &MRRDebugContainer = _con.as_any().downcast_ref::<MRRDebugContainer>().unwrap();
        if self.input_var_idx.iter().filter(|&idx| self.vars[*idx].get_id() == con.var.get_id()).count() == 0 {
            return;
        }

        let example = &con.example;
        let true_fscore = self.score_assignment(&con.true_assignment);
        let false_fscore = self.score_assignment(&con.false_assignment);
        let pk_scope_domain = Rc::new(BinaryVectorDomain::new(vec![String::from("not_same_scope"), "same_scope".to_owned()]));
        let stype_assist_domain = Rc::new(BinaryVectorDomain::new(vec![String::from("potential_gain"), "1-potential_gain".to_owned()]));

        let mut var_info = Vec::new();
        for &idx in self.input_var_idx.iter() {
            let edge = con.example.graph.get_edge_by_id(self.vars[idx].id);
            var_info.push(format!("({}--{}--{})", edge.get_source_node(&con.example.graph).label, edge.label, edge.get_target_node(&con.example.graph).label));
        }

        // TODO: only print var_info if not include all children
        println!("\t+ true score={:.4} - false score={:.4}. {:?} non-zero features between {}:", true_fscore, false_fscore, self.factor_type, var_info.join(", "));

        let false_features = self.features_tensor.at(slice![self.assignment2feature_idx(&con.false_assignment), ;]);
        let true_features = self.features_tensor.at(slice![self.assignment2feature_idx(&con.true_assignment), ;]);

        let weights = self.weights.get_value();

        match self.factor_type {
            SufficientSubFactorType::PairwisePKFactor | SufficientSubFactorType::PairwiseScopeFactor | SufficientSubFactorType::PairwiseCooccurrenceFactor => {
                let (false_arg_slice, true_arg_slice) = if self.vars[self.input_var_idx[0]].id == con.var.get_id() {
                    let another_var_value = example.label.as_ref().unwrap().edge2label[self.vars[self.input_var_idx[1]].id] as i64;
                    (slice![0, another_var_value, ;], slice![1, another_var_value, ;])
                } else {
                    let another_var_value = example.label.as_ref().unwrap().edge2label[self.vars[self.input_var_idx[0]].id] as i64;
                    (slice![another_var_value, 0, ;], slice![another_var_value, 1, ;])
                };
                
                let domain: &BinaryVectorDomain<String> = match self.factor_type {
                    SufficientSubFactorType::PairwisePKFactor => &con.pk_pairwise_domain,
                    SufficientSubFactorType::PairwiseScopeFactor => &pk_scope_domain,
                    SufficientSubFactorType::PairwiseCooccurrenceFactor => &con.cooccur_domain,
                    SufficientSubFactorType::STypeAssistantFactor => &stype_assist_domain,
                    _ => panic!("never reach here")
                };
                
                println!("\t\t> when this triple is false:");
                
                let _tmp = filter_non_empty_domain_value(
                    domain, 
                    &false_features.view(&[2, 2, -1]).at(&false_arg_slice).to_1darray(), 
                    &weights.view(&[2, 2, -1]).at(&false_arg_slice).to_1darray());
                for r in _tmp {
                    println!("\t\t . feature={} = {:8.4} * w={:8.4} = {:8.4}", r.0, r.1, r.2, r.1 * r.2);
                }
                println!("\t\t> when this triple is true:");
                let _tmp = filter_non_empty_domain_value(
                    domain, 
                    &true_features.view(&[2, 2, -1]).at(&true_arg_slice).to_1darray(), 
                    &weights.view(&[2, 2, -1]).at(&true_arg_slice).to_1darray());
                for r in _tmp {
                    println!("\t\t . feature={} = {:8.4} * w={:8.4} = {:8.4}", r.0, r.1, r.2, r.1 * r.2);
                }
            },
            SufficientSubFactorType::DuplicationFactor => {
                let domain: &BinaryVectorDomain<String> = &con.dup_pairwise_domain;

                println!("\t\t> when this triple is false:");
                for i in 0..6 {
                    let _tmp = filter_non_empty_domain_value(
                        domain, 
                        &false_features.view(&[6, -1]).at(slice![i, ;]).to_1darray(), 
                        &weights.view(&[6, -1]).at(slice![i, ;]).to_1darray());
                    for r in _tmp {
                        println!("\t\t . i={}, cat={} = {:8.4} * w={:8.4} = {:8.4}", i, r.0, r.1, r.2, r.1 * r.2);
                    }
                }
                println!("\t\t> when this triple is true:");
                for i in 0..6 {
                    let _tmp = filter_non_empty_domain_value(
                        domain, 
                        &true_features.view(&[6, -1]).at(slice![i, ;]).to_1darray(), 
                        &weights.view(&[6, -1]).at(slice![i, ;]).to_1darray());
                    for r in _tmp {
                        println!("\t\t . i={}, cat={} = {:8.4} * w={:8.4} = {:8.4}", i, r.0, r.1, r.2, r.1 * r.2);
                    }
                }
            },
            SufficientSubFactorType::AllChildrenWrongFactor => {
                println!("Skip all children wrong factors");
            },
            SufficientSubFactorType::STypeAssistantFactor => {
                let domain: &BinaryVectorDomain<String> = &stype_assist_domain;

                let _tmp = filter_non_empty_domain_value(
                    domain,
                    &false_features.to_1darray(),
                    &weights.to_1darray());
                for r in _tmp {
                    println!("\t\t . feature={} = {:8.4} * w={:8.4} = {:8.4}", r.0, r.1, r.2, r.1 * r.2);
                }
                println!("\t\t> when this triple is true:");
                let _tmp = filter_non_empty_domain_value(
                    domain,
                    &true_features.to_1darray(),
                    &weights.to_1darray());
                for r in _tmp {
                    println!("\t\t . feature={} = {:8.4} * w={:8.4} = {:8.4}", r.0, r.1, r.2, r.1 * r.2);
                }
            },
            _ => panic!("Not handle yet")
        }
    }
}