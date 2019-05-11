use rdb2rdf::models::semantic_model::SemanticModel;
use std::collections::HashMap;
use std::ops::AddAssign;
use utils::*;
use itertools::Itertools;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Statistic {
    pub p_n: HashMap<String, f32>,
    pub p_l_given_s: HashMap<String, HashMap<String, f32>>,
    pub p_o_given_sl: HashMap<(String, String), HashMap<String, f32>>,
    p_l_given_so: HashMap<(String, String), HashMap<String, f32>>,
}

impl Statistic {
    pub fn new(train_sms: &[&SemanticModel]) -> Statistic {
        let mut c_l_given_s: HashMap<String, HashMap<String, i32>> = Default::default();
        let mut c_l_given_so: HashMap<(String, String), HashMap<String, i32>> = Default::default();
        let mut c_o_given_sl: HashMap<(String, String), HashMap<String, i32>> = Default::default();
        let mut c_n: HashMap<String, i32> = Default::default();

        // estimate stat variables
        for sm in train_sms {
            let g = &sm.graph;
            for e in g.iter_edges() {
                let source = g.get_node_by_id(e.source_id);
                let target = g.get_node_by_id(e.target_id);

                // COMPUTE c_l_given_so
                if target.is_class_node() {
                    let so = (source.label.clone(), target.label.clone());
                    let entry = c_l_given_so
                        .entry(so).or_insert(Default::default())
                        .entry(e.label.clone()).or_insert(0);
                    *entry += 1;

                    *c_o_given_sl
                        .entry((source.label.clone(), e.label.clone())).or_insert(Default::default())
                        .entry(target.label.clone()).or_insert(0) += 1;
                }
            }

            for n in g.iter_class_nodes() {
                // COMPUTE c_n
                if !c_n.contains_key(&n.label) {
                    c_n.insert(n.label.clone(), 0);
                }
                *c_n.get_mut(&n.label).unwrap() += 1;

                // COMPUTE c_l_given_s
                if !c_l_given_s.contains_key(&n.label) {
                    c_l_given_s.insert(n.label.clone(), Default::default());
                }

                for e in n.iter_outgoing_edges(&g).unique_by(|e| &e.label) {
                    c_l_given_s.get_mut(&n.label).unwrap().entry(e.label.clone()).or_insert(0).add_assign(1);
                }
            }
        }

        let mut p_n: HashMap<String, f32> = Default::default();
        let mut p_l_given_so: HashMap<(String, String), HashMap<String, f32>> = Default::default();
        let mut p_l_given_s: HashMap<String, HashMap<String, f32>> = Default::default();
        let mut p_o_given_sl: HashMap<(String, String), HashMap<String, f32>> = Default::default();

        for (n, &v) in c_n.iter() {
            p_n.insert(n.clone(), v as f32 / train_sms.len() as f32);
        }

        for so in c_l_given_so.keys() {
            let total: f32 = c_l_given_so[so].values().sum::<i32>() as f32;
            p_l_given_so.insert(so.clone(), Default::default());
            for (l, &v) in c_l_given_so[so].iter() {
                p_l_given_so.get_mut(so).unwrap().insert(l.clone(), v as f32 / total);
            }
        }

        for s in c_l_given_s.keys() {
            let total: f32 = c_n[s] as f32;
            p_l_given_s.insert(s.clone(), c_l_given_s[s].iter()
                .map(|(l, &v)| (l.clone(), v as f32 / total))
                .collect::<HashMap<_, _>>());
        }

        for sl in c_o_given_sl.keys() {
            let total = c_o_given_sl[sl].values().sum::<i32>() as f32;
            p_o_given_sl.insert(sl.clone(), Default::default());
            for (o, &v) in c_o_given_sl[sl].iter() {
                p_o_given_sl.get_mut(sl).unwrap().insert(o.clone(), v as f32 / total);
            }
        }

        Statistic { p_n, p_l_given_so, p_l_given_s, p_o_given_sl }
    }

    pub fn p_l_given_s(&self, s: &str, l: &str, default: f32) -> f32 {
        if !self.p_l_given_s.contains_key(s) {
            default
        } else {
            *self.p_l_given_s[s].get(l).unwrap_or(&default)
        }
    }

    pub fn p_o_given_sl(&self, s: &String, l: &String, o: &str, default: f32) -> f32 {
        if !dict_has(&self.p_o_given_sl, s, l) {
            default
        } else {
            *dict_get(&self.p_o_given_sl, s, l).unwrap().get(o).unwrap_or(&default)
        }
    }

    pub fn p_l_given_so(&self, s: &String, l: &String, o: &String, default: f32) -> f32 {
        if !dict_has(&self.p_l_given_so, s, o) {
            default
        } else {
            *dict_get(&self.p_l_given_so, s, o).unwrap().get(l).unwrap_or(&default)
        }
    }

    pub fn p_n(&self, lbl: &str, default: f32) -> f32 {
        // Compute prior of a class label P(node=lbl)
        *self.p_n.get(lbl).unwrap_or(&default)
    }

    pub fn p_triple(&self, s: &String, l: &String, o: &String, default: f32) -> f32 {
        // Compute P(source=s, predicate=l, object=0)
        return self.p_n(s, default) * self.p_n(o, default) * self.p_l_given_so(s, l, o, default)
    }
}

