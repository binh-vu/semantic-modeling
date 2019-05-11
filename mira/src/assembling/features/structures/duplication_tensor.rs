use super::super::local_structure::LocalStructure;
use super::super::stats_predicate::StatsPredicate;
use fnv::{ FnvHashMap, FnvHashSet };

use std::collections::HashMap;
use settings::Settings;
use gmtk::prelude::*;
use assembling::models::variable::*;

pub struct DuplicationTensor {
    tensors: HashMap<String, FnvHashMap<usize, Vec<DenseTensor>>>
}

impl Default for DuplicationTensor {
    fn default() -> DuplicationTensor {
        DuplicationTensor {
            tensors: HashMap::new()
        }
    }
}

impl DuplicationTensor {
    pub fn get_n_features() -> i64 {
        6
    }

    pub fn new(stats_predicate: &StatsPredicate, structure: &LocalStructure) -> DuplicationTensor {
        let mut tensors: HashMap<String, FnvHashMap<usize, Vec<DenseTensor>>> = Default::default();
        let n_features = DuplicationTensor::get_n_features();
        let mut features = vec![0.0; n_features as usize];

        let max_n_dups = Settings::get_instance().mrf.max_n_duplications;

        for (lbl, space) in structure.node_structure_space.iter() {
            tensors.insert(lbl.clone(), Default::default());
            for (ctriple, &child_idx) in space.children.iter() {
                // let child_tensors
                let mut child_tensors = vec![DenseTensor::<TDefault>::default(), DenseTensor::<TDefault>::default()];
                for n_dup in 2..(max_n_dups + 1) {
                    let mut tensor = DenseTensor::zeros(&[2i64.pow(n_dup as u32), n_features]);
                    // let mut dims = vec![2; n_dup];
                    // dims.push(n_features);

                    iter_values(n_dup, 0, |count, values| {
                        if values.len() == 0 {
                            features[0] = 0.01;
                            features[1] = 0.01;
                            features[2] = 0.0;
                            features[3] = 0.0;
                            features[4] = 0.0;
                            features[5] = 0.0;
                        } else if values.len() <= 1 {
                            let is_multi_val = stats_predicate.is_multi_val(&ctriple.0).unwrap_or(false);
                            features[0] = 0.0;
                            features[1] = 0.0;
                            features[2] = is_multi_val as usize as f32;
                            features[3] = !is_multi_val as usize as f32;
                            features[4] = 0.0;
                            features[5] = 0.0;
                        } else {
                            let is_multi_val = stats_predicate.is_multi_val(&ctriple.0).unwrap_or(false);
                            features[0] = 0.0;
                            features[1] = 0.0;
                            features[2] = 0.0;
                            features[3] = 0.0;
                            features[4] = is_multi_val as usize as f32;
                            features[5] = !is_multi_val as usize as f32;
                        }

                        tensor.assign(slice![count as i64, ;], &features);
                    });
                    // child_tensors.push(tensor.view(&dims));
                    child_tensors.push(tensor);
                }
                tensors.get_mut(lbl).unwrap().insert(child_idx, child_tensors);
            }
        }

        DuplicationTensor {
            tensors
        }
    }

    pub fn get_tensor(&self, source_lbl: &str, link_id: usize, n_dups: usize) -> DenseTensor {
        self.tensors[source_lbl][&link_id][n_dups].clone_reference()
    }
}

fn iter_values<F>(n_children: usize, n_offset: usize, mut func: F) 
    where F: FnMut(usize, &FnvHashSet<usize>) -> ()
{
    let mut current_val_index = vec![0; n_children];
    let max_val_index = vec![2; n_children];
    let mut values: FnvHashSet<usize> = Default::default();
    let mut count = 0;

    func(count, &values);

    loop {
        // iterate through each assignment
        let mut i = current_val_index.len() - 1;
        loop {
            // move to next state & set value of variables to next state value
            current_val_index[i] += 1;
            if current_val_index[i] == max_val_index[i] {
                current_val_index[i] = 0;
                values.remove(&(i + n_offset));
                if i == 0 {
                    // no more values, terminate
                    return;    
                }
                i = i - 1;
            } else {
                values.insert(i + n_offset);
                break;
            }
        }

        count += 1;
        // yield current values
        func(count, &values)
    }
}