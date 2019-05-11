use graph_models::traits::Variable;
use std::collections::HashSet;
use fnv::FnvHashMap;

pub fn get_variables_index<'a, V: Variable>(variables: &[&'a V]) -> HashSet<usize> {
    let mut vars_set = HashSet::with_capacity(variables.len());
    for v in variables {
        vars_set.insert(v.get_id());
    }
    vars_set
}

/// Iter all values using counter method and apply a function, the produce
pub fn iter_values<V: 'static + Variable, F>(vars: &[&V], mut func: F)
    where F: FnMut(i64, &Vec<i64>, &Vec<V::Value>) -> ()
{
    let max_val_index: Vec<i64> = vars.iter().map(|v| v.get_domain_size()).collect();
    let mut current_val = Vec::with_capacity(vars.len());
    let mut current_ravelled_index: i64 = -1;
    let mut current_val_index: Vec<i64> = vec![0; vars.len()];
    current_val_index[vars.len() - 1] = -1;

    // set default assignment to index 0
    for var in vars.iter() {
        current_val.push(var.get_domain().get_value(0));
    }

    let mut no_more_vals = false;

    loop {
        // iterate through each assignment
        let mut i = current_val_index.len() - 1;
        current_ravelled_index += 1;
        loop {
            // move to next state & set value of variables to next state value
            current_val_index[i] += 1;
            if current_val_index[i] == max_val_index[i] {
                current_val_index[i] = 0;
                current_val[i] = vars[i].get_domain().get_value(current_val_index[i] as usize);
                if i == 0 {
                    no_more_vals = true;
                    break;
                }

                i -= 1;
            } else {
                current_val[i] = vars[i].get_domain().get_value(current_val_index[i] as usize);
                break;
            }
        }

        if no_more_vals {
            // already iterated through all values
            break;
        }

        func(current_ravelled_index, &current_val_index, &current_val);
    }
}

/// Iter all assignment using counter method and apply a function
pub fn iter_assignment<V: Variable, F>(vars: &[&V], mut func: F)
    where F: FnMut(&Vec<i64>, &mut FnvHashMap<usize, V::Value>) -> ()
{
    let max_val_index: Vec<i64> = vars.iter().map(|v| v.get_domain_size()).collect();

    let mut current_val = Vec::with_capacity(vars.len());
    let mut target_assignment: FnvHashMap<usize, V::Value> = Default::default();

    let mut current_val_index: Vec<i64> = vec![0; vars.len()];
    current_val_index[vars.len() - 1] = -1;

    let current_val_ptr = &mut current_val as *mut Vec<V::Value>;

    // set default assignment to index 0
    for var in vars.iter() {
        unsafe { (&mut *current_val_ptr).push(var.get_domain().get_value(0)); }
        target_assignment.insert(var.get_id(), var.get_domain().get_value(0));
    }

    let mut no_more_vals = false;

    loop {
        // iterate through each assignment
        let mut i = current_val_index.len() - 1;
        loop {
            // move to next state & set value of variables to next state value
            current_val_index[i] += 1;
            if current_val_index[i] == max_val_index[i] {
                current_val_index[i] = 0;
                unsafe {
                    (&mut *current_val_ptr)[i] = vars[i].get_domain().get_value(current_val_index[i] as usize);
                }
                target_assignment.insert(vars[i].get_id(), vars[i].get_domain().get_value(current_val_index[i] as usize));
                if i == 0 {
                    // already iterated through all values
                    no_more_vals = true;
                    break;
                }

                i -= 1;
            } else {
                unsafe {
                    (&mut *current_val_ptr)[i] = vars[i].get_domain().get_value(current_val_index[i] as usize);
                }
                target_assignment.insert(vars[i].get_id(), vars[i].get_domain().get_value(current_val_index[i] as usize));
                break;
            }
        }

        if no_more_vals {
            // already iterated through all values
            break;
        }

        func(&current_val_index, &mut target_assignment);
    }
}