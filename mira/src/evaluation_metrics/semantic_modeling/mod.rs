pub mod alignment;
pub mod internal_structure;
pub mod find_best_map;
pub mod dependent_groups;

use algorithm::prelude::Graph;
use super::semantic_labeling::accuracy_;
pub use self::alignment::{ f1_precision_recall, get_dependent_groups, DataNodeMode };
pub use self::internal_structure::{ Bijection, Triple, TripleSet };

pub fn stype_acc(gold_sm: &Graph, pred_sm: &Graph) -> f64 {
    accuracy_(gold_sm, pred_sm)
}