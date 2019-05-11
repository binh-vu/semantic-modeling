use input::RustInput;
use serde_json;
use itertools::Itertools;

/// Convert query semantic models to list of semantic models match the query
/// For example: ["s00-s11"] to list of ["s00..", "s01...", .., "s11.."] (inclusive range)
pub fn parse_sm_query(input: &RustInput, sm_query: &str) -> Vec<String> {
    let queries: Vec<String> = serde_json::from_str(sm_query).unwrap();
    let mut results: Vec<String> = vec![];
    for query in &queries {
        let (first, second) = query.split("-").next_tuple().unwrap();
        let mut is_in_range = false;
        for sm in &input.semantic_models {
            if sm.id.starts_with(&first) {
                is_in_range = true;
            }

            if is_in_range {
                results.push(sm.id.clone());
            }

            if sm.id.starts_with(&second) {
                is_in_range = false;
            }
        }
    }

    results
}