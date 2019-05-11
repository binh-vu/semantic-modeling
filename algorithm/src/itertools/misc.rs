use std::collections::HashMap;
use std::hash::Hash;

/// Group elements created in iterator by keys. The iterator doesn't need to be sorted,
/// however, if it's sorted, using itertools.group_by will be faster.combination
/// 
/// Example:
/// ```
/// extern crate algorithm;
/// use algorithm::itertools::group_by;
/// 
/// let x = ["John".to_owned(), "Peter".to_owned(), "Bob".to_owned(), "John".to_owned(), "Peter".to_owned()];
/// let mut result: Vec<Vec<&String>> = group_by(x.into_iter(), |&s| s);
/// result.sort_by_key(|v| v[0]);
/// assert_eq!(result, vec![vec!["Bob"], vec!["John", "John"], vec!["Peter", "Peter"]]);
/// ```
/// 
pub fn group_by<K: Eq + Hash, V, F, I: Iterator<Item=V>>(iterator: I, func: F) -> Vec<Vec<V>>
    where F: Fn(&V) -> K 
{
    let mut index: HashMap<K, Vec<V>> = Default::default();
    for elem in iterator {
        index.entry(func(&elem)).or_insert(Vec::new()).push(elem);
    }

    index.into_iter().map(|(_k, v)| v).collect::<Vec<_>>()
}