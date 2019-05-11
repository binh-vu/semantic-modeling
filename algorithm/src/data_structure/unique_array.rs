use std::collections::HashSet;
use std::ops::Index;
use std::process::id;
use std::hash::Hash;

pub struct UniqueArray<V, K: Hash + Eq + PartialEq + Clone=String> {
    data: Vec<V>,
    id: HashSet<K>
}

impl<V, K: Hash + Eq + PartialEq + Clone> UniqueArray<V, K> {
    pub fn new() -> UniqueArray<V, K> {
        UniqueArray {
            data: Vec::new(),
            id: HashSet::new()
        }
    }

    pub fn with_capacity(capacity: usize) -> UniqueArray<V, K> {
        UniqueArray {
            data: Vec::with_capacity(capacity),
            id: HashSet::with_capacity(capacity)
        }
    }

    pub fn push_borrow(&mut self, id: &K, value: V) -> bool {
        if !self.id.contains(id) {
            self.id.insert(id.clone());
            self.data.push(value);

            return true;
        }

        return false;
    }

    pub fn push(&mut self, id: K, value: V) -> bool {
        if !self.id.contains(&id) {
            self.id.insert(id);
            self.data.push(value);

            return true;
        }

        return false;
    }

    pub fn get_ref_value(&self) -> &Vec<V> {
        &self.data
    }

    pub fn get_value(self) -> Vec<V> {
        self.data
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }
}

impl<V, K: Hash + Eq + PartialEq + Clone> Index<usize> for UniqueArray<V, K> {
    type Output = V;

    fn index(&self, idx: usize)-> &V {
        &self.data[idx]
    }
}