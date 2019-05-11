use std::hash::Hash;
use tensors::*;
use graph_models::traits::*;
use fnv::FnvHashMap;

#[derive(Serialize, Deserialize)]
pub struct BinaryVectorValue<T: TensorType=TDefault> {
    pub tensor: DenseTensor<T>,
    pub idx: usize
}

impl<T: TensorType> Clone for BinaryVectorValue<T> {
    fn clone(&self) -> BinaryVectorValue<T> {
        BinaryVectorValue {
            tensor: self.tensor.clone(),
            idx: self.idx.clone()
        }
    }
}

pub trait TBinaryVectorDomain<V: Eq + Hash + Clone, U: TensorType=TDefault>: Domain {
    fn get_category(&self, idx: usize) -> &V;
    fn get_category_index(&self, category: &V) -> usize;
    fn has_category(&self, category: &V) -> bool;
    fn encode_value(&self, category: &V) -> BinaryVectorValue<U>;
    fn get_domain_tensor(&self) -> &DenseTensor<U>;
    fn cuda_(&mut self);
    fn cpu_(&mut self);
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BinaryVectorDomain<V: 'static + Eq + Hash + Clone, U: 'static + TensorType=TDefault> {
    pub categories: FnvHashMap<V, usize>,
    inversed_categories: Vec<V>,
    is_sparse: bool,
    domain_tensor: Option<DenseTensor<U>>,
}

impl<V: Eq + Hash + Clone, U: TensorType> Clone for BinaryVectorDomain<V, U> {
    fn clone(&self) -> BinaryVectorDomain<V, U> {
        return BinaryVectorDomain {
            categories: self.categories.clone(),
            inversed_categories: self.inversed_categories.clone(),
            is_sparse: self.is_sparse,
            domain_tensor: self.domain_tensor.clone()
        }
    }
}

impl<V: Eq + Hash + Clone, U: TensorType> Default for BinaryVectorDomain<V, U> {
    fn default() -> BinaryVectorDomain<V, U> {
        return BinaryVectorDomain {
            categories: Default::default(),
            inversed_categories: Default::default(),
            is_sparse: false,
            domain_tensor: None
        }
    }
}

impl<V: Eq + Hash + Clone, U: TensorType> BinaryVectorDomain<V, U> {
    pub fn new(cats: Vec<V>) -> BinaryVectorDomain<V, U> {
        let categories: FnvHashMap<V, usize> = cats.iter().enumerate().map(|(idx, v)| (v.clone(), idx)).collect();

        return BinaryVectorDomain {
            categories,
            inversed_categories: cats,
            is_sparse: false,
            domain_tensor: None,
        }
    }

    pub fn compute_domain_tensor(&mut self) {
        if let None = self.domain_tensor {
            let n_cat: i64 = self.categories.len() as i64;
            let mut domain_tensor = DenseTensor::<U>::create(&[n_cat, n_cat]);
            for i in 0..n_cat {
                domain_tensor.assign(i, self.get_value(i as usize).tensor);
            }

            self.domain_tensor = Some(domain_tensor);
        }
    }
}

impl<V: Eq + Hash + Clone, U: TensorType> TBinaryVectorDomain<V, U> for BinaryVectorDomain<V, U> {
    fn get_category(&self, idx: usize) -> &V {
        return &self.inversed_categories[idx];
    }

    fn get_category_index(&self, category: &V) -> usize {
        return self.categories[category];
    }

    fn has_category(&self, category: &V) -> bool {
        return self.categories.contains_key(category);
    }

    fn encode_value(&self, category: &V) -> BinaryVectorValue<U> {
        if self.categories.contains_key(category) {
            self.get_value(self.categories[category]) 
        } else {
            return BinaryVectorValue {
                tensor: DenseTensor::<U>::zeros(&vec![self.categories.len() as i64]),
                idx: self.categories.len() + 1
            }
        }
    }

    fn get_domain_tensor(&self) -> &DenseTensor<U> {
        return self.domain_tensor.as_ref().expect("`compute_domain_tensor` must be called first");
    }

    fn cuda_(&mut self) {
        self.domain_tensor.as_mut().unwrap().cuda_();
    }

    fn cpu_(&mut self) {
        self.domain_tensor.as_mut().unwrap().cpu_();
    }
}

impl<V: 'static + Eq + Hash + Clone, U: 'static + TensorType> Domain for BinaryVectorDomain<V, U> {
    type Value = BinaryVectorValue<U>;

    fn numel(&self) -> usize {
        self.categories.len()
    }

    fn get_index(&self, value: &<Self as Domain>::Value) -> usize {
        return value.idx;
    }

    fn get_value(&self, index: usize) -> <Self as Domain>::Value {
        let mut tensor = DenseTensor::<U>::zeros(&vec![self.categories.len() as i64]);
        tensor.assign(index as i64, 1.0);

        return BinaryVectorValue {
            tensor,
            idx: index,
        }
    }
}