use std::collections::HashSet;
use num_traits::*;
use std::hash::Hash;
use tensors::*;
use graph_models::*;
use std::ops::AddAssign;
use std::fmt::Debug;


#[derive(Debug)]
pub struct ObservedFeaturesBuilder<V: 'static + Eq + Hash + Clone + Debug, T: 'static + TensorType=TDefault> {
    active_categories: Vec<V>,
    active_values: Vec<T::PrimitiveType>
}

impl<V: 'static + Eq + Hash + Clone + Debug, T: 'static + TensorType> ObservedFeaturesBuilder<V, T> {
    pub fn new() -> ObservedFeaturesBuilder<V, T> {
        ObservedFeaturesBuilder {
            active_categories: Vec::new(),
            active_values: Vec::new(),
        }
    }

    pub fn create_domain(builders: &[ObservedFeaturesBuilder<V, T>]) -> BinaryVectorDomain<V, T> {
        let mut catset: HashSet<&V> = HashSet::new();
        let mut cats: Vec<V> = Vec::new();

        for builder in builders {
            for cat in &builder.active_categories {
                if !catset.contains(cat) {
                    catset.insert(cat);
                    cats.push(cat.clone());
                }
            }
        }

        BinaryVectorDomain::new(cats)
    }

    pub fn create_tensor(&self, domain: &BinaryVectorDomain<V, T>) -> DenseTensor<T> {
        let mut val = DenseTensor::zeros(&[domain.numel() as i64]);
        if self.active_categories.len() == 0 {
            return val;
        }

        let (idx, active_values): (Vec<i64>, Vec<T::PrimitiveType>) = self.active_categories.iter().zip(self.active_values.iter())
            .filter(|(c, _v)| domain.has_category(&c))
            .map(|(c, v)| (domain.get_category_index(&c) as i64, v)).unzip();

        if idx.len() == 0 {
            return val;
        }

        val.assign(&idx, &active_values);
        val
    }
}

impl<V: 'static + Eq + Hash + Clone + Debug, T: 'static + TensorType> AddAssign<(V, f64)> for ObservedFeaturesBuilder<V, T> {
    fn add_assign(&mut self, rhs: (V, f64)) {
        self.active_categories.push(rhs.0);
        self.active_values.push(T::PrimitiveType::from_f64(rhs.1).unwrap());
    }
}

impl<V: 'static + Eq + Hash + Clone + Debug, T: 'static + TensorType> AddAssign<(V, f32)> for ObservedFeaturesBuilder<V, T> {
    fn add_assign(&mut self, rhs: (V, f32)) {
        self.active_categories.push(rhs.0);
        self.active_values.push(T::PrimitiveType::from_f32(rhs.1).unwrap());
    }
}

impl<V: 'static + Eq + Hash + Clone + Debug, T: 'static + TensorType> AddAssign<V> for ObservedFeaturesBuilder<V, T> {
    fn add_assign(&mut self, rhs: V) {
        self.active_categories.push(rhs);
        self.active_values.push(T::PrimitiveType::from_f64(1.0).unwrap());
    }
}