use tensors::*;
use std::hash::Hash;
use fnv::FnvHashMap;
use std::sync::Mutex;
use std::clone::Clone;
use std::sync::Arc;

pub struct ValueAccumulator {
    pub(super) value: f64
}

impl ValueAccumulator {
    pub fn new() -> ValueAccumulator {
        ValueAccumulator { value: 0.0 }
    }

    pub fn clear(&mut self) {
        self.value = 0.0;
    }

    pub fn get_value(&self) -> f64 {
        self.value
    }

    pub fn accumulate(&mut self, value: f64) {
        self.value += value;
    }
}

pub trait Tensor1Accumulator<V: Eq + Hash + Clone, T: TensorType> {
    fn accumulate(&mut self, obj: &V, value: &DenseTensor<T>);
    fn accumulate_minus(&mut self, obj: &V, value: &DenseTensor<T>);
}

pub struct Tensor1AccumulatorDict<V: Eq + Hash + Clone, T: TensorType> {
    pub(super) tensors: FnvHashMap<V, DenseTensor<T>>
}

impl<V: Eq + Hash + Clone, T: TensorType> Tensor1AccumulatorDict<V, T> {
    pub fn new() -> Tensor1AccumulatorDict<V, T> {
        Tensor1AccumulatorDict { tensors: Default::default() }
    }

    pub fn track_object(&mut self, obj: V, zero_like: &DenseTensor<T>) {
        self.tensors.insert(obj, DenseTensor::<T>::zeros_like(zero_like));
    }

    pub fn get_value(&self, obj: &V) -> &DenseTensor<T> {
        return &self.tensors[obj];
    }

    pub fn get_value_mut(&mut self, obj: &V) -> &mut DenseTensor<T> {
        return self.tensors.get_mut(obj).unwrap();
    }

    pub fn clear(&mut self) {
        for v in self.tensors.values_mut() {
            v.zero_();
        }
    }
}

impl<V: Eq + Hash + Clone, T: TensorType> Tensor1Accumulator<V, T> for Tensor1AccumulatorDict<V, T> {
    fn accumulate(&mut self, obj: &V, value: &DenseTensor<T>) {
        *self.tensors.get_mut(obj).unwrap() += value;
    }

    fn accumulate_minus(&mut self, obj: &V, value: &DenseTensor<T>) {
        *self.tensors.get_mut(obj).unwrap() -= value;
    }
}

pub struct SafeTensor1AccumulatorDict<V: Eq + Hash + Clone, T: TensorType> {
    pub(super) tensors: Arc<Mutex<FnvHashMap<V, DenseTensor<T>>>>
}

impl<V: Eq + Hash + Clone, T: TensorType> SafeTensor1AccumulatorDict<V, T> {
    pub fn new(accumulator: &Tensor1AccumulatorDict<V, T>) -> SafeTensor1AccumulatorDict<V, T> {
        SafeTensor1AccumulatorDict { tensors: Arc::new(Mutex::new(accumulator.tensors.clone())) }
    }

    pub fn update(&self, accumulator: &mut Tensor1AccumulatorDict<V, T>) {
        let tensors = self.tensors.lock().unwrap();
        for (k, v) in accumulator.tensors.iter_mut() {
            *v += &tensors[k];
        }
    }

    pub fn clone_reference(&self) -> SafeTensor1AccumulatorDict<V, T> {
        SafeTensor1AccumulatorDict {
            tensors: Arc::clone(&self.tensors)
        }
    }
}

impl<V: Eq + Hash + Clone, T: TensorType> Tensor1Accumulator<V, T> for SafeTensor1AccumulatorDict<V, T> {
    fn accumulate(&mut self, obj: &V, value: &DenseTensor<T>) {
        *self.tensors.lock().unwrap().get_mut(obj).unwrap() += value;
    }

    fn accumulate_minus(&mut self, obj: &V, value: &DenseTensor<T>) {
        *self.tensors.lock().unwrap().get_mut(obj).unwrap() -= value;
    }
}

