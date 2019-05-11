use tensors::*;
use std::ops::*;
use std::sync::*;

#[derive(Serialize, Deserialize)]
pub struct Weights<T: TensorType=TDefault> {
    pub id: i64,
    val: RwLock<DenseTensor<T>>,
}

static mut ID_COUNTER: i64 = 0;
fn get_id() -> i64 {
    unsafe {
        ID_COUNTER += 1;
        ID_COUNTER
    }
}

impl<T: TensorType> Weights<T> {
    pub fn new(val: DenseTensor<T>) -> Weights<T> {
        Weights { val: RwLock::new(val), id: get_id() }
    }

    pub fn new_with_id(val: DenseTensor<T>, id: i64) -> Weights<T> {
        Weights { val: RwLock::new(val), id }
    }

    pub fn clone(&self) -> Weights<T> {
        Weights { val: RwLock::new(self.val.read().unwrap().clone()), id: self.id }
    }

    #[inline]
    pub fn get_value(&self) -> RwLockReadGuard<DenseTensor<T>> {
        return self.val.read().unwrap();
    }

    #[inline]
    pub fn get_value_mut(&self) -> RwLockWriteGuard<DenseTensor<T>> {
        self.val.write().unwrap()
    }

    pub fn copy_(&self, weight: &Weights<T>) {
        // TODO: fix me!! we should borrow mut instead of immutable
        self.val.write().unwrap().copy_(weight.val.write().unwrap().deref());
    }

    pub fn cuda_(&self) {
        self.val.write().unwrap().cuda_();
    }

    pub fn cpu_(&self) {
        self.val.write().unwrap().cpu_();
    }
}