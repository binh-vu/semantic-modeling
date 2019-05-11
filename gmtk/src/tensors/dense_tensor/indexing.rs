use super::c_api::*;
use super::tensor::DenseTensor;
use tensors::AdvancedSlice;
use tensors::tensor_index::*;
use tensors::tensor_type::*;


impl<T: TensorType> TensorIndex<i64> for DenseTensor<T> {
    type Output = DenseTensor<T>;
    fn at(&self, idx: i64) -> DenseTensor<T> {
        unsafe {
            return DenseTensor::new(cten_select(self.tensor, 0, idx));
        }
    }
}
impl<'a, T: TensorType> TensorIndex<&'a Vec<i64>> for DenseTensor<T> {
    type Output = DenseTensor<T>;
    fn at(&self, idx: &'a Vec<i64>) -> DenseTensor<T> {
        unsafe {
            return DenseTensor::new(cten_index_select_v(self.tensor, 0, idx.len() as i64, idx.as_ptr()));
        }
    }
}

impl<T: TensorType> TensorIndex<Slice> for DenseTensor<T> {
    type Output = DenseTensor<T>;
    fn at(&self, idx: Slice) -> DenseTensor<T> {
        unsafe {
            return DenseTensor::new(cten_slice(self.tensor, 0, idx.start, idx.end, idx.step));
        }
    }
}

impl<'a, T: TensorType> TensorIndex<&'a Slice> for DenseTensor<T> {
    type Output = DenseTensor<T>;
    fn at(&self, idx: &'a Slice) -> DenseTensor<T> {
        unsafe {
            return DenseTensor::new(cten_slice(self.tensor, 0, idx.start, idx.end, idx.step));
        }
    }
}

impl<T: TensorType> TensorIndex<Vec<AdvancedSlice>> for DenseTensor<T> {
    type Output = DenseTensor<T>;
    fn at(&self, indices: Vec<AdvancedSlice>) -> DenseTensor<T> {
        unsafe {
            return DenseTensor::new(cten_advance_access_index(self.tensor, indices.len() as i32, indices.as_ptr()));
        }
    }
}

impl<'a, T: TensorType> TensorIndex<&'a Vec<AdvancedSlice>> for DenseTensor<T> {
    type Output = DenseTensor<T>;
    fn at(&self, indices: &'a Vec<AdvancedSlice>) -> DenseTensor<T> {
        unsafe {
            return DenseTensor::new(cten_advance_access_index(self.tensor, indices.len() as i32, indices.as_ptr()));
        }
    }
}

impl<T: TensorType> TensorIndex<(i64, i64)> for DenseTensor<T> {
    type Output = DenseTensor<T>;

    fn at(&self, idx: (i64, i64)) -> DenseTensor<T> {
        unsafe {
            return DenseTensor::new(cten_select_along_dims(self.tensor, 2, [idx.0, idx.1].as_ptr()));
        }
    }
}

impl<T: TensorType> TensorIndex<(i64, i64, i64)> for DenseTensor<T> {
    type Output = DenseTensor<T>;

    fn at(&self, idx: (i64, i64, i64)) -> DenseTensor<T> {
        unsafe {
            return DenseTensor::new(cten_select_along_dims(self.tensor, 3, [idx.0, idx.1, idx.2].as_ptr()));
        }
    }
}

impl<T: TensorType> TensorAssign<i64, f64> for DenseTensor<T> {
    fn assign(&mut self, idx: i64, val: f64) {
        unsafe { cten_select_fill_(self.tensor, 0, idx, val); }
    }
}
impl<T: TensorType> TensorAssign<i64, DenseTensor<T>> for DenseTensor<T> {
    fn assign(&mut self, idx: i64, val: DenseTensor<T>) {
        unsafe {
            // our default is non-blocking
            cten_select_copy_(self.tensor, 0, idx, val.tensor);
        }
    }
}

impl<T: TensorType> TensorAssign<i64, Vec<T::PrimitiveType>> for DenseTensor<T> {
    fn assign(&mut self, idx: i64, val: Vec<T::PrimitiveType>) {
        unsafe {
            // our default is non-blocking
            cten_select_copy_(self.tensor, 0, idx, DenseTensor::<T>::borrow_from_array(&val).tensor);
        }
    }
}

impl<T: TensorType> TensorAssign<i64, [T::PrimitiveType; 2]> for DenseTensor<T> {
    fn assign(&mut self, idx: i64, val: [T::PrimitiveType; 2]) {
        unsafe {
            // our default is non-blocking
            cten_select_copy_(self.tensor, 0, idx, DenseTensor::<T>::borrow_from_array(&val).tensor);
        }
    }
}

impl<'a, T: TensorType> TensorAssign<i64, &'a DenseTensor<T>> for DenseTensor<T> {
    fn assign(&mut self, idx: i64, val: &'a DenseTensor<T>) {
        unsafe {
            // our default is non-blocking
            cten_select_copy_(self.tensor, 0, idx, val.tensor);
        }
    }
}
impl<'a, T: TensorType> TensorAssign<i64, &'a Vec<T::PrimitiveType>> for DenseTensor<T> {
    fn assign(&mut self, idx: i64, val: &'a Vec<T::PrimitiveType>) {
        unsafe {
            // our default is non-blocking
            cten_select_copy_(self.tensor, 0, idx, DenseTensor::<T>::borrow_from_array(val).tensor);
        }
    }
}

impl<'a, T: TensorType> TensorAssign<&'a Vec<i64>, f64> for DenseTensor<T> {
    fn assign(&mut self, idx: &'a Vec<i64>, val: f64) {
        unsafe {
            cten_index_fill_(self.tensor, 0, DenseTensor::<TLong>::borrow_from_array(idx).tensor, val);
        }
    }
}
impl<'a, T: TensorType> TensorAssign<&'a Vec<i64>, DenseTensor<T>> for DenseTensor<T> {
    fn assign(&mut self, idx: &'a Vec<i64>, val: DenseTensor<T>) {
        unsafe {
            cten_index_copy_(self.tensor, 0,
                                    DenseTensor::<TLong>::borrow_from_array(idx).tensor, val.tensor);
        }
    }
}
impl<'a, T: TensorType> TensorAssign<&'a Vec<i64>, Vec<T::PrimitiveType>> for DenseTensor<T> {
    fn assign(&mut self, idx: &'a Vec<i64>, val: Vec<T::PrimitiveType>) {
        unsafe {
            cten_index_copy_(self.tensor, 0,
                                    DenseTensor::<TLong>::borrow_from_array(idx).tensor,
                                    DenseTensor::<T>::borrow_from_array(&val).tensor);
        }
    }
}
impl<'a, 'a2, T: TensorType> TensorAssign<&'a2 Vec<i64>, &'a DenseTensor<T>> for DenseTensor<T> {
    fn assign(&mut self, idx: &'a2 Vec<i64>, val: &'a DenseTensor<T>) {
        unsafe {
            cten_index_copy_(self.tensor, 0,
                                    DenseTensor::<TLong>::borrow_from_array(idx).tensor, val.tensor);
        }
    }
}
impl<'a, 'a2, T: TensorType> TensorAssign<&'a2 Vec<i64>, &'a Vec<T::PrimitiveType>> for DenseTensor<T> {
    fn assign(&mut self, idx: &'a2 Vec<i64>, val: &'a Vec<T::PrimitiveType>) {
        unsafe {
            cten_index_copy_(self.tensor, 0,
                                    DenseTensor::<TLong>::borrow_from_array(idx).tensor,
                                    DenseTensor::<T>::borrow_from_array(val).tensor);
        }
    }
}

impl<T: TensorType> TensorAssign<Slice, f64> for DenseTensor<T> {
    fn assign(&mut self, idx: Slice, val: f64) {
        unsafe { cten_slice_fill_(self.tensor, 0, idx.start, idx.end, idx.step, val); }
    }
}
impl<T: TensorType> TensorAssign<Slice, DenseTensor<T>> for DenseTensor<T> {
    fn assign(&mut self, idx: Slice, val: DenseTensor<T>) {
        unsafe {
            cten_slice_copy_(self.tensor, 0, idx.start, idx.end, idx.step, val.tensor);
        }
    }
}
impl<T: TensorType> TensorAssign<Slice, Vec<T::PrimitiveType>> for DenseTensor<T> {
    fn assign(&mut self, idx: Slice, val: Vec<T::PrimitiveType>) {
        unsafe {
            cten_slice_copy_(self.tensor, 0, idx.start, idx.end, idx.step, DenseTensor::<T>::borrow_from_array(&val).tensor);
        }
    }
}
impl<'a, T: TensorType> TensorAssign<Slice, &'a DenseTensor<T>> for DenseTensor<T> {
    fn assign(&mut self, idx: Slice, val: &'a DenseTensor<T>) {
        unsafe {
            cten_slice_copy_(self.tensor, 0, idx.start, idx.end, idx.step, val.tensor);
        }
    }
}
impl<'a, T: TensorType> TensorAssign<Slice, &'a Vec<T::PrimitiveType>> for DenseTensor<T> {
    fn assign(&mut self, idx: Slice, val: &'a Vec<T::PrimitiveType>) {
        unsafe {
            cten_slice_copy_(self.tensor, 0, idx.start, idx.end, idx.step, DenseTensor::<T>::borrow_from_array(val).tensor);
        }
    }
}

impl<T: TensorType> TensorAssign<Vec<AdvancedSlice>, f64> for DenseTensor<T> {
    fn assign(&mut self, indices: Vec<AdvancedSlice>, val: f64) {
        unsafe {
            let aten = cten_advance_access_index(self.tensor, indices.len() as i32, indices.as_ptr());
            cten_fill_(aten, val);
            cten_drop_tensor(aten);
        }
    }
}
impl<T: TensorType> TensorAssign<Vec<AdvancedSlice>, DenseTensor<T>> for DenseTensor<T> {
    fn assign(&mut self, indices: Vec<AdvancedSlice>, val: DenseTensor<T>) {
        unsafe {
            let aten = cten_advance_access_index(self.tensor, indices.len() as i32, indices.as_ptr());
            cten_copy_(aten, val.tensor, false);
            cten_drop_tensor(aten);
        }
    }
}
impl<T: TensorType> TensorAssign<Vec<AdvancedSlice>, Vec<T::PrimitiveType>> for DenseTensor<T> {
    fn assign(&mut self, indices: Vec<AdvancedSlice>, val: Vec<T::PrimitiveType>) {
        unsafe {
            let aten = cten_advance_access_index(self.tensor, indices.len() as i32, indices.as_ptr());
            cten_copy_(aten, DenseTensor::<T>::borrow_from_array(&val).tensor, false);
            cten_drop_tensor(aten);
        }
    }
}
impl<'a, T: TensorType> TensorAssign<Vec<AdvancedSlice>, &'a DenseTensor<T>> for DenseTensor<T> {
    fn assign(&mut self, indices: Vec<AdvancedSlice>, val: &'a DenseTensor<T>) {
        unsafe {
            let aten = cten_advance_access_index(self.tensor, indices.len() as i32, indices.as_ptr());
            cten_copy_(aten, val.tensor, false);
            cten_drop_tensor(aten);
        }
    }
}
impl<'a, T: TensorType> TensorAssign<Vec<AdvancedSlice>, &'a Vec<T::PrimitiveType>> for DenseTensor<T> {
    fn assign(&mut self, indices: Vec<AdvancedSlice>, val: &'a Vec<T::PrimitiveType>) {
        unsafe {
            let aten = cten_advance_access_index(self.tensor, indices.len() as i32, indices.as_ptr());
            cten_copy_(aten, DenseTensor::<T>::borrow_from_array(val).tensor, false);
            cten_drop_tensor(aten);
        }
    }
}

impl<T: TensorType> TensorAssign<(i64, i64), f64> for DenseTensor<T> {
    fn assign(&mut self, idx: (i64, i64), val: f64) {
        unsafe {
            cten_select_along_dims_fill_(self.tensor, 2, [idx.0, idx.1].as_ptr(), val);
        }
    }
}
impl<T: TensorType> TensorAssign<(i64, i64), DenseTensor<T>> for DenseTensor<T> {
    fn assign(&mut self, idx: (i64, i64), val: DenseTensor<T>) {
        unsafe {
            cten_select_along_dims_copy_(self.tensor, 2, [idx.0, idx.1].as_ptr(), val.tensor);
        }
    }
}
impl<T: TensorType> TensorAssign<(i64, i64), Vec<T::PrimitiveType>> for DenseTensor<T> {
    fn assign(&mut self, idx: (i64, i64), val: Vec<T::PrimitiveType>) {
        unsafe {
            cten_select_along_dims_copy_(self.tensor, 2, [idx.0, idx.1].as_ptr(), DenseTensor::<T>::borrow_from_array(&val).tensor);
        }
    }
}
impl<'a, T: TensorType> TensorAssign<(i64, i64), &'a DenseTensor<T>> for DenseTensor<T> {
    fn assign(&mut self, idx: (i64, i64), val: &'a DenseTensor<T>) {
        unsafe {
            cten_select_along_dims_copy_(self.tensor, 2, [idx.0, idx.1].as_ptr(), val.tensor);
        }
    }
}
impl<'a, T: TensorType> TensorAssign<(i64, i64), &'a Vec<T::PrimitiveType>> for DenseTensor<T> {
    fn assign(&mut self, idx: (i64, i64), val: &'a Vec<T::PrimitiveType>) {
        unsafe {
            cten_select_along_dims_copy_(self.tensor, 2, [idx.0, idx.1].as_ptr(), DenseTensor::<T>::borrow_from_array(val).tensor);
        }
    }
}

impl<T: TensorType> TensorAssign<(i64, i64, i64), f64> for DenseTensor<T> {
    fn assign(&mut self, idx: (i64, i64, i64), val: f64) {
        unsafe {
            cten_select_along_dims_fill_(self.tensor, 3, [idx.0, idx.1, idx.2].as_ptr(), val);
        }
    }
}
impl<T: TensorType> TensorAssign<(i64, i64, i64), DenseTensor<T>> for DenseTensor<T> {
    fn assign(&mut self, idx: (i64, i64, i64), val: DenseTensor<T>) {
        unsafe {
            cten_select_along_dims_copy_(self.tensor, 3, [idx.0, idx.1, idx.2].as_ptr(), val.tensor);
        }
    }
}
impl<T: TensorType> TensorAssign<(i64, i64, i64), Vec<T::PrimitiveType>> for DenseTensor<T> {
    fn assign(&mut self, idx: (i64, i64, i64), val: Vec<T::PrimitiveType>) {
        unsafe {
            cten_select_along_dims_copy_(self.tensor, 3, [idx.0, idx.1, idx.2].as_ptr(), DenseTensor::<T>::borrow_from_array(&val).tensor);
        }
    }
}
impl<'a, T: TensorType> TensorAssign<(i64, i64, i64), &'a DenseTensor<T>> for DenseTensor<T> {
    fn assign(&mut self, idx: (i64, i64, i64), val: &'a DenseTensor<T>) {
        unsafe {
            cten_select_along_dims_copy_(self.tensor, 3, [idx.0, idx.1, idx.2].as_ptr(), val.tensor);
        }
    }
}
impl<'a, T: TensorType> TensorAssign<(i64, i64, i64), &'a Vec<T::PrimitiveType>> for DenseTensor<T> {
    fn assign(&mut self, idx: (i64, i64, i64), val: &'a Vec<T::PrimitiveType>) {
        unsafe {
            cten_select_along_dims_copy_(self.tensor, 3, [idx.0, idx.1, idx.2].as_ptr(), DenseTensor::<T>::borrow_from_array(val).tensor);
        }
    }
}