use super::tensor::DenseTensor;
use tensors::tensor_type::TensorType;
use super::c_api::*;
use std::ops::*;

impl<T: TensorType> PartialEq for DenseTensor<T> {
    fn eq(&self, other: &DenseTensor<T>) -> bool {
        unsafe {
            return cten_equal(self.tensor, other.tensor);
        }
    }
}
impl<T: TensorType> Eq for DenseTensor<T> {}

impl<'a, T: TensorType> Neg for &'a DenseTensor<T> {
    type Output = DenseTensor<T>;

    fn neg(self) -> DenseTensor<T> {
        unsafe {
            return DenseTensor::new(cten_neg(self.tensor));
        }
    }
}
impl<T: TensorType> Neg for DenseTensor<T> {
    type Output = DenseTensor<T>;

    fn neg(self) -> DenseTensor<T> {
        unsafe {
            return DenseTensor::new(cten_neg(self.tensor));
        }
    }
}

/// ********* OpsAssign definition goes here

impl<T: TensorType> AddAssign<f64> for DenseTensor<T> {
    fn add_assign(&mut self, val: f64) {
        unsafe { cten_add_v_(self.tensor, val); }
    }
}
impl<T: TensorType> SubAssign<f64> for DenseTensor<T> {
    fn sub_assign(&mut self, val: f64) {
        unsafe { cten_sub_v_(self.tensor, val); }
    }
}
impl<T: TensorType> MulAssign<f64> for DenseTensor<T> {
    fn mul_assign(&mut self, val: f64) {
        unsafe { cten_mul_v_(self.tensor, val); }
    }
}
impl<T: TensorType> DivAssign<f64> for DenseTensor<T> {
    fn div_assign(&mut self, val: f64) {
        unsafe { cten_div_v_(self.tensor, val); }
    }
}

impl<'a, T: TensorType> AddAssign<&'a DenseTensor<T>> for DenseTensor<T> {
    fn add_assign(&mut self, tensor: &'a DenseTensor<T>) {
        unsafe { cten_add_t_(self.tensor, tensor.tensor); }
    }
}
impl<'a, T: TensorType> SubAssign<&'a DenseTensor<T>> for DenseTensor<T> {
    fn sub_assign(&mut self, tensor: &'a DenseTensor<T>) {
        unsafe { cten_sub_t_(self.tensor, tensor.tensor); }
    }
}
impl<'a, T: TensorType> MulAssign<&'a DenseTensor<T>> for DenseTensor<T> {
    fn mul_assign(&mut self, tensor: &'a DenseTensor<T>) {
        unsafe { cten_mul_t_(self.tensor, tensor.tensor); }
    }
}
impl<'a, T: TensorType> DivAssign<&'a DenseTensor<T>> for DenseTensor<T> {
    fn div_assign(&mut self, tensor: &'a DenseTensor<T>) {
        unsafe { cten_div_t_(self.tensor, tensor.tensor); }
    }
}

impl<T: TensorType> AddAssign<DenseTensor<T>> for DenseTensor<T> {
    fn add_assign(&mut self, tensor: DenseTensor<T>) {
        unsafe { cten_add_t_(self.tensor, tensor.tensor); }
    }
}
impl<T: TensorType> SubAssign<DenseTensor<T>> for DenseTensor<T> {
    fn sub_assign(&mut self, tensor: DenseTensor<T>) {
        unsafe { cten_sub_t_(self.tensor, tensor.tensor); }
    }
}
impl<T: TensorType> MulAssign<DenseTensor<T>> for DenseTensor<T> {
    fn mul_assign(&mut self, tensor: DenseTensor<T>) {
        unsafe { cten_mul_t_(self.tensor, tensor.tensor); }
    }
}
impl<T: TensorType> DivAssign<DenseTensor<T>> for DenseTensor<T> {
    fn div_assign(&mut self, tensor: DenseTensor<T>) {
        unsafe { cten_div_t_(self.tensor, tensor.tensor); }
    }
}

/// ********* Ops definition goes here

impl<'a, 'b, T: TensorType> Add<&'b DenseTensor<T>> for &'a DenseTensor<T> {
    type Output = DenseTensor<T>;

    fn add(self, tensor: &'b DenseTensor<T>) -> DenseTensor<T> {
        unsafe {
            return DenseTensor::new(cten_add_t(self.tensor, tensor.tensor));
        }
    }
}
impl<'a, 'b, T: TensorType> Sub<&'b DenseTensor<T>> for &'a DenseTensor<T> {
    type Output = DenseTensor<T>;

    fn sub(self, tensor: &'b DenseTensor<T>) -> DenseTensor<T> {
        unsafe {
            return DenseTensor::new(cten_sub_t(self.tensor, tensor.tensor));
        }
    }
}
impl<'a, 'b, T: TensorType> Mul<&'b DenseTensor<T>> for &'a DenseTensor<T> {
    type Output = DenseTensor<T>;

    fn mul(self, tensor: &'b DenseTensor<T>) -> DenseTensor<T> {
        unsafe {
            return DenseTensor::new(cten_mul_t(self.tensor, tensor.tensor));
        }
    }
}
impl<'a, 'b, T: TensorType> Div<&'b DenseTensor<T>> for &'a DenseTensor<T> {
    type Output = DenseTensor<T>;

    fn div(self, tensor: &'b DenseTensor<T>) -> DenseTensor<T> {
        unsafe {
            return DenseTensor::new(cten_div_t(self.tensor, tensor.tensor));
        }
    }
}

impl<'a, T: TensorType> Add<&'a DenseTensor<T>> for DenseTensor<T> {
    type Output = DenseTensor<T>;

    fn add(self, tensor: &'a DenseTensor<T>) -> DenseTensor<T> {
        unsafe {
            return DenseTensor::new(cten_add_t(self.tensor, tensor.tensor));
        }
    }
}
impl<'a, T: TensorType> Sub<&'a DenseTensor<T>> for DenseTensor<T> {
    type Output = DenseTensor<T>;

    fn sub(self, tensor: &'a DenseTensor<T>) -> DenseTensor<T> {
        unsafe {
            return DenseTensor::new(cten_sub_t(self.tensor, tensor.tensor));
        }
    }
}
impl<'a, T: TensorType> Mul<&'a DenseTensor<T>> for DenseTensor<T> {
    type Output = DenseTensor<T>;

    fn mul(self, tensor: &'a DenseTensor<T>) -> DenseTensor<T> {
        unsafe {
            return DenseTensor::new(cten_mul_t(self.tensor, tensor.tensor));
        }
    }
}
impl<'a, T: TensorType> Div<&'a DenseTensor<T>> for DenseTensor<T> {
    type Output = DenseTensor<T>;

    fn div(self, tensor: &'a DenseTensor<T>) -> DenseTensor<T> {
        unsafe {
            return DenseTensor::new(cten_div_t(self.tensor, tensor.tensor));
        }
    }
}

impl<'a, 'b, T: TensorType> Add<&'b mut DenseTensor<T>> for &'a DenseTensor<T> {
    type Output = DenseTensor<T>;

    fn add(self, tensor: &'b mut DenseTensor<T>) -> DenseTensor<T> {
        unsafe {
            return DenseTensor::new(cten_add_t(self.tensor, tensor.tensor));
        }
    }
}
impl<'a, 'b, T: TensorType> Sub<&'b mut DenseTensor<T>> for &'a DenseTensor<T> {
    type Output = DenseTensor<T>;

    fn sub(self, tensor: &'b mut DenseTensor<T>) -> DenseTensor<T> {
        unsafe {
            return DenseTensor::new(cten_sub_t(self.tensor, tensor.tensor));
        }
    }
}
impl<'a, 'b, T: TensorType> Mul<&'b mut DenseTensor<T>> for &'a DenseTensor<T> {
    type Output = DenseTensor<T>;

    fn mul(self, tensor: &'b mut DenseTensor<T>) -> DenseTensor<T> {
        unsafe {
            return DenseTensor::new(cten_mul_t(self.tensor, tensor.tensor));
        }
    }
}
impl<'a, 'b, T: TensorType> Div<&'b mut DenseTensor<T>> for &'a DenseTensor<T> {
    type Output = DenseTensor<T>;

    fn div(self, tensor: &'b mut DenseTensor<T>) -> DenseTensor<T> {
        unsafe {
            return DenseTensor::new(cten_div_t(self.tensor, tensor.tensor));
        }
    }
}

impl<'a, T: TensorType> Add<&'a mut DenseTensor<T>> for DenseTensor<T> {
    type Output = DenseTensor<T>;

    fn add(self, tensor: &'a mut DenseTensor<T>) -> DenseTensor<T> {
        unsafe {
            return DenseTensor::new(cten_add_t(self.tensor, tensor.tensor));
        }
    }
}
impl<'a, T: TensorType> Sub<&'a mut DenseTensor<T>> for DenseTensor<T> {
    type Output = DenseTensor<T>;

    fn sub(self, tensor: &'a mut DenseTensor<T>) -> DenseTensor<T> {
        unsafe {
            return DenseTensor::new(cten_sub_t(self.tensor, tensor.tensor));
        }
    }
}
impl<'a, T: TensorType> Mul<&'a mut DenseTensor<T>> for DenseTensor<T> {
    type Output = DenseTensor<T>;

    fn mul(self, tensor: &'a mut DenseTensor<T>) -> DenseTensor<T> {
        unsafe {
            return DenseTensor::new(cten_mul_t(self.tensor, tensor.tensor));
        }
    }
}
impl<'a, T: TensorType> Div<&'a mut DenseTensor<T>> for DenseTensor<T> {
    type Output = DenseTensor<T>;

    fn div(self, tensor: &'a mut DenseTensor<T>) -> DenseTensor<T> {
        unsafe {
            return DenseTensor::new(cten_div_t(self.tensor, tensor.tensor));
        }
    }
}

impl<'a, T: TensorType> Add<DenseTensor<T>> for &'a DenseTensor<T> {
    type Output = DenseTensor<T>;

    fn add(self, tensor: DenseTensor<T>) -> DenseTensor<T> {
        unsafe {
            return DenseTensor::new(cten_add_t(self.tensor, tensor.tensor));
        }
    }
}
impl<'a, T: TensorType> Sub<DenseTensor<T>> for &'a DenseTensor<T> {
    type Output = DenseTensor<T>;

    fn sub(self, tensor: DenseTensor<T>) -> DenseTensor<T> {
        unsafe {
            return DenseTensor::new(cten_sub_t(self.tensor, tensor.tensor));
        }
    }
}
impl<'a, T: TensorType> Mul<DenseTensor<T>> for &'a DenseTensor<T> {
    type Output = DenseTensor<T>;

    fn mul(self, tensor: DenseTensor<T>) -> DenseTensor<T> {
        unsafe {
            return DenseTensor::new(cten_mul_t(self.tensor, tensor.tensor));
        }
    }
}
impl<'a, T: TensorType> Div<DenseTensor<T>> for &'a DenseTensor<T> {
    type Output = DenseTensor<T>;

    fn div(self, tensor: DenseTensor<T>) -> DenseTensor<T> {
        unsafe {
            return DenseTensor::new(cten_div_t(self.tensor, tensor.tensor));
        }
    }
}

impl<T: TensorType> Add<DenseTensor<T>> for DenseTensor<T> {
    type Output = DenseTensor<T>;

    fn add(self, tensor: DenseTensor<T>) -> DenseTensor<T> {
        unsafe {
            return DenseTensor::new(cten_add_t(self.tensor, tensor.tensor));
        }
    }
}
impl<T: TensorType> Sub<DenseTensor<T>> for DenseTensor<T> {
    type Output = DenseTensor<T>;

    fn sub(self, tensor: DenseTensor<T>) -> DenseTensor<T> {
        unsafe {
            return DenseTensor::new(cten_sub_t(self.tensor, tensor.tensor));
        }
    }
}
impl<T: TensorType> Mul<DenseTensor<T>> for DenseTensor<T> {
    type Output = DenseTensor<T>;

    fn mul(self, tensor: DenseTensor<T>) -> DenseTensor<T> {
        unsafe {
            return DenseTensor::new(cten_mul_t(self.tensor, tensor.tensor));
        }
    }
}
impl<T: TensorType> Div<DenseTensor<T>> for DenseTensor<T> {
    type Output = DenseTensor<T>;

    fn div(self, tensor: DenseTensor<T>) -> DenseTensor<T> {
        unsafe {
            return DenseTensor::new(cten_div_t(self.tensor, tensor.tensor));
        }
    }
}

impl<'a, T: TensorType> Add<&'a DenseTensor<T>> for f64 {
    type Output = DenseTensor<T>;

    fn add(self, tensor: &'a DenseTensor<T>) -> DenseTensor<T> {
        unsafe {
            return DenseTensor::new(cten_add_v(tensor.tensor, self));
        }
    }
}
impl<'a, T: TensorType> Sub<&'a DenseTensor<T>> for f64 {
    type Output = DenseTensor<T>;

    fn sub(self, tensor: &'a DenseTensor<T>) -> DenseTensor<T> {
        unsafe {
            return DenseTensor::new(cten_rsub_v(tensor.tensor, self));
        }
    }
}
impl<'a, T: TensorType> Mul<&'a DenseTensor<T>> for f64 {
    type Output = DenseTensor<T>;

    fn mul(self, tensor: &'a DenseTensor<T>) -> DenseTensor<T> {
        unsafe {
            return DenseTensor::new(cten_mul_v(tensor.tensor, self));
        }
    }
}
impl<'a, T: TensorType> Div<&'a DenseTensor<T>> for f64 {
    type Output = DenseTensor<T>;

    fn div(self, tensor: &'a DenseTensor<T>) -> DenseTensor<T> {
        unsafe {
            return DenseTensor::new(cten_rdiv(self, tensor.tensor));
        }
    }
}

impl<'a, T: TensorType> Add<&'a mut DenseTensor<T>> for f64 {
    type Output = DenseTensor<T>;

    fn add(self, tensor: &'a mut DenseTensor<T>) -> DenseTensor<T> {
        unsafe {
            return DenseTensor::new(cten_add_v(tensor.tensor, self));
        }
    }
}
impl<'a, T: TensorType> Sub<&'a mut DenseTensor<T>> for f64 {
    type Output = DenseTensor<T>;

    fn sub(self, tensor: &'a mut DenseTensor<T>) -> DenseTensor<T> {
        unsafe {
            return DenseTensor::new(cten_rsub_v(tensor.tensor, self));
        }
    }
}
impl<'a, T: TensorType> Mul<&'a mut DenseTensor<T>> for f64 {
    type Output = DenseTensor<T>;

    fn mul(self, tensor: &'a mut DenseTensor<T>) -> DenseTensor<T> {
        unsafe {
            return DenseTensor::new(cten_mul_v(tensor.tensor, self));
        }
    }
}
impl<'a, T: TensorType> Div<&'a mut DenseTensor<T>> for f64 {
    type Output = DenseTensor<T>;

    fn div(self, tensor: &'a mut DenseTensor<T>) -> DenseTensor<T> {
        unsafe {
            return DenseTensor::new(cten_rdiv(self, tensor.tensor));
        }
    }
}

impl<T: TensorType> Add<DenseTensor<T>> for f64 {
    type Output = DenseTensor<T>;

    fn add(self, tensor: DenseTensor<T>) -> DenseTensor<T> {
        unsafe {
            return DenseTensor::new(cten_add_v(tensor.tensor, self));
        }
    }
}
impl<T: TensorType> Sub<DenseTensor<T>> for f64 {
    type Output = DenseTensor<T>;

    fn sub(self, tensor: DenseTensor<T>) -> DenseTensor<T> {
        unsafe {
            return DenseTensor::new(cten_rsub_v(tensor.tensor, self));
        }
    }
}
impl<T: TensorType> Mul<DenseTensor<T>> for f64 {
    type Output = DenseTensor<T>;

    fn mul(self, tensor: DenseTensor<T>) -> DenseTensor<T> {
        unsafe {
            return DenseTensor::new(cten_mul_v(tensor.tensor, self));
        }
    }
}
impl<T: TensorType> Div<DenseTensor<T>> for f64 {
    type Output = DenseTensor<T>;

    fn div(self, tensor: DenseTensor<T>) -> DenseTensor<T> {
        unsafe {
            return DenseTensor::new(cten_rdiv(self, tensor.tensor));
        }
    }
}

impl<'a, T: TensorType> Add<f64> for &'a DenseTensor<T> {
    type Output = DenseTensor<T>;

    fn add(self, val: f64) -> DenseTensor<T> {
        unsafe {
            return DenseTensor::new(cten_add_v(self.tensor, val));
        }
    }
}
impl<'a, T: TensorType> Sub<f64> for &'a DenseTensor<T> {
    type Output = DenseTensor<T>;

    fn sub(self, val: f64) -> DenseTensor<T> {
        unsafe {
            return DenseTensor::new(cten_sub_v(self.tensor, val));
        }
    }
}
impl<'a, T: TensorType> Mul<f64> for &'a DenseTensor<T> {
    type Output = DenseTensor<T>;

    fn mul(self, val: f64) -> DenseTensor<T> {
        unsafe {
            return DenseTensor::new(cten_mul_v(self.tensor, val));
        }
    }
}
impl<'a, T: TensorType> Div<f64> for &'a DenseTensor<T> {
    type Output = DenseTensor<T>;

    fn div(self, val: f64) -> DenseTensor<T> {
        unsafe {
            return DenseTensor::new(cten_div_v(self.tensor, val));
        }
    }
}

impl<T: TensorType> Add<f64> for DenseTensor<T> {
    type Output = DenseTensor<T>;

    fn add(self, val: f64) -> DenseTensor<T> {
        unsafe {
            return DenseTensor::new(cten_add_v(self.tensor, val));
        }
    }
}
impl<T: TensorType> Sub<f64> for DenseTensor<T> {
    type Output = DenseTensor<T>;

    fn sub(self, val: f64) -> DenseTensor<T> {
        unsafe {
            return DenseTensor::new(cten_sub_v(self.tensor, val));
        }
    }
}
impl<T: TensorType> Mul<f64> for DenseTensor<T> {
    type Output = DenseTensor<T>;

    fn mul(self, val: f64) -> DenseTensor<T> {
        unsafe {
            return DenseTensor::new(cten_mul_v(self.tensor, val));
        }
    }
}
impl<T: TensorType> Div<f64> for DenseTensor<T> {
    type Output = DenseTensor<T>;

    fn div(self, val: f64) -> DenseTensor<T> {
        unsafe {
            return DenseTensor::new(cten_div_v(self.tensor, val));
        }
    }
}

impl<'a, T: TensorType> Add<&'a DenseTensor<T>> for f32 {
    type Output = DenseTensor<T>;

    fn add(self, tensor: &'a DenseTensor<T>) -> DenseTensor<T> {
        unsafe {
            return DenseTensor::new(cten_add_v(tensor.tensor, self as f64));
        }
    }
}
impl<'a, T: TensorType> Sub<&'a DenseTensor<T>> for f32 {
    type Output = DenseTensor<T>;

    fn sub(self, tensor: &'a DenseTensor<T>) -> DenseTensor<T> {
        unsafe {
            return DenseTensor::new(cten_rsub_v(tensor.tensor, self as f64));
        }
    }
}
impl<'a, T: TensorType> Mul<&'a DenseTensor<T>> for f32 {
    type Output = DenseTensor<T>;

    fn mul(self, tensor: &'a DenseTensor<T>) -> DenseTensor<T> {
        unsafe {
            return DenseTensor::new(cten_mul_v(tensor.tensor, self as f64));
        }
    }
}
impl<'a, T: TensorType> Div<&'a DenseTensor<T>> for f32 {
    type Output = DenseTensor<T>;

    fn div(self, tensor: &'a DenseTensor<T>) -> DenseTensor<T> {
        unsafe {
            return DenseTensor::new(cten_rdiv(self as f64, tensor.tensor));
        }
    }
}

impl<T: TensorType> Add<DenseTensor<T>> for f32 {
    type Output = DenseTensor<T>;

    fn add(self, tensor: DenseTensor<T>) -> DenseTensor<T> {
        unsafe {
            return DenseTensor::new(cten_add_v(tensor.tensor, self as f64));
        }
    }
}
impl<T: TensorType> Sub<DenseTensor<T>> for f32 {
    type Output = DenseTensor<T>;

    fn sub(self, tensor: DenseTensor<T>) -> DenseTensor<T> {
        unsafe {
            return DenseTensor::new(cten_rsub_v(tensor.tensor, self as f64));
        }
    }
}
impl<T: TensorType> Mul<DenseTensor<T>> for f32 {
    type Output = DenseTensor<T>;

    fn mul(self, tensor: DenseTensor<T>) -> DenseTensor<T> {
        unsafe {
            return DenseTensor::new(cten_mul_v(tensor.tensor, self as f64));
        }
    }
}
impl<T: TensorType> Div<DenseTensor<T>> for f32 {
    type Output = DenseTensor<T>;

    fn div(self, tensor: DenseTensor<T>) -> DenseTensor<T> {
        unsafe {
            return DenseTensor::new(cten_rdiv(self as f64, tensor.tensor));
        }
    }
}

impl<'a, T: TensorType> Add<f32> for &'a DenseTensor<T> {
    type Output = DenseTensor<T>;

    fn add(self, val: f32) -> DenseTensor<T> {
        unsafe {
            return DenseTensor::new(cten_add_v(self.tensor, val as f64));
        }
    }
}
impl<'a, T: TensorType> Sub<f32> for &'a DenseTensor<T> {
    type Output = DenseTensor<T>;

    fn sub(self, val: f32) -> DenseTensor<T> {
        unsafe {
            return DenseTensor::new(cten_sub_v(self.tensor, val as f64));
        }
    }
}
impl<'a, T: TensorType> Mul<f32> for &'a DenseTensor<T> {
    type Output = DenseTensor<T>;

    fn mul(self, val: f32) -> DenseTensor<T> {
        unsafe {
            return DenseTensor::new(cten_mul_v(self.tensor, val as f64));
        }
    }
}
impl<'a, T: TensorType> Div<f32> for &'a DenseTensor<T> {
    type Output = DenseTensor<T>;

    fn div(self, val: f32) -> DenseTensor<T> {
        unsafe {
            return DenseTensor::new(cten_div_v(self.tensor, val as f64));
        }
    }
}

impl<T: TensorType> Add<f32> for DenseTensor<T> {
    type Output = DenseTensor<T>;

    fn add(self, val: f32) -> DenseTensor<T> {
        unsafe {
            return DenseTensor::new(cten_add_v(self.tensor, val as f64));
        }
    }
}
impl<T: TensorType> Sub<f32> for DenseTensor<T> {
    type Output = DenseTensor<T>;

    fn sub(self, val: f32) -> DenseTensor<T> {
        unsafe {
            return DenseTensor::new(cten_sub_v(self.tensor, val as f64));
        }
    }
}
impl<T: TensorType> Mul<f32> for DenseTensor<T> {
    type Output = DenseTensor<T>;

    fn mul(self, val: f32) -> DenseTensor<T> {
        unsafe {
            return DenseTensor::new(cten_mul_v(self.tensor, val as f64));
        }
    }
}
impl<T: TensorType> Div<f32> for DenseTensor<T> {
    type Output = DenseTensor<T>;

    fn div(self, val: f32) -> DenseTensor<T> {
        unsafe {
            return DenseTensor::new(cten_div_v(self.tensor, val as f64));
        }
    }
}