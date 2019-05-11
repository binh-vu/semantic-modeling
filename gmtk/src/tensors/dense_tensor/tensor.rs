use std;
use super::c_api::*;
use num_traits::*;
use tensors::tensor_type::*;
use std::marker::PhantomData;
use std::fmt;
use std::ffi::CStr;
use libc::c_void;
use std::iter::Sum;

pub struct DenseTensor<T: TensorType=TDefault> {
    /// A Rust wrapper for DenseTensor
    pub(super) tensor: *const ATensor,
    _marker: std::marker::PhantomData<*const T>
}

impl<T: TensorType> DenseTensor<T> {

    #[inline]
    pub(super) fn new(tensor: *const ATensor) -> DenseTensor<T> {
        return DenseTensor { tensor, _marker: PhantomData };
    }

    pub fn manual_seed(seed: u64) {
        unsafe {
            cten_manual_seed(seed, DEFAULT_BACKEND);
        }
    }

    pub fn create(shape: &[i64]) -> DenseTensor<T> {
        unsafe {
            return DenseTensor { tensor: cten_create(shape.len(), shape.as_ptr(), T::get_dtype(), DEFAULT_BACKEND), _marker: PhantomData };
        }
    }

    pub fn create_randn(shape: &[i64]) -> DenseTensor<T> {
        unsafe {
            return DenseTensor { tensor: cten_create_randn(shape.len(), shape.as_ptr(), T::get_dtype(), DEFAULT_BACKEND), _marker: PhantomData };
        }
    }

    pub fn zeros(shape: &[i64]) -> DenseTensor<T> {
        unsafe {
            return DenseTensor { tensor: cten_zeros(shape.len(), shape.as_ptr(), T::get_dtype(), DEFAULT_BACKEND), _marker: PhantomData };
        }
    }

    pub fn ones(shape: &[i64]) -> DenseTensor<T> {
        unsafe {
            return DenseTensor { tensor: cten_ones(shape.len(), shape.as_ptr(), T::get_dtype(), DEFAULT_BACKEND), _marker: PhantomData };
        }
    }

    pub fn zeros_like(tensor: &DenseTensor<T>) -> DenseTensor<T> {
        unsafe {
            return DenseTensor { tensor: cten_zeros_like(tensor.tensor, T::get_dtype(), DEFAULT_BACKEND), _marker: PhantomData };
        }
    }

    pub fn ones_like(tensor: &DenseTensor<T>) -> DenseTensor<T> {
        unsafe {
            return DenseTensor { tensor: cten_ones_like(tensor.tensor, T::get_dtype(), DEFAULT_BACKEND), _marker: PhantomData };
        }
    }

    pub fn from_array(data: &[T::PrimitiveType]) -> DenseTensor<T> {
        unsafe {
            return DenseTensor { tensor: cten_from_array(data.len() as i64, data.as_ptr() as *const c_void, T::get_dtype(), DEFAULT_BACKEND), _marker: PhantomData };
        }
    }

    pub fn borrow_from_array(data: &[T::PrimitiveType]) -> DenseTensor<T> {
        unsafe {
            return DenseTensor::new(cten_borrow_from_array(data.len() as i64, data.as_ptr() as *const c_void, T::get_dtype(), Backend::CPU as i32));
        }
    }

    pub fn from_ndarray(data: &[T::PrimitiveType], shape: &[i64]) -> DenseTensor<T> {
        unsafe {
            return DenseTensor { tensor: cten_from_array_and_view(data.len() as i64, shape.len(), shape.as_ptr(), data.as_ptr() as *const c_void, T::get_dtype(), DEFAULT_BACKEND), _marker: PhantomData };
        }
    }

    pub fn stack_ref(tensors: &Vec<&DenseTensor<T>>, dim: i64) -> DenseTensor<T> {
        unsafe {
            let ptr: Vec<*const ATensor> = tensors.iter().map(|t| t.tensor).collect();
            let tensor = cten_stack(tensors.len(), ptr.as_ptr(), dim);
            return DenseTensor { tensor, _marker: PhantomData };
        }
    }

    pub fn stack(tensors: &Vec<DenseTensor<T>>, dim: i64) -> DenseTensor<T> {
        unsafe {
            let ptr: Vec<*const ATensor> = tensors.iter().map(|t| t.tensor).collect();
            let tensor = cten_stack(tensors.len(), ptr.as_ptr(), dim);
            return DenseTensor { tensor, _marker: PhantomData };
        }
    }

    pub fn concat(tensors: &Vec<DenseTensor<T>>, dim: i64) -> DenseTensor<T> {
        unsafe {
            let ptr: Vec<*const ATensor> = tensors.iter().map(|t| t.tensor).collect();
            let tensor = cten_concat(tensors.len(), ptr.as_ptr(), dim);
            return DenseTensor { tensor, _marker: PhantomData };
        }
    }

    pub fn clone_reference(&self) -> DenseTensor<T> {
        unsafe {
            return DenseTensor { tensor: cten_create_reference(self.tensor), _marker: PhantomData }
        }
    }

    pub fn zero_(&mut self) {
        unsafe { cten_zero_(self.tensor); }
    }

    /// Copies the elements from src into self tensor.
    pub fn copy_(&self, another: &DenseTensor<T>) {
        unsafe {
            cten_copy_(self.tensor, another.tensor, false);
        }
    }

    /// Sum all elements in tensor
    pub fn sum(&self) -> DenseTensor<T> {
        unsafe {
            return DenseTensor { tensor: cten_sum(self.tensor), _marker: PhantomData };
        }
    }

    /// Returns the sum of each row of the input tensor in the given dimension dim.
    pub fn sum_along_dim(&self, dim: i32, keepdim: bool) -> DenseTensor<T> {
        unsafe {
            return DenseTensor { tensor: cten_sum_along_dim(self.tensor, dim, keepdim), _marker: PhantomData };
        }
    }

    pub fn pow(&self, exp: i32) -> DenseTensor<T> {
        unsafe {
            return DenseTensor { tensor: cten_pow_tensor(self.tensor, exp), _marker: PhantomData };
        }
    }

    pub fn exp(&self) -> DenseTensor<T> {
        unsafe {
            return DenseTensor { tensor: cten_exp_tensor(self.tensor), _marker: PhantomData };
        }
    }

    pub fn log(&self) -> DenseTensor<T> {
        unsafe {
            return DenseTensor { tensor: cten_log_tensor(self.tensor), _marker: PhantomData };
        }
    }

    pub fn max(&self) -> DenseTensor<T> {
        unsafe {
            return DenseTensor { tensor: cten_max(self.tensor), _marker: PhantomData };
        }
    }

    pub fn max_w_tensor(&self, other: &DenseTensor<T>) -> DenseTensor<T> {
        unsafe {
            return DenseTensor { tensor: cten_max_w_tensor(self.tensor, other.tensor), _marker: PhantomData };
        }
    }

    pub fn max_in_dim(&self, dim: i32, keepdim: bool) -> (DenseTensor<T>, DenseTensor<TLong>) {
        unsafe {
            let res = cten_max_in_dim(self.tensor, dim, keepdim);
            return (
                DenseTensor { tensor: res.first, _marker: PhantomData },
                DenseTensor { tensor: res.second, _marker: PhantomData }
            );
        }
    }

    pub fn transpose(&self) -> DenseTensor<T> {
        unsafe {
            return DenseTensor { tensor: cten_transpose(self.tensor), _marker: PhantomData };
        }
    }

    pub fn swap_axes(&self, dim1: i64, dim2: i64) -> DenseTensor<T> {
        unsafe {
            return DenseTensor { tensor: cten_swap_axes(self.tensor, dim1, dim2), _marker: PhantomData };
        }
    }

    pub fn dot(&self, another: &DenseTensor<T>) -> DenseTensor<T> {
        unsafe {
            return DenseTensor { tensor: cten_dot(self.tensor, another.tensor), _marker: PhantomData };
        }
    }

    pub fn outer(&self, another: &DenseTensor<T>) -> DenseTensor<T> {
        unsafe {
            return DenseTensor { tensor: cten_outer(self.tensor, another.tensor), _marker: PhantomData };
        }
    }

    pub fn matmul(&self, another: &DenseTensor<T>) -> DenseTensor<T> {
        unsafe {
            return DenseTensor { tensor: cten_matmul(self.tensor, another.tensor), _marker: PhantomData };
        }
    }

    pub fn mm(&self, another: &DenseTensor<T>) -> DenseTensor<T> {
        unsafe {
            return DenseTensor { tensor: cten_mm(self.tensor, another.tensor), _marker: PhantomData };
        }
    }

    pub fn mv(&self, another: &DenseTensor<T>) -> DenseTensor<T> {
        unsafe {
            return DenseTensor { tensor: cten_mv(self.tensor, another.tensor), _marker: PhantomData };
        }
    }

    pub fn expand(&self, shape: &Vec<i64>) -> DenseTensor<T> {
        unsafe {
            return DenseTensor { tensor: cten_expand(self.tensor, shape.len(), shape.as_ptr()), _marker: PhantomData };
        }
    }

    pub fn view(&self, shape: &[i64]) -> DenseTensor<T> {
        unsafe {
            return DenseTensor { tensor: cten_view(self.tensor, shape.len(), shape.as_ptr()), _marker: PhantomData };
        }
    }

    pub fn view_(&mut self, shape: Vec<i64>) {
        unsafe {
            let _tensor = cten_view(self.tensor, shape.len(), shape.as_ptr());
            cten_drop_tensor(self.tensor);
            self.tensor = _tensor;
        }
    }

    pub fn view1(&self) -> Self {
        unsafe {
            return DenseTensor { tensor: cten_view1(self.tensor), _marker: PhantomData };
        }
    }

    pub fn view1_(mut self) -> Self {
        unsafe {
            let _tensor = cten_view1(self.tensor);
            cten_drop_tensor(self.tensor);
            self.tensor = _tensor;
        }

        self
    }

    pub fn unbind(&self, dim: i64) -> Vec<DenseTensor<T>> {
        unsafe {
            let dim_size = cten_size_in_dim(self.tensor, dim);
            let mut vec = Vec::with_capacity(dim_size as usize);

            for i in 0..dim_size {
                vec.push(DenseTensor { tensor: cten_select(self.tensor, dim, i), _marker: PhantomData });
            }

            return vec;
        }
    }

    pub fn sigmoid(&self) -> DenseTensor<T> {
        unsafe {
            return DenseTensor { tensor: cten_sigmoid(self.tensor), _marker: PhantomData };
        }
    }

    pub fn sigmoid_(&self) {
        unsafe { cten_sigmoid_(self.tensor); }
    }

    pub fn log_sum_exp(&self) -> DenseTensor<T> {
        unsafe {
            return DenseTensor { tensor: cten_log_sum_exp(self.tensor), _marker: PhantomData };
        }
    }

    /// Default of dim should be 1
    pub fn log_sum_exp_2dim(&self, dim: i32) -> DenseTensor<T> {
        unsafe {
            return DenseTensor { tensor: cten_log_sum_exp_2dim(self.tensor, dim), _marker: PhantomData };
        }
    }

    pub fn cuda(&self) -> DenseTensor<T> {
        unsafe {
            return DenseTensor { tensor: cten_cuda(self.tensor), _marker: PhantomData };
        }
    }

    pub fn cuda_(&mut self) {
        unsafe {
            let _tensor = cten_cuda(self.tensor);
            cten_drop_tensor(self.tensor);
            self.tensor = _tensor;
        }
    }

    pub fn cpu(&self) -> DenseTensor<T> {
        unsafe {
            return DenseTensor { tensor: cten_cpu(self.tensor), _marker: PhantomData };
        }
    }

    pub fn cpu_(&mut self) {
        unsafe {
            let _tensor = cten_cpu(self.tensor);
            cten_drop_tensor(self.tensor);
            self.tensor = _tensor;
        }
    }

    pub fn is_cuda(&self) -> bool {
        unsafe { return cten_is_cuda(self.tensor); }
    }

    pub fn is_contiguous(&self) -> bool {
        unsafe { return cten_is_contiguous(self.tensor); }
    }

    pub fn contiguous_(&mut self) -> &mut Self {
        unsafe {
            let tensor = cten_contiguous(self.tensor);
            cten_drop_tensor(self.tensor);
            self.tensor = tensor;
        }
        self
    }

    #[inline]
    pub fn numel(&self) -> i64 {
        unsafe { cten_numel(self.tensor) }
    }

    pub fn size(&self) -> &[i64] {
        unsafe {
            let shape = cten_size(self.tensor);
            return std::slice::from_raw_parts(shape.dimensions, shape.ndim);
        }
    }

    #[inline]
    pub fn size_in_dim(&self, dim: i64) -> i64 {
        unsafe { cten_size_in_dim(self.tensor, dim) }
    }

    #[inline]
    pub fn ndim(&self) -> i64 {
        unsafe {
            return cten_ndim(self.tensor);
        }
    }

    pub fn squeeze(&self) -> DenseTensor<T> {
        unsafe {
            return DenseTensor { tensor: cten_squeeze(self.tensor), _marker: PhantomData };
        }
    }

    pub fn get_i32(&self) -> i32 {
        unsafe { return cten_get_i32(self.tensor); }
    }

    pub fn get_i64(&self) -> i64 {
        unsafe { return cten_get_i64(self.tensor); }
    }

    pub fn get_f32(&self) -> f32 {
        unsafe { return cten_get_f32(self.tensor); }
    }

    pub fn get_f64(&self) -> f64 {
        unsafe { return cten_get_f64(self.tensor); }
    }

    pub fn rdiv(&self, val: f64) -> DenseTensor<T> {
        unsafe {
            return DenseTensor { tensor: cten_rdiv(val, self.tensor), _marker: PhantomData };
        }
    }

    pub fn sqrt(&self) -> DenseTensor<T> {
        unsafe {
            return DenseTensor { tensor: cten_sqrt(self.tensor), _marker: PhantomData };
        }
    }

    pub fn addcdiv_(&mut self, val: f64, tensor1: &DenseTensor<T>, tensor2: &DenseTensor<T>) {
        unsafe {
            cten_addcdiv_(self.tensor, val, tensor1.tensor, tensor2.tensor)
        }
    }

    pub fn addcmul_(&mut self, val: f64, tensor1: &DenseTensor<T>, tensor2: &DenseTensor<T>) {
        unsafe {
            cten_addcmul_(self.tensor, val, tensor1.tensor, tensor2.tensor)
        }
    }

    pub fn to_1darray(&self) -> Vec<T::PrimitiveType> {
        // TODO: improve this function, currently very inefficiently
        unsafe {
            assert_eq!(cten_ndim(self.tensor), 1);
            let numel = cten_numel(self.tensor);

            let mut vec: Vec<T::PrimitiveType> = Vec::with_capacity(numel as usize);
            for i in 0..numel {
                vec.push(T::PrimitiveType::from_f64(cten_unsafe_select_scalar(self.tensor, i)).unwrap());
            }

            return vec;
        }
    }

    pub fn to_2darray(&self) -> Vec<Vec<T::PrimitiveType>> {
        // TODO: improve this function, currently very inefficiently
        unsafe {
            let _tensor1 = cten_contiguous(self.tensor);
            let _tensor = cten_view1(_tensor1);
            cten_drop_tensor(_tensor1);

            let shape = self.size();
            let d0 = shape[0];
            let d1 = shape[1];
            assert_eq!(shape.len(), 2);

            let mut vec: Vec<Vec<T::PrimitiveType>> = Vec::with_capacity(d0 as usize);
            for i in 0..d0 {
                let mut v = Vec::with_capacity(d1 as usize);
                for j in 0..d1 {
                    v.push(T::PrimitiveType::from_f64(cten_unsafe_select_scalar(_tensor, i * d1 + j)).unwrap());
                }
                vec.push(v);
            }

            cten_drop_tensor(_tensor);
            return vec;
        }
    }
}

impl<T: TensorType> fmt::Debug for DenseTensor<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // TODO: FIX ME!! memory leak because not cleaning C_string
        unsafe {
            let c_buf = cten_to_string(self.tensor);
            let content = CStr::from_ptr(c_buf).to_str().unwrap();
            return write!(f, "{}", content);
        }
    }
}

impl<T: TensorType> Clone for DenseTensor<T> {
    fn clone(&self) -> DenseTensor<T> {
        unsafe {
            return DenseTensor { tensor: cten_clone(self.tensor), _marker: PhantomData };
        }
    }
}

impl<T: TensorType> fmt::Display for DenseTensor<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "DenseTensor(size={:?}, type={:?})", self.size(), T::get_scalar_type())
    }
}

impl<T: TensorType> Drop for DenseTensor<T> {
    fn drop(&mut self) {
        unsafe { cten_drop_tensor(self.tensor); }
    }
}

impl<T: TensorType> Default for DenseTensor<T> {
    fn default() -> DenseTensor<T> {
        DenseTensor::zeros(&vec![0])
    }
}

impl<T: TensorType> Sum<DenseTensor<T>> for DenseTensor<T> {
    // note this function is not safe, and should be used with care, sum will
    // only produce new tensor if you are sum more than 1 tensor, otherwise
    // it return a new reference to the first one.
    fn sum<I: Iterator<Item=DenseTensor<T>>>(iter: I) -> DenseTensor<T> {
        unsafe {
            let ptr: Vec<*const ATensor> = iter.map(|t| t.tensor).collect();
            if ptr.len() == 0 {
                DenseTensor::zeros(&vec![0])
            } else {
                let tensor = cten_sum_tensors(ptr.len(), ptr.as_ptr());
                DenseTensor { tensor, _marker: PhantomData }
            }
        }
    }
}

impl<'a, T: TensorType> Sum<&'a DenseTensor<T>> for DenseTensor<T> {
    // note this function is not safe, and should be used with care, sum will
    // only produce new tensor if you are sum more than 1 tensor, otherwise
    // it return a new reference to the first one.
    fn sum<I: Iterator<Item=&'a DenseTensor<T>>>(iter: I) -> DenseTensor<T> {
        unsafe {
            let ptr: Vec<*const ATensor> = iter.map(|t| t.tensor).collect();
            let tensor = cten_sum_tensors(ptr.len(), ptr.as_ptr());
            return DenseTensor { tensor, _marker: PhantomData };
        }
    }
}

unsafe impl<T: TensorType> Sync for DenseTensor<T> {
}

unsafe impl<T: TensorType> Send for DenseTensor<T> {
}