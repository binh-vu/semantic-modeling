use libc::size_t;
use libc::c_void;
use std::os::raw::c_char;
use tensors::tensor_index::AdvancedSlice;

#[repr(C)] pub struct ATensor { _unused: [u8; 0] }
#[repr(C)] pub struct Shape {
    pub ndim: size_t,
    pub dimensions: *mut i64
}
#[repr(C)] pub struct TensorPair {
    pub first: *const ATensor,
    pub second: *const ATensor
}

#[link(name = "ctensor")]
#[allow(dead_code)]
extern {
    pub fn cten_manual_seed(seed: u64, backend: i32);
    pub fn cten_get_dtype(tensor: *const ATensor) -> i32;
    pub fn cten_create_reference(tensor: *const ATensor) -> *const ATensor;
    pub fn cten_create(ndim: size_t, dimensions: *const i64, dtype: i32, backend: i32) -> *const ATensor;
    pub fn cten_create_randn(ndim: size_t, dimensions: *const i64, dtype: i32, backend: i32) -> *const ATensor;
    pub fn cten_from_array(size: i64, array: *const c_void, dtype: i32, backend: i32) -> *const ATensor;
    pub fn cten_from_array_and_view(size: i64, ndim: size_t, dimensions: *const i64, array: *const c_void, dtype: i32, backend: i32) -> *const ATensor;
    pub fn cten_borrow_from_array(size: i64, array: *const c_void, dtype: i32, backend: i32) -> *const ATensor;
    pub fn cten_zeros(ndim: size_t, dimensions: *const i64, dtype: i32, backend: i32) -> *const ATensor;
    pub fn cten_ones(ndim: size_t, dimensions: *const i64, dtype: i32, backend: i32) -> *const ATensor;
    pub fn cten_zeros_like(tensor: *const ATensor, dtype: i32, backend: i32) -> *const ATensor;
    pub fn cten_ones_like(tensor: *const ATensor, dtype: i32, backend: i32) -> *const ATensor;
    pub fn cten_stack(ntensor: size_t, tensors: *const *const ATensor, dim: i64) -> *const ATensor;
    pub fn cten_concat(ntensor: size_t, tensors: *const *const ATensor, dim: i64) -> *const ATensor;
    pub fn cten_sum_tensors(ntensor: size_t, tensors: *const *const ATensor) -> *const ATensor;
    pub fn cten_zero_(tensor: *const ATensor);
    pub fn cten_equal(tensor: *const ATensor, that: *const ATensor) -> bool;
    pub fn cten_clone(tensor: *const ATensor) -> *const ATensor;
    pub fn cten_sum(tensor: *const ATensor) -> *const ATensor;
    pub fn cten_sum_along_dim(tensor: *const ATensor, dim: i32, keep_dim: bool) -> *const ATensor;
    pub fn cten_pow_tensor(tensor: *const ATensor, exp: i32) -> *const ATensor;
    pub fn cten_exp_tensor(tensor: *const ATensor) -> *const ATensor;
    pub fn cten_log_tensor(tensor: *const ATensor) -> *const ATensor;
    pub fn cten_max(tensor: *const ATensor) -> *const ATensor;
    pub fn cten_max_w_tensor(this: *const ATensor, other: *const ATensor) -> *const ATensor;
    pub fn cten_max_in_dim(tensor: *const ATensor, dim: i32, keep_dim: bool) -> TensorPair;
    pub fn cten_transpose(tensor: *const ATensor) -> *const ATensor;
    pub fn cten_swap_axes(tensor: *const ATensor, dim1: i64, dim2: i64) -> *const ATensor;
    pub fn cten_dot(tensor: *const ATensor, another: *const ATensor) -> *const ATensor;
    pub fn cten_outer(tensor: *const ATensor, another: *const ATensor) -> *const ATensor;
    pub fn cten_matmul(tensor: *const ATensor, another: *const ATensor) -> *const ATensor;
    pub fn cten_mm(tensor: *const ATensor, another: *const ATensor) -> *const ATensor;
    pub fn cten_mv(tensor: *const ATensor, another: *const ATensor) -> *const ATensor;
    pub fn cten_repeat(tensor: *const ATensor, ndim: size_t, dimensions: *const i64) -> *const ATensor;
    pub fn cten_expand(tensor: *const ATensor, ndim: size_t, dimensions: *const i64) -> *const ATensor;
    pub fn cten_view(tensor: *const ATensor, ndim: size_t, dimensions: *const i64) -> *const ATensor;
    pub fn cten_view1(tensor: *const ATensor) -> *const ATensor;
    pub fn cten_view2(tensor: *const ATensor, dim1: i64, dim2: i64) -> *const ATensor;
    pub fn cten_view3(tensor: *const ATensor, dim1: i64, dim2: i64, dim3: i64) -> *const ATensor;
    pub fn cten_size(tensor: *const ATensor) -> Shape;
    pub fn cten_size_in_dim(tensor: *const ATensor, dim: i64) -> i64;
    pub fn cten_ndim(tensor: *const ATensor) -> i64;
    pub fn cten_squeeze(tensor: *const ATensor) -> *const ATensor;
    pub fn cten_numel(tensor: *const ATensor) -> i64;
    pub fn cten_cuda(tensor: *const ATensor) -> *const ATensor;
    pub fn cten_cpu(tensor: *const ATensor) -> *const ATensor;
    pub fn cten_is_cuda(tensor: *const ATensor) -> bool;
    pub fn cten_is_contiguous(tensor: *const ATensor) -> bool;
    pub fn cten_contiguous(tensor: *const ATensor) -> *const ATensor;
    pub fn cten_sigmoid(tensor: *const ATensor) -> *const ATensor;
    pub fn cten_sigmoid_(tensor: *const ATensor);
    pub fn cten_log_sum_exp(tensor: *const ATensor) -> *const ATensor;
    pub fn cten_log_sum_exp_2dim(tensor: *const ATensor, dim: i32) -> *const ATensor;

    pub fn cten_get_i32(tensor: *const ATensor) -> i32;
    pub fn cten_get_i64(tensor: *const ATensor) -> i64;
    pub fn cten_get_f32(tensor: *const ATensor) -> f32;
    pub fn cten_get_f64(tensor: *const ATensor) -> f64;
    pub fn cten_unsafe_select_scalar(tensor: *const ATensor, index: i64) -> f64;
    
    pub fn cten_add_t(tensor: *const ATensor, another: *const ATensor) -> *const ATensor;
    pub fn cten_add_v(tensor: *const ATensor, val: f64) -> *const ATensor;
    pub fn cten_add_t_(tensor: *const ATensor, another: *const ATensor);
    pub fn cten_add_v_(tensor: *const ATensor, val: f64);
    pub fn cten_neg(tensor: *const ATensor) -> *const ATensor;
    pub fn cten_neg_(tensor: *const ATensor);
    pub fn cten_sub_t(tensor: *const ATensor, another: *const ATensor) -> *const ATensor;
    pub fn cten_sub_v(tensor: *const ATensor, val: f64) -> *const ATensor;
    pub fn cten_rsub_v(tensor: *const ATensor, val: f64) -> *const ATensor;
    pub fn cten_sub_t_(tensor: *const ATensor, another: *const ATensor);
    pub fn cten_sub_v_(tensor: *const ATensor, val: f64);
    pub fn cten_mul_t(tensor: *const ATensor, another: *const ATensor) -> *const ATensor;
    pub fn cten_mul_v(tensor: *const ATensor, val: f64) -> *const ATensor;
    pub fn cten_mul_t_(tensor: *const ATensor, another: *const ATensor);
    pub fn cten_mul_v_(tensor: *const ATensor, val: f64);
    pub fn cten_div_t(tensor: *const ATensor, another: *const ATensor) -> *const ATensor;
    pub fn cten_div_v(tensor: *const ATensor, val: f64) -> *const ATensor;
    pub fn cten_div_t_(tensor: *const ATensor, another: *const ATensor);
    pub fn cten_div_v_(tensor: *const ATensor, val: f64);
    pub fn cten_rdiv(val: f64, tensor: *const ATensor) -> *const ATensor;
    pub fn cten_sqrt(tensor: *const ATensor) -> *const ATensor;
    pub fn cten_addcdiv_(this: *const ATensor, value: f64, tensor1: *const ATensor, tensor2: *const ATensor);
    pub fn cten_addcmul_(this: *const ATensor, value: f64, tensor1: *const ATensor, tensor2: *const ATensor);
    pub fn cten_select(tensor: *const ATensor, dim: i64, index: i64) -> *const ATensor;
    pub fn cten_slice(tensor: *const ATensor, dim: i64, start: i64, end: i64, step: i64) -> *const ATensor;
    pub fn cten_index_select(tensor: *const ATensor, dim: i64, index: *const ATensor) -> *const ATensor;
    pub fn cten_index_select_v(tensor: *const ATensor, dim: i64, size: i64, array: *const i64) -> *const ATensor;
    pub fn cten_select_along_dims(tensor: *const ATensor, n_indices: i32, indices: *const i64) -> *const ATensor;
    pub fn cten_advance_access_index(tensor: *const ATensor, n_indice: i32, indices: *const AdvancedSlice) -> *const ATensor;

    pub fn cten_fill_(tensor: *const ATensor, val: f64);
    pub fn cten_copy_(tensor: *const ATensor, another: *const ATensor, async: bool);

    pub fn cten_select_fill_(tensor: *const ATensor, dim: i64, index: i64, val: f64);
    pub fn cten_select_copy_(tensor: *const ATensor, dim: i64, index: i64, val: *const ATensor);
    pub fn cten_index_fill_(tensor: *const ATensor, dim: i64, index: *const ATensor, val: f64);
    pub fn cten_index_copy_(tensor: *const ATensor, dim: i64, index: *const ATensor, val: *const ATensor);
    pub fn cten_slice_fill_(tensor: *const ATensor, dim: i64, start: i64, end: i64, step: i64, val: f64);
    pub fn cten_slice_copy_(tensor: *const ATensor, dim: i64, start: i64, end: i64, step: i64, val: *const ATensor);
    pub fn cten_select_along_dims_fill_(tensor: *const ATensor, n_indices: i32, indices: *const i64, val: f64);
    pub fn cten_select_along_dims_copy_(tensor: *const ATensor, n_indices: i32, indices: *const i64, val: *const ATensor);

    pub fn cten_get_data_ptr(tensor: *const ATensor) -> *const c_void;
    pub fn cten_to_string(tensor: *const ATensor) -> *const c_char;
    pub fn cten_drop_tensor(tensor: *const ATensor);
}