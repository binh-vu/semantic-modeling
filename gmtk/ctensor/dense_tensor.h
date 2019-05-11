//
//

#ifndef CTENSOR_DENSE_TENSOR_H
#define CTENSOR_DENSE_TENSOR_H

#include <ATen/Tensor.h>
#include <ATen/ATen.h>

using namespace at;

extern "C" {
    /// A C wrapper for C++ DenseTensor library

    struct Shape {
        size_t ndim;
        const int64_t *dimensions;
    };

    struct TensorPair {
        Tensor *first;
        Tensor *second;
    };

    struct AdvanceIndexing {
        int64_t idx;
        int64_t start;
        int64_t end;
        int64_t step;
        bool is_slice;
        bool is_select_all;
    };

    AdvanceIndexing AdvanceIndexing_set_idx(int64_t idx);
    AdvanceIndexing AdvanceIndexing_set_slice(int64_t start, int64_t end, int64_t step);
    void cten_manual_seed(uint64_t seed, int backend);
    int cten_get_dtype(Tensor* tensor);
    Tensor *cten_create_reference(Tensor *tensor);
    Tensor *cten_create(size_t ndim, int64_t *dimensions, int dtype, int backend);
    Tensor *cten_create_randn(size_t ndim, int64_t *dimensions, int dtype, int backend);
    Tensor *cten_from_array(int64_t size, void *array, int dtype, int backend);
    Tensor *cten_from_array_and_view(int64_t size, size_t ndim, int64_t *dimensions, void *array, int dtype, int backend);
    Tensor *cten_borrow_from_array(int64_t size, void *array, int dtype, int backend);
    Tensor *cten_zeros(size_t ndim, int64_t *dimensions, int dtype, int backend);
    Tensor *cten_ones(size_t ndim, int64_t *dimensions, int dtype, int backend);
    Tensor *cten_zeros_like(Tensor *tensor, int dtype, int backend);
    Tensor *cten_ones_like(Tensor *tensor, int dtype, int backend);
    Tensor *cten_stack(size_t ntensor, Tensor **tensors, int64_t dim);
    Tensor *cten_concat(size_t ntensor, Tensor **tensors, int64_t dim);
    Tensor *cten_sum_tensors(size_t ntensor, Tensor **tensors);
    void cten_zero_(Tensor *tensor);
    bool cten_equal(Tensor *self, Tensor *that);
    Tensor *cten_clone(Tensor *self);
    Tensor *cten_sum(Tensor *self);
    Tensor *cten_sum_along_dim(Tensor *self, int dim, bool keep_dim);
    Tensor *cten_pow_tensor(Tensor *self, int exp);
    Tensor *cten_exp_tensor(Tensor *self);
    Tensor *cten_log_tensor(Tensor *self);
    Tensor *cten_max(Tensor *self);
    Tensor *cten_max_w_tensor(Tensor *self, Tensor *other);
    TensorPair cten_max_in_dim(Tensor *self, int dim, bool keep_dim);
    Tensor *cten_transpose(Tensor *self);
    Tensor *cten_swap_axes(Tensor *self, int64_t dim1, int64_t dim2);
    Tensor *cten_dot(Tensor *self, Tensor *another);
    Tensor *cten_outer(Tensor *self, Tensor *another);
    Tensor *cten_matmul(Tensor *self, Tensor *another);
    Tensor *cten_mm(Tensor *self, Tensor *another);
    Tensor *cten_mv(Tensor *self, Tensor *another);
    Tensor *cten_repeat(Tensor *self, size_t ndim, int64_t *dimensions);
    Tensor *cten_expand(Tensor *self, size_t ndim, int64_t *dimensions);
    Tensor *cten_view(Tensor *tensor, size_t ndim, int64_t *dimensions);
    Tensor *cten_view1(Tensor *tensor);
    Tensor *cten_view2(Tensor *tensor, int64_t dim1, int64_t dim2);
    Tensor *cten_view3(Tensor *tensor, int64_t dim1, int64_t dim2, int64_t dim3);
    Shape cten_size(Tensor *tensor);
    int64_t cten_size_in_dim(Tensor *tensor, int64_t dim);
    int64_t cten_ndim(Tensor *tensor);
    Tensor *cten_squeeze(Tensor *tensor);
    int64_t cten_numel(Tensor *tensor);
    Tensor *cten_cuda(Tensor *tensor);
    Tensor *cten_cpu(Tensor *tensor);
    bool cten_is_cuda(Tensor *tensor);
    bool cten_is_contiguous(Tensor *tensor);
    Tensor *cten_contiguous(Tensor *tensor);
    Tensor *cten_sigmoid(Tensor *tensor);
    void cten_sigmoid_(Tensor *tensor);
    Tensor *cten_log_sum_exp(Tensor *tensor);
    Tensor *cten_log_sum_exp_2dim(Tensor *tensor, int dim);

    int32_t cten_get_i32(Tensor *tensor);
    int64_t cten_get_i64(Tensor *tensor);
    float cten_get_f32(Tensor *tensor);
    double cten_get_f64(Tensor *tensor);
    double cten_unsafe_select_scalar(Tensor* tensor, int64_t index);

    Tensor *cten_add_t(Tensor *self, Tensor *another);
    Tensor *cten_add_v(Tensor *self, double val);
    void cten_add_t_(Tensor *self, Tensor *another);
    void cten_add_v_(Tensor *self, double val);
    Tensor *cten_neg(Tensor *self);
    void cten_neg_(Tensor *self);
    Tensor *cten_sub_t(Tensor *self, Tensor *another);
    Tensor *cten_sub_v(Tensor *self, double val);
    Tensor *cten_rsub_v(Tensor *self, double val);
    void cten_sub_t_(Tensor *self, Tensor *another);
    void cten_sub_v_(Tensor *self, double val);
    Tensor *cten_mul_t(Tensor *self, Tensor *another);
    Tensor *cten_mul_v(Tensor *self, double val);
    void cten_mul_t_(Tensor *self, Tensor *another);
    void cten_mul_v_(Tensor *self, double val);
    Tensor *cten_div_t(Tensor *self, Tensor *another);
    Tensor *cten_div_v(Tensor *self, double val);
    void cten_div_t_(Tensor *self, Tensor *another);
    void cten_div_v_(Tensor *self, double val);
    Tensor *cten_rdiv(double val, Tensor *self);
    Tensor *cten_sqrt(Tensor* self);
    void cten_addcdiv_(Tensor* self, double value, Tensor* tensor1, Tensor* tensor2);
    void cten_addcmul_(Tensor* self, double value, Tensor* tensor1, Tensor* tensor2);

    Tensor *cten_select(Tensor *tensor, int64_t dim, int64_t index);
    Tensor *cten_slice(Tensor *tensor, int64_t dim, int64_t start, int64_t end, int64_t step);
    Tensor *cten_index_select(Tensor *tensor, int64_t dim, Tensor *index);
    Tensor *cten_index_select_v(Tensor *tensor, int64_t dim, int64_t size, int64_t *array);
    Tensor *cten_select_along_dims(Tensor* tensor, int n_indices, int64_t *indices);
    Tensor *cten_advance_access_index(Tensor *tensor, int n_indice, AdvanceIndexing *indices);

    void cten_fill_(Tensor *self, double val);
    void cten_copy_(Tensor *self, Tensor *another, bool async);

    void cten_select_fill_(Tensor *tensor, int64_t dim, int64_t index, double val);
    void cten_select_copy_(Tensor *tensor, int64_t dim, int64_t index, Tensor *value);
    void cten_index_fill_(Tensor *tensor, int64_t dim, Tensor *index, double val);
    void cten_index_copy_(Tensor *tensor, int64_t dim, Tensor *index, Tensor *source);
    void cten_slice_fill_(Tensor *tensor, int64_t dim, int64_t start, int64_t end, int64_t step, double val);
    void cten_slice_copy_(Tensor *tensor, int64_t dim, int64_t start, int64_t end, int64_t step, Tensor* value);
    void cten_select_along_dims_fill_(Tensor *tensor, int n_indices, int64_t *indices, double val);
    void cten_select_along_dims_copy_(Tensor *tensor, int n_indices, int64_t *indices, Tensor *val);

    void *cten_get_data_ptr(Tensor *tensor);
    const char *cten_to_string(Tensor *tensor);
    void cten_drop_tensor(Tensor *tensor);
}

#endif //CTENSOR_DENSE_TENSOR_H
