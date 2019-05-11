//
//

#include <ATen/ATen.h>
#include "dense_tensor.h"
#include <string.h>
#include <unordered_map>

#define ENABLE_CATCHING_EXCEPTION

using namespace at;

extern "C" {
    Backend _CTEN_BACKEND[] = {Backend::CPU, Backend::CUDA};
    ScalarType _CTEN_SCALAR_TYPES[] = {
            ScalarType::Char, ScalarType::Short, ScalarType::Int,
            ScalarType::Long, ScalarType::Half, ScalarType::Float, ScalarType::Double
    };
    std::unordered_map<int, int> _CTEN_INT_to_SCALAR_TYPES = {
            {int(ScalarType::Char), 0}, {int(ScalarType::Short), 1},
            {int(ScalarType::Int), 2}, {int(ScalarType::Long), 3}, {int(ScalarType::Half), 4},
            {int(ScalarType::Float), 5}, {int(ScalarType::Double), 6}
    };

    void cten_manual_seed(uint64_t seed, int backend) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        at::Generator &gen = at::globalContext().defaultGenerator(_CTEN_BACKEND[backend]);
        gen.manualSeed(seed);
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_manual_seed`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }

    int cten_get_dtype(Tensor* tensor) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        return _CTEN_INT_to_SCALAR_TYPES[int(tensor->type().scalarType())];
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_get_dtype`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }

    Tensor *cten_create_reference(Tensor *tensor) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        return new Tensor(*tensor);
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_create`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }

    Tensor *cten_create(size_t ndim, int64_t *dimensions, int dtype, int backend) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        at::IntList shape(dimensions, ndim);
        return new Tensor(at::getType(_CTEN_BACKEND[backend], _CTEN_SCALAR_TYPES[dtype]).tensor(shape));
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_create`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }

    Tensor *cten_create_randn(size_t ndim, int64_t *dimensions, int dtype, int backend) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        at::IntList shape(dimensions, ndim);
        return new Tensor(at::getType(_CTEN_BACKEND[backend], _CTEN_SCALAR_TYPES[dtype]).randn(shape));
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_create_randn`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }

    Tensor *cten_from_array(int64_t size, void *array, int dtype, int backend) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        auto tensor = at::getType(at::kCPU, _CTEN_SCALAR_TYPES[dtype]).tensorFromBlob(array, {size});
        return new Tensor(at::getType(_CTEN_BACKEND[backend], _CTEN_SCALAR_TYPES[dtype]).copy(tensor));
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_from_array`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }

    Tensor *cten_from_array_and_view(int64_t size, size_t ndim, int64_t *dimensions, void *array, int dtype, int backend) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        auto tensor = at::getType(at::kCPU, _CTEN_SCALAR_TYPES[dtype])
                .tensorFromBlob(array, {size})
                .view(at::IntList(dimensions, ndim));
        return new Tensor(at::getType(_CTEN_BACKEND[backend], _CTEN_SCALAR_TYPES[dtype]).copy(tensor));
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_from_array_and_view`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }

    Tensor *cten_borrow_from_array(int64_t size, void *array, int dtype, int backend) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        return new Tensor(at::getType(_CTEN_BACKEND[backend], _CTEN_SCALAR_TYPES[dtype]).tensorFromBlob(array, {size}));
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_borrow_from_array`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }

    Tensor *cten_zeros(size_t ndim, int64_t *dimensions, int dtype, int backend) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        at::IntList shape(dimensions, ndim);
        return new Tensor(at::getType(_CTEN_BACKEND[backend], _CTEN_SCALAR_TYPES[dtype]).zeros(shape));
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_zeros`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }

    Tensor *cten_ones(size_t ndim, int64_t *dimensions, int dtype, int backend) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        at::IntList shape(dimensions, ndim);
        return new Tensor(at::getType(_CTEN_BACKEND[backend], _CTEN_SCALAR_TYPES[dtype]).ones(shape));
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_ones`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }

    Tensor *cten_zeros_like(Tensor *tensor, int dtype, int backend) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        return new Tensor(at::getType(_CTEN_BACKEND[backend], _CTEN_SCALAR_TYPES[dtype]).zeros_like(*tensor));
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_zeros_like`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }

    Tensor *cten_ones_like(Tensor *tensor, int dtype, int backend) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        return new Tensor(at::getType(_CTEN_BACKEND[backend], _CTEN_SCALAR_TYPES[dtype]).ones_like(*tensor));
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_ones_like`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }

    Tensor *cten_stack(size_t ntensor, Tensor **tensors, int64_t dim) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        Tensor *x = new Tensor[ntensor];
        for (int i = 0; i < ntensor; i++) {
            x[i] = *tensors[i];
        }

        at::TensorList aten_list(x, ntensor);
        return new at::Tensor(at::stack(aten_list, dim));
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_stack`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }

    Tensor *cten_concat(size_t ntensor, Tensor **tensors, int64_t dim) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        Tensor *x = new Tensor[ntensor];
        for (int i = 0; i < ntensor; i++) {
            x[i] = *tensors[i];
        }

        at::TensorList aten_list(x, ntensor);
        return new at::Tensor(at::cat(aten_list, dim));
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_concat`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }

    Tensor *cten_sum_tensors(size_t ntensor, Tensor **tensors) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        if (ntensor == 1) {
            return new Tensor(*tensors[0]);
        }

        at::Tensor aten = *tensors[0] + *tensors[1];
        for (int i = 2; i < ntensor; i++) {
            aten += *tensors[i];
        }

        return new Tensor(aten);
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_sum_tensors`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }

    void cten_zero_(Tensor *tensor) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        tensor->zero_();
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_zero_`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }

    bool cten_equal(Tensor *self, Tensor *that) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        return self->equal(*that);
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_equal`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }

    Tensor *cten_clone(Tensor *self) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        return new Tensor(self->clone());
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_clone`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }

    Tensor *cten_sum(Tensor *self) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        return new Tensor(self->sum());
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_sum`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }

    Tensor *cten_sum_along_dim(Tensor *self, int dim, bool keep_dim) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        return new Tensor(self->sum(dim, keep_dim));
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_sum_along_dim`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }

    Tensor *cten_pow_tensor(Tensor *self, int exp) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        return new Tensor(self->pow(exp));
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_pow_tensor`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }

    Tensor *cten_exp_tensor(Tensor *self) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        return new Tensor(self->exp());
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_exp_tensor`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }

    Tensor *cten_log_tensor(Tensor *self) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        return new Tensor(self->log());
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_log_tensor`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }

    Tensor *cten_max(Tensor *self) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        return new Tensor(self->max());
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_max`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }

    Tensor *cten_max_w_tensor(Tensor *self, Tensor *other) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
            return new Tensor(self->max(*other));
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_max`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }

    TensorPair cten_max_in_dim(Tensor *self, int dim, bool keep_dim) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        auto result = self->max(dim, keep_dim);
        return TensorPair{ new Tensor(std::get<0>(result)), new Tensor(std::get<1>(result)) };
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_max_in_dim`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }

    Tensor *cten_transpose(Tensor *self) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        return new Tensor(self->t());
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_transpose`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }

    Tensor *cten_swap_axes(Tensor *self, int64_t dim1, int64_t dim2) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        return new Tensor(self->transpose(dim1, dim2));
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_swap_axes`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }

    Tensor *cten_dot(Tensor *self, Tensor *another) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        return new Tensor(self->dot(*another));
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_dot`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }

    Tensor *cten_outer(Tensor *self, Tensor *another) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        return new Tensor(self->ger(*another));
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_outer`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }

    Tensor *cten_matmul(Tensor *self, Tensor *another) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        return new Tensor(self->matmul(*another));
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_matmul`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }

    Tensor *cten_mm(Tensor *self, Tensor *another) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        return new Tensor(self->mm(*another));
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_mm`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }

    Tensor *cten_mv(Tensor *self, Tensor *another) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        return new Tensor(self->mv(*another));
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_mv`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }

    Tensor *cten_repeat(Tensor *self, size_t ndim, int64_t *dimensions) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        at::IntList shape(dimensions, ndim);
        return new Tensor(self->repeat(shape));
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_repeat`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }

    Tensor *cten_expand(Tensor *self, size_t ndim, int64_t *dimensions) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        at::IntList shape(dimensions, ndim);
        return new Tensor(self->expand(shape));
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_expand`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }

    Tensor *cten_view(Tensor *tensor, size_t ndim, int64_t *dimensions) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
            return new Tensor(tensor->view(at::IntList(dimensions, ndim)));
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_view`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }

    Tensor *cten_view1(Tensor *tensor) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
            return new Tensor(tensor->view({-1}));
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_view1`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }

    Tensor *cten_view2(Tensor *tensor, int64_t dim1, int64_t dim2) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
            return new Tensor(tensor->view({dim1, dim2}));
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_view2`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }

    Tensor *cten_view3(Tensor *tensor, int64_t dim1, int64_t dim2, int64_t dim3) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
            return new Tensor(tensor->view({dim1, dim2, dim3}));
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_view3`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }

    Shape cten_size(Tensor *tensor) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif    
        auto shape = tensor->sizes();
        return Shape{shape.size(), shape.data()};
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_size`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }

    int64_t cten_size_in_dim(Tensor *tensor, int64_t dim) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        return tensor->size(dim);
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_size_in_dim`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }

    int64_t cten_ndim(Tensor *tensor) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        return tensor->dim();
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_ndim`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }

    Tensor *cten_squeeze(Tensor *tensor) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        return new Tensor(tensor->squeeze());
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_squeeze`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }

    int64_t cten_numel(Tensor *tensor) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        return tensor->numel();
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_numel`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }

    Tensor *cten_cuda_(Tensor *tensor) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        return new Tensor(tensor->toBackend(Backend::CUDA));
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_cuda_`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }

    Tensor *cten_cpu(Tensor *tensor) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        return new Tensor(tensor->toBackend(Backend::CPU));
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_cpu`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }

    bool cten_is_cuda(Tensor *tensor) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        return tensor->is_cuda();
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_is_cuda`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }

    bool cten_is_contiguous(Tensor *tensor) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        return tensor->is_contiguous();
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_is_contiguous`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }

    Tensor *cten_contiguous(Tensor *tensor) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        return new Tensor(tensor->contiguous());
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_contiguous`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }

    Tensor *cten_sigmoid(Tensor *tensor) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        return new Tensor(tensor->sigmoid());
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_sigmoid`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }

    void cten_sigmoid_(Tensor *tensor) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        tensor->sigmoid_();
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_sigmoid_`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }

    Tensor *cten_log_sum_exp(Tensor *tensor) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        at::Tensor x_prime = tensor->max();
        return new Tensor((*tensor - x_prime).exp().sum().log() + x_prime);
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_log_sum_exp`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }

    Tensor *cten_log_sum_exp_2dim(Tensor *tensor, int dim) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        auto x_prime = std::get<0>(tensor->max(dim, true));
        return new Tensor((*tensor - x_prime).exp().sum(dim).log() + x_prime.squeeze());
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_log_sum_exp_2dim`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }

    int32_t cten_get_i32(Tensor *tensor) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        return tensor->toCInt();
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_get_i32`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }

    int64_t cten_get_i64(Tensor *tensor) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
            return tensor->toCLong();
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_get_i64`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }

    float cten_get_f32(Tensor *tensor) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
            return tensor->toCFloat();
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_get_f32`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }

    double cten_get_f64(Tensor *tensor) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
            return tensor->toCDouble();
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_get_f64`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }

    double cten_unsafe_select_scalar(Tensor* tensor, int64_t index) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        return tensor->select(0, index).toCDouble();
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_unsafe_select_scalar`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }

    Tensor *cten_add_t(Tensor *self, Tensor *another) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        return new Tensor(self->add(*another));
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_add_t`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }
    Tensor *cten_add_v(Tensor *self, double val) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        return new Tensor(self->add(val));
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_add_v`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }
    void cten_add_t_(Tensor *self, Tensor *another) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        self->add_(*another);
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_add_t_`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }
    void cten_add_v_(Tensor *self, double val) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        self->add_(val);
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_add_v_`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }
    Tensor *cten_neg(Tensor *self) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        return new Tensor(self->neg());
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_neg`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }
    void cten_neg_(Tensor *self) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        self->neg_();
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_neg_`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }
    Tensor *cten_sub_t(Tensor *self, Tensor *another) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        return new Tensor(self->sub(*another));
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_sub_t`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }
    Tensor *cten_sub_v(Tensor *self, double val) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        return new Tensor(self->sub(val));
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_sub_v`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }
    Tensor *cten_rsub_v(Tensor *self, double val) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        return new Tensor(self->neg().add(val));
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_rsub_v`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }
    void cten_sub_t_(Tensor *self, Tensor *another) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        self->sub_(*another);
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_sub_t_`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }
    void cten_sub_v_(Tensor *self, double val) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        self->sub_(val);
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_sub_v_`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }
    Tensor *cten_mul_t(Tensor *self, Tensor *another) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        return new Tensor(self->mul(*another));
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_mul_t`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }
    Tensor *cten_mul_v(Tensor *self, double val) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        return new Tensor(self->mul(val));
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_mul_v`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }
    void cten_mul_t_(Tensor *self, Tensor *another) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        self->mul_(*another);
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_mul_t_`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }
    void cten_mul_v_(Tensor *self, double val) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        self->mul_(val);
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_mul_v_`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }
    Tensor *cten_div_t(Tensor *self, Tensor *another) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        return new Tensor(self->div(*another));
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_div_t`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }
    Tensor *cten_div_v(Tensor *self, double val) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        return new Tensor(self->div(val));
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_div_v`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }
    void cten_div_t_(Tensor *self, Tensor *another) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        self->div_(*another);
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_div_t_`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }
    void cten_div_v_(Tensor *self, double val) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        self->div_(val);
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_div_v_`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }
    Tensor *cten_rdiv(double val, Tensor *self) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        return new Tensor(at::Scalar(val) / *self);
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_rdiv`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }

    Tensor *cten_sqrt(Tensor* self) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
            return new Tensor(self->sqrt());
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_rdiv`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }

    void cten_addcdiv_(Tensor* self, double value, Tensor* tensor1, Tensor* tensor2) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
            self->addcdiv_(*tensor1, *tensor2, value);
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_rdiv`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }

    void cten_addcmul_(Tensor* self, double value, Tensor* tensor1, Tensor* tensor2) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
            self->addcmul_(*tensor1, *tensor2, value);
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_rdiv`: " << error.what() << std::endl;
            exit(1);
        }
#endif      
    }

    Tensor *cten_select(Tensor *tensor, int64_t dim, int64_t index) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        return new Tensor(tensor->select(dim, index));
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_select`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }
    Tensor *cten_slice(Tensor *tensor, int64_t dim, int64_t start, int64_t end, int64_t step) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        return new Tensor(tensor->slice(dim, start, end, step));
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_slice`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }
    Tensor *cten_index_select(Tensor *tensor, int64_t dim, Tensor *index) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        return new Tensor(tensor->index_select(dim, *index));
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_index_select`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }
    Tensor *cten_index_select_v(Tensor* tensor, int64_t dim, int64_t size, int64_t *array) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        Tensor index = at::CPU(at::kLong).tensorFromBlob(array, {size});
        return new Tensor(tensor->index_select(dim, index));
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_index_select_v`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }
    Tensor* cten_select_along_dims(Tensor* tensor, int n_indices, int64_t *indices) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        // support select like python: x[1, 1, 3] (different with index_select, which is: x[[1, 1, 3]])
        Tensor curr_tensor = *tensor;
        for (long dim = 0; dim < n_indices; dim++) {
            curr_tensor = curr_tensor.select(0, indices[dim]);
        }
        return new Tensor(curr_tensor);
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_select_along_dims`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }
    Tensor *cten_advance_access_index(Tensor *tensor, int n_indice, AdvanceIndexing *indices) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        Tensor curr_tensor = *tensor;
        long dim_offset = 0;

        for (long dim = 0; dim < n_indice; dim++) {
            if (indices[dim].is_select_all) {
                continue;
            } else if (indices[dim].is_slice) {
                curr_tensor = curr_tensor.slice(dim - dim_offset, indices[dim].start, indices[dim].end,
                                                indices[dim].step);
            } else {
                curr_tensor = curr_tensor.select(dim - dim_offset++, indices[dim].idx);
            }
        }

        return new Tensor(curr_tensor);
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_advance_access_index`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }

    void cten_fill_(Tensor *self, double val) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        self->fill_(val);
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_fill_`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }
    void cten_copy_(Tensor *self, Tensor *another, bool async) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        self->copy_(*another, async);
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_copy_`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }
    void cten_index_copy_(Tensor *tensor, int64_t dim, Tensor *index, Tensor *source) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        tensor->index_copy_(dim, *index, *source);
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_index_copy_`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }

    void cten_select_fill_(Tensor *tensor, int64_t dim, int64_t index, double val) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        tensor->select(dim, index).fill_(val);
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_select_fill_`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }
    void cten_select_copy_(Tensor *tensor, int64_t dim, int64_t index, Tensor *val) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        tensor->select(dim, index).copy_(*val);
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_select_copy_`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }
    void cten_index_fill_val_(Tensor *tensor, int64_t dim, Tensor *index, double val) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        tensor->index_fill_(dim, *index, val);
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_index_fill_val_`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }
    void cten_slice_fill_(Tensor *tensor, int64_t dim, int64_t start, int64_t end, int64_t step, double val) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        tensor->slice(dim, start, end, step).fill_(val);
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_slice_fill_`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }
    void cten_slice_copy_(Tensor *tensor, int64_t dim, int64_t start, int64_t end, int64_t step, Tensor *val) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        tensor->slice(dim, start, end, step).copy_(*val);
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_slice_copy_`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }
    void cten_select_along_dims_fill_(Tensor* tensor, int n_indices, int64_t *indices, double val) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        // support select like python: x[1, 1, 3] (different with index_select, which is: x[[1, 1, 3]])
        Tensor curr_tensor = *tensor;
        for (long dim = 0; dim < n_indices; dim++) {
            curr_tensor = curr_tensor.select(0, indices[dim]);
        }
        curr_tensor.fill_(val);
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_select_along_dims_fill_`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }
    void cten_select_along_dims_copy_(Tensor* tensor, int n_indices, int64_t *indices, Tensor *val) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        // support select like python: x[1, 1, 3] (different with index_select, which is: x[[1, 1, 3]])
        Tensor curr_tensor = *tensor;
        for (long dim = 0; dim < n_indices; dim++) {
            curr_tensor = curr_tensor.select(0, indices[dim]);
        }
        curr_tensor.copy_(*val);
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_select_along_dims_copy_`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }

    void *cten_get_data_ptr(Tensor *tensor) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        return tensor->data_ptr();
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_get_data_ptr`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }
    const char *cten_to_string(Tensor *tensor) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        std::stringstream buffer;
        buffer << *tensor << std::endl;

        const std::string &tmp_str = buffer.str();
        char *cstr = new char[tmp_str.length() + 1];
        strcpy(cstr, tmp_str.c_str());

        return cstr;
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_to_string`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }
    void cten_drop_tensor(Tensor *tensor) {
#if defined ENABLE_CATCHING_EXCEPTION
        try {
#endif
        delete tensor;
#if defined ENABLE_CATCHING_EXCEPTION
        } catch (const std::runtime_error &error) {
            std::cerr << "Error while calling `cten_drop_tensor`: " << error.what() << std::endl;
            exit(1);
        }
#endif
    }
}