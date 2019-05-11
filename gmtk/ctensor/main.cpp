//
//

#include <iostream>
#include "dense_tensor.h"
#include <ATen/ATen.h>

using namespace std;

ScalarType _cten_scalar_types2[] = {ScalarType::Char, ScalarType::Short, ScalarType::Int, ScalarType::Long};


void print_enum() {
    cout << at::ScalarType::Short << ":" << int(at::ScalarType::Short) << " ";
    cout << at::ScalarType::Int << ":" << int(at::ScalarType::Int) << " ";
    cout << at::ScalarType::Long << ":" << int(at::ScalarType::Long) << " ";
    cout << at::ScalarType::Half << ":" << int(at::ScalarType::Half) << " ";
    cout << at::ScalarType::Float << ":" << int(at::ScalarType::Float) << " ";
    cout << at::ScalarType::Double << ":" << int(at::ScalarType::Double) << " ";
    cout << endl;
    cout << at::Backend::CPU << ":" << int(at::Backend::CPU) << " ";
    cout << at::Backend::CUDA << ":" << int(at::Backend::CUDA) << " ";
    cout << endl;
}

void print_converted() {
    cout << _cten_scalar_types2[0] << endl;
}


int main() {
    float data[] = {1,2,3,4,5,6};
    long data2[] = {0, 1};
    at::Tensor *idx = cten_from_array(2, data2, 3, 0);
    cout << *idx << endl;

    at::Tensor *t = cten_from_array(6, data, 5, 0);
    cout << *t << endl;
    at::Tensor *t2 = cten_create_reference(t);
    cout << t << " " << t2 << endl;
    cout << *t2 << endl;
    cout << *t << endl;
    delete t;
    cout << *t2 << endl;
//    (*t)[0] = 10;
//    cout << t->index_select(0, *idx) << endl;
//    t->index_select(0, *idx)[1] = 100;
//    cout << t->index_select(0, *idx) << endl;
//
//    at::Tensor *empty = cten_from_array(0, data, 5, 0);
//    cten_add_t_(empty, t);

//    auto v = t->view({3, 2});
//    at::Tensor *t2 = transpose(t);
//        int64_t x[]
//    cout << v << endl;
//    cout << index_select_v(&v, 0, 2, )
//    print_enum();
//    print_converted();
//    cout << cten_get_dtype(t) << int(t->type().scalarType()) << endl;
    cout << "FINISH" << endl;
    return 0;
}