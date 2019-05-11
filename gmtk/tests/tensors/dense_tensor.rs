use gmtk::tensors::*;
use bincode;

/// Premises:
///     + from_array & from_ndarray return correct tensors
///     + to_array & to_2d_array: return correct value content of a tensor (value content need to have correct type!!)
///     + size: return correct dimensions of a tensor
#[test]
fn premise() {
    let tensor = DenseTensor::<TDefault>::from_array(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(tensor.size(), vec![6].as_slice());
    assert_eq!(tensor.to_1darray(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let tensor = DenseTensor::<TDefault>::from_ndarray(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    assert_eq!(tensor.size(), vec![2, 3].as_slice());
    assert_eq!(tensor.to_2darray(), vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
}

#[test]
fn test_create() {
    let tensor = DenseTensor::<TDefault>::create(&vec![5, 30, 200]);
    assert_eq!(tensor.size(), vec![5, 30, 200].as_slice());
}

#[test]
fn test_create_randn() {
    DenseTensor::<TDefault>::manual_seed(120);

    let tensor = DenseTensor::<TDefault>::create_randn(&[2, 3]);
    assert_eq!(tensor.size(), vec![2, 3].as_slice());
    assert_eq!(tensor.to_2darray(), vec![
        vec![-1.1915583610534668, 0.3627992570400238, -0.37679386138916016],
        vec![0.5234138369560242, -0.7340877652168274, -0.6237930059432983]
    ]);
}

#[test]
fn test_zeros() {
    let tensor = DenseTensor::<TDefault>::zeros(&vec![2, 3]);
    assert_eq!(tensor.size(), vec![2, 3].as_slice());
    assert_eq!(tensor.to_2darray(), vec![vec![0.0, 0.0, 0.0], vec![0.0, 0.0, 0.0]]);
}

#[test]
fn test_ones() {
    let tensor = DenseTensor::<TDefault>::ones(&vec![2, 3]);
    assert_eq!(tensor.size(), vec![2, 3].as_slice());
    assert_eq!(tensor.to_2darray(), vec![vec![1.0, 1.0, 1.0], vec![1.0, 1.0, 1.0]]);
}

#[test]
fn test_zeros_like() {
    let aten = DenseTensor::<TDefault>::from_ndarray(&[2.0, 3.0, 4.0, 5.0, 6.0, 7.0], &[3, 2]);
    let tensor = DenseTensor::<TDefault>::zeros_like(&aten);
    assert_eq!(tensor.to_2darray(), vec![vec![0.0, 0.0], vec![0.0, 0.0], vec![0.0, 0.0]]);
}

#[test]
fn test_ones_like() {
    let aten = DenseTensor::<TDefault>::from_ndarray(&[2.0, 3.0, 4.0, 5.0, 6.0, 7.0], &[3, 2]);
    let tensor = DenseTensor::<TDefault>::ones_like(&aten);
    assert_eq!(tensor.to_2darray(), vec![vec![1.0, 1.0], vec![1.0, 1.0], vec![1.0, 1.0]]);
}

#[test]
fn test_stack() {
    let tensors = vec![
        DenseTensor::<TDefault>::from_array(&[2.0, 3.0, 4.0]),
        DenseTensor::<TDefault>::from_array(&[5.0, 7.0, 9.0])];

    let tensor = DenseTensor::<TDefault>::stack(&tensors, 0);
    assert_eq!(tensor.size(), vec![2, 3].as_slice());
    assert_eq!(tensor.to_2darray(), vec![
        vec![2.0, 3.0, 4.0],
        vec![5.0, 7.0, 9.0]
    ]);

    let tensor = DenseTensor::<TDefault>::stack(&tensors, 1);
    assert_eq!(tensor.size(), vec![3, 2].as_slice());
    assert_eq!(tensor.to_2darray(), vec![
        vec![2.0, 5.0], vec![3.0, 7.0], vec![4.0, 9.0]
    ]);
}

#[test]
fn test_concat() {
    let vals = vec![DenseTensor::<TDefault>::from_array(&[2.0, 3.0, 4.0]), DenseTensor::<TDefault>::from_array(&[5.0, 7.0, 9.0])];
    assert_eq!(DenseTensor::<TDefault>::concat(&vals, 0).to_1darray(), vec![2.0, 3.0, 4.0, 5.0, 7.0, 9.0]);
}

#[test]
fn test_zero_() {
    let mut tensor = DenseTensor::<TDefault>::from_ndarray(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    assert_eq!(tensor.size(), vec![3, 2].as_slice());
    assert_eq!(tensor.to_2darray(), vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]]);
    tensor.zero_();
    assert_eq!(tensor.to_2darray(), vec![vec![0.0, 0.0], vec![0.0, 0.0], vec![0.0, 0.0]]);
}

#[test]
fn test_equal() {
    let v = DenseTensor::<TDefault>::from_ndarray(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    let u = DenseTensor::<TDefault>::from_ndarray(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.1], &[3, 2]);
    assert_ne!(v, u);
    let u2 = DenseTensor::<TDefault>::from_ndarray(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    assert_eq!(v, u2);
}

#[test]
fn test_clone() {
    let mut v = DenseTensor::<TDefault>::from_ndarray(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    let u = v.clone();
    assert_ne!(&v as *const _, &u as *const _);
    assert_eq!(v, u);
    v.zero_();
    assert_eq!(u.to_2darray(), vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]]);
}

#[test]
fn test_sum() {
    let v = DenseTensor::<TDefault>::from_ndarray(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    assert_eq!(v.sum().get_f64(), 21.0);
}

#[test]
fn test_sum_along_dim() {
    let v = DenseTensor::<TDefault>::from_ndarray(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    assert_eq!(v.sum_along_dim(1, false), DenseTensor::<TDefault>::from_ndarray(&[3.0, 7.0, 11.0], &[3]));
    assert_eq!(v.sum_along_dim(0, false), DenseTensor::<TDefault>::from_ndarray(&[9.0, 12.0], &[2]));
    assert_eq!(v.sum_along_dim(1, true), DenseTensor::<TDefault>::from_ndarray(&[3.0, 7.0, 11.0], &[3, 1]));
    assert_eq!(v.sum_along_dim(0, true), DenseTensor::<TDefault>::from_ndarray(&[9.0, 12.0], &[1, 2]));
}


#[test]
fn test_pow() {
    let v = DenseTensor::<TDefault>::from_array(&[2.0, 3.0]);
    assert_eq!(v.pow(2), DenseTensor::<TDefault>::from_ndarray(&[4.0, 9.0], &[2]));
}


#[test]
fn test_exp() {
    let v = DenseTensor::<TDefault>::from_array(&[2.0, 3.0]);
    assert_eq!(v.exp(), DenseTensor::<TDefault>::from_ndarray(&[2.0_f32.exp(), 3.0_f32.exp()], &[2]));
}


#[test]
fn test_log() {
    let v = DenseTensor::<TDefault>::from_array(&[2.0, 3.0]);
    assert_eq!(v.exp().log(), v);
}


#[test]
fn test_max() {
    let v = DenseTensor::<TDefault>::from_ndarray(&[1.0, 2.0, 3.0, 9.0, 5.0, 6.0], &[3, 2]);
    assert_eq!(v.max().get_f64(), 9.0);
}


#[test]
fn test_max_in_dim() {
    let v = DenseTensor::<TDefault>::from_ndarray(&[5.0, 2.0, 3.0, 9.0, 5.0, 6.0], &[3, 2]);
    let (max_val, idx_val) = v.max_in_dim(1, false);
    assert_eq!(max_val, DenseTensor::<TDefault>::from_ndarray(&[5.0, 9.0, 6.0], &[3]));
    assert_eq!(idx_val, DenseTensor::<TLong>::from_ndarray(&[0, 1, 1], &[3]));
}

#[test]
fn test_swap_axes() {
    let v = DenseTensor::<TDefault>::from_ndarray(&[5.0, 2.0, 3.0, 9.0, 5.0, 6.0], &[3, 2]);
    assert_eq!(v.swap_axes(0, 1).to_2darray(), vec![vec![5.0, 3.0, 5.0], vec![2.0, 9.0, 6.0]]);
}


#[test]
fn test_transpose() {
    let v = DenseTensor::<TDefault>::from_ndarray(&[5.0, 2.0, 3.0, 9.0, 5.0, 6.0], &[3, 2]);
    assert_eq!(v.transpose().to_2darray(), vec![vec![5.0, 3.0, 5.0], vec![2.0, 9.0, 6.0]]);
}

#[test]
fn test_dot() {
    assert_eq!(DenseTensor::<TDefault>::from_array(&[3.0, 2.0, 4.0]).dot(
        &DenseTensor::<TDefault>::from_array(&[1.0, 2.0, 0.5])).get_f64(), 9.0);
}


#[test]
fn test_outer() {
    let v = DenseTensor::<TDefault>::from_array(&[2.0, 3.0]);
    let u = DenseTensor::<TDefault>::from_array(&[4.0, 2.0]);
    assert_eq!(v.outer(&u).to_2darray(), vec![
        vec![8.0, 4.0], vec![12.0, 6.0]
    ]);
}


#[test]
fn test_matmul() {
    let v = DenseTensor::<TDefault>::from_array(&[3.0, 2.0, 4.0]);
    let u = DenseTensor::<TDefault>::from_array(&[1.0, 2.0, 0.5]);
    // both are one dim
    assert_eq!(v.matmul(&u).get_f64(), 9.0);

    // both are 2-dim
    let v = DenseTensor::<TDefault>::from_ndarray(&[2.0, 3.0, 4.0, 5.0, 7.0, 9.0], &[2, 3]);
    let u = DenseTensor::<TDefault>::from_ndarray(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    assert_eq!(v.matmul(&u), v.mm(&u));

    // TODO: add more test cases
    //    + one of them is 1-dim
    //    + at least one is 1-dim and at least one is N-dim
}

#[test]
fn test_mm() {
    let v = DenseTensor::<TDefault>::from_ndarray(&[2.0, 3.0, 4.0, 5.0, 7.0, 9.0], &[2, 3]);
    let u = DenseTensor::<TDefault>::from_ndarray(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    assert_eq!(v.mm(&u).to_2darray(), vec![vec![31.0, 40.0], vec![71.0, 92.0]]);
}

#[test]
fn test_mv() {
    let v = DenseTensor::<TDefault>::from_ndarray(&[2.0, 3.0, 4.0, 5.0, 7.0, 9.0], &[2, 3]);
    let u = DenseTensor::<TDefault>::from_ndarray(&[1.0, 2.0, 4.0], &[3]);
    assert_eq!(v.mv(&u), DenseTensor::<TDefault>::from_array(&[24.0, 55.0]));
}


#[test]
fn test_expand() {
    let u = DenseTensor::<TDefault>::from_ndarray(&[4.0, 9.0], &[2, 1]);
    let v = u.expand(&vec![2, 2]);
    assert_eq!(v.to_2darray(), vec![vec![4.0, 4.0], vec![9.0, 9.0]]);
}


#[test]
fn test_view() {
    let v = DenseTensor::<TDefault>::from_ndarray(&[2.0, 3.0, 4.0, 5.0, 7.0, 9.0], &[2, 3]);
    assert_eq!(v.view(&vec![-1, 2]), DenseTensor::<TDefault>::from_ndarray(&[2.0, 3.0, 4.0, 5.0, 7.0, 9.0], &[3, 2]));
    assert_eq!(v.view(&vec![-1]),  DenseTensor::<TDefault>::from_ndarray(&[2.0, 3.0, 4.0, 5.0, 7.0, 9.0], &[6]));
}

#[test]
fn test_unbind() {
    let v = DenseTensor::<TDefault>::from_ndarray(&[2.0, 3.0, 4.0, 5.0, 7.0, 9.0], &[2, 3]);
    assert_eq!(v.unbind(1), vec![
        DenseTensor::<TDefault>::from_ndarray(&[2.0, 5.0], &[2]),
        DenseTensor::<TDefault>::from_ndarray(&[3.0, 7.0], &[2]),
        DenseTensor::<TDefault>::from_ndarray(&[4.0, 9.0], &[2]),
    ]);
    assert_eq!(v.unbind(0), vec![
        DenseTensor::<TDefault>::from_ndarray(&[2.0, 3.0, 4.0], &[3]),
        DenseTensor::<TDefault>::from_ndarray(&[5.0, 7.0, 9.0], &[3])
    ]);
}


#[test]
fn test_sigmoid() {
    let sigmoid_func = |x: f64| 1.0 / (1.0 + (-x).exp());
    let u = DenseTensor::<TDouble>::from_ndarray(&[1.0, 0.5], &[2]);
    assert_eq!(u.sigmoid(), DenseTensor::<TDouble>::from_ndarray(&[sigmoid_func(1.0), sigmoid_func(0.5)], &[2]));
}


#[test]
fn test_log_sum_exp() {
    let v = DenseTensor::<TDouble>::from_ndarray(&[1.0, 2.0, 3.0], &[3]);
    assert_eq!(v.log_sum_exp().get_f64(), 3.4076059644443806);
}


#[test]
fn test_log_sum_exp_2dim() {
    let v = DenseTensor::<TDouble>::from_ndarray(&[1.0, 2.0, 3.0, 1.0, 2.0, 3.0], &[2, 3]);
    assert_eq!(v.log_sum_exp_2dim(1).to_1darray(), vec![3.4076059644443806, 3.4076059644443806]);
}


#[test]
fn test_contiguous() {
    let v = DenseTensor::<TDefault>::from_ndarray(&[5.0, 2.0, 3.0, 9.0, 5.0, 6.0], &[3, 2]);
    assert!(!v.transpose().is_contiguous());
    v.transpose().contiguous_();
    assert!(v.is_contiguous());
}


#[test]
fn test_ndim() {
    let v = DenseTensor::<TDefault>::from_ndarray(&[5.0, 2.0, 3.0, 9.0, 5.0, 6.0], &[3, 2]);
    assert_eq!(v.size().len(), v.ndim() as usize);
}


#[test]
fn test_squeeze() {
    let v = DenseTensor::<TDefault>::from_ndarray(&[2.0, 3.0, 4.0], &[1, 3]);
    assert_eq!(v.squeeze(), DenseTensor::<TDefault>::from_ndarray(&[2.0, 3.0, 4.0], &[3]));
}


#[test]
fn test_operators() {
    let v = DenseTensor::<TDefault>::from_ndarray(&[2.0, 3.0, 4.0], &[3]);
    let u = DenseTensor::<TDefault>::from_ndarray(&[5.0, 7.0, 9.0], &[3]);

    // test operators: +, -, *, /
    assert_eq!(&v + &u, DenseTensor::<TDefault>::from_ndarray(&[7.0, 10.0, 13.0], &[3]));
    assert_eq!(&v + 5.0, DenseTensor::<TDefault>::from_ndarray(&[7.0, 8.0, 9.0], &[3]));
    assert_eq!(5.0 + &v, DenseTensor::<TDefault>::from_ndarray(&[7.0, 8.0, 9.0], &[3]));
    assert_eq!(-&v, DenseTensor::<TDefault>::from_ndarray(&[-2.0, -3.0, -4.0], &[3]));
    assert_eq!(&v - &u, DenseTensor::<TDefault>::from_ndarray(&[-3.0, -4.0, -5.0], &[3]));
    assert_eq!(&v - 5.0, DenseTensor::<TDefault>::from_ndarray(&[-3.0, -2.0, -1.0], &[3]));
    assert_eq!(5.0 - &v, DenseTensor::<TDefault>::from_ndarray(&[3.0, 2.0, 1.0], &[3]));
    assert_eq!(&v * 2.0, DenseTensor::<TDefault>::from_ndarray(&[4.0, 6.0, 8.0], &[3]));
    assert_eq!(&v * 2.0 / 2.0, v);
    assert_eq!(&u / &v, DenseTensor::<TDefault>::from_ndarray(&[5.0 / 2.0, 7.0 / 3.0, 9.0 / 4.0], &[3]));
    assert_eq!(2.0 / &v, DenseTensor::<TDefault>::from_ndarray(&[2.0 / 2.0, 2.0 / 3.0, 2.0 / 4.0], &[3]));

    // test operator +=, -=, *=, /=
    let mut k = DenseTensor::<TDefault>::zeros_like(&v);
    k += &v;
    k += &u;
    assert_eq!(k, &v + &u);
    k -= &u;
    assert_eq!(k, v);
    k *= 2.0;
    assert_eq!(k, &v * 2.0);
    k /= 2.0;
    assert_eq!(k, v);
    k -= 2.0;
    assert_eq!(k, &v - 2.0);
}


#[test]
fn test_indexing() {
    let v = DenseTensor::<TDefault>::from_ndarray(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    assert_eq!(v.at(1), DenseTensor::<TDefault>::from_ndarray(&[3.0, 4.0], &[2]));
    assert_eq!(v.at((1, 1)).get_f64(), 4.0);

    // test slice
    assert_eq!(v.at(slice![2;3]).to_2darray(), vec![vec![5.0, 6.0]]);
    assert_eq!(v.at(slice![2;3, 1;2]).to_2darray(), vec![vec![6.0]]);
    assert_eq!(v.at(slice![0;3;2, ;]).to_2darray(), vec![vec![1.0, 2.0], vec![5.0, 6.0]]);
    assert_eq!(v.at(slice![0;-1;1, ;]), v.at(slice![0;2;1, ;]));
    assert_eq!(v.at(slice![0;;2, ;]), v.at(slice![0;3;2, ;]));

    // test mixed
    assert_eq!(v.at(slice![2, 1;2]), DenseTensor::<TDefault>::from_array(&[6.0]));
    assert_eq!(v.at(slice![1;3, 1]), DenseTensor::<TDefault>::from_array(&[4.0, 6.0]));
    assert_eq!(v.at(slice![0;3;2, 0]), DenseTensor::<TDefault>::from_array(&[1.0, 5.0]));
    assert_eq!(v.at(slice![0;3;2, 1]), DenseTensor::<TDefault>::from_array(&[2.0, 6.0]));

    // test list of values
    assert_eq!(v.at(&vec![1, 0]), DenseTensor::<TDefault>::from_ndarray(&[3.0, 4.0, 1.0, 2.0], &[2, 2]));
}


#[test]
fn test_assign_indexing() {
    let v = DenseTensor::<TDefault>::from_ndarray(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    let u = DenseTensor::<TDefault>::from_ndarray(&[8.0, 3.0, 4.0, 1.0, 9.0, 2.0], &[3, 2]);

    // test assign to single value or broadcast
    let mut mv = v.clone();
    mv.assign((0, 0), 5.0);
    assert_eq!(mv.to_2darray(), vec![vec![5.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]]);
    mv.assign((1, -1), 11.0);
    assert_eq!(mv.to_2darray(), vec![vec![5.0, 2.0], vec![3.0, 11.0], vec![5.0, 6.0]]);
    mv.assign(0, 5.0);
    assert_eq!(mv.to_2darray(), vec![vec![5.0, 5.0], vec![3.0, 11.0], vec![5.0, 6.0]]);
    mv.assign(0, u.at(0));
    assert_eq!(mv.to_2darray(), vec![vec![8.0, 3.0], vec![3.0, 11.0], vec![5.0, 6.0]]);
    mv.assign(0, &v.at(0));
    assert_eq!(mv.to_2darray(), vec![vec![1.0, 2.0], vec![3.0, 11.0], vec![5.0, 6.0]]);
    mv.assign((0, 0), u.at((0, 0)));
    assert_eq!(mv.to_2darray(), vec![vec![8.0, 2.0], vec![3.0, 11.0], vec![5.0, 6.0]]);

    // test assign to a vector of values
    let mut mv = v.clone();
    mv.assign(&vec![0, 1], u.at( &vec![0, 1]));
    assert_eq!(mv.to_2darray(), vec![vec![8.0, 3.0], vec![4.0, 1.0], vec![5.0, 6.0]]);

    // test assign to slice
    let mut mv = v.clone();
    mv.assign(slice![0;3;2, ;], 5.0);
    assert_eq!(mv.to_2darray(), vec![vec![5.0, 5.0], vec![3.0, 4.0], vec![5.0, 5.0]]);
    mv.assign(slice![0;3;2, ;], u.at(slice![0;3;2, ;]));
    assert_eq!(mv.to_2darray(), vec![vec![8.0, 3.0], vec![3.0, 4.0], vec![9.0, 2.0]]);

    // test assign mix
    let mut mv = v.clone();
    mv.assign(slice![0;3;2, 1], 100.0);
    assert_eq!(mv.to_2darray(), vec![vec![1.0, 100.0], vec![3.0, 4.0], vec![5.0, 100.0]]);
}


#[test]
fn test_pickling() {
    let v = DenseTensor::<TDefault>::from_ndarray(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    let dumped_v: Vec<u8> = bincode::serialize(&v).unwrap();

    let u = bincode::deserialize(&dumped_v).unwrap();
    assert_ne!(&v as *const _, &u as *const _);
    assert_eq!(v, u);
}


