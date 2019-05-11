pub fn unravel_index(mut ravelled_dim: i64, dims: &[i64]) -> Vec<i64> {
    let mut index = Vec::with_capacity(dims.len());
    let mut sizes = vec![0; dims.len()];

    sizes[dims.len() - 1] = 1;
    for i in (0..(dims.len() - 1)).rev() {
        sizes[i] = dims[i + 1] * sizes[i + 1];
    }

    for i in 0..dims.len() {
        index.push(ravelled_dim / sizes[i]);
        ravelled_dim = ravelled_dim % sizes[i];
    }

    index
}

pub fn unravel_index_ptr(mut ravelled_dim: i64, dims: &[i64], n_dim: usize) -> Vec<i64> {
    let mut index = Vec::with_capacity(n_dim);
    let mut sizes = vec![0; n_dim];

    sizes[n_dim - 1] = 1;
    for i in (0..(n_dim - 1)).rev() {
        sizes[i] = dims[i + 1] * sizes[i + 1];
    }

    for i in 0..n_dim {
        index.push(ravelled_dim / sizes[i]);
        ravelled_dim = ravelled_dim % sizes[i];
    }

    index
}

/// Formular: index: [a1, a2, a3, ..., a_n], dims: [d1, d2, d3, ..., d_n]
/// Output: a_n + a_(n-1) * d_(n) + a_(n-2) * d_(n) * d_(n-1) + ... + a1 * d_n * ... * d2
pub fn ravel_index(index: &Vec<i64>, dims: &[i64]) -> i64 {
    let mut ravelled_idx = index[index.len() - 1];
    let mut accum_dimsize = 1;

    for i in (0..(index.len() - 1)).rev() {
        accum_dimsize *= dims[i + 1];
        ravelled_idx += accum_dimsize * index[i];
    }

    ravelled_idx
}