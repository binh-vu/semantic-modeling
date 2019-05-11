use std::ops::Index;
use std::ops::IndexMut;
use std::fmt::Debug;

/// This is a 2d array, the reason we have it is because ndarray package throw a memory allocation errors
#[derive(Debug)]
pub struct Matrix<T: Default + Clone + Debug + PartialEq + Eq> {
    value: Vec<T>,
    n_row: usize,
    n_col: usize
}

impl<T: Default + Clone + Debug + PartialEq + Eq> Matrix<T> {
    pub fn new(n_row: usize, n_col: usize) -> Matrix<T> {
        Matrix {
            value: vec![T::default(); n_row * n_col],
            n_row,
            n_col,
        }
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.n_row, self.n_col)
    }

    pub fn is_equal(&self, arr2: &[Vec<T>]) -> bool {
        if arr2.len() != self.n_row {
            return false;
        }

        for i in 0..self.n_row {
            if arr2[i].len() != self.n_col {
                return false;
            }

            for j in 0..self.n_col {
                if &arr2[i][j] != self.get(i, j) {
                    return false;
                }
            }
        }

        return true;
    }

    #[inline]
    pub fn set(&mut self, i: usize, j: usize, val: T) {
        let idx = self.n_row * i + j;
        self.value[idx] = val;
    }

    #[inline]
    pub fn get(&self, i: usize, j: usize) -> &T {
        &self.value[self.n_col * i + j]
    }
}

impl<T: Default + Clone + Debug + PartialEq + Eq> Index<(usize, usize)> for Matrix<T> {
    type Output = T;

    #[inline]
    fn index<'a>(&'a self, idx: (usize, usize)) -> &'a T {
        &self.value[self.n_col * idx.0 + idx.1]
    }
}

impl<T: Default + Clone + Debug + PartialEq + Eq> IndexMut<(usize, usize)> for Matrix<T> {

    #[inline]
    fn index_mut(&mut self, idx: (usize, usize)) -> &mut T {
        &mut self.value[self.n_col * idx.0 + idx.1]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    pub fn test_matrix() {
        for &(n_rows, n_cols) in [(5, 7), (3, 2), (7, 3)].iter() {
            let mut matrix = Matrix::<usize>::new(n_rows, n_cols);
            let mut arr2: Vec<Vec<usize>> = Vec::new();

            for i in 0..n_rows {
                arr2.push(Vec::new());
                for j in 0..n_cols {
                    let val = i * n_rows * 2 + j;
                    arr2[i].push(val);
                    matrix[(i, j)] = val;
                }
            }

            for i in 0..n_rows {
                for j in 0..n_cols {
                    assert_eq!(matrix[(i, j)], arr2[i][j]);
                }
            }
        }
    }
}