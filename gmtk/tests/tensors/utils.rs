use gmtk::tensors::utils::*;

#[test]
#[allow(unused_mut)]
fn test_unravel_index() {
    assert_eq!(unravel_index(0, &vec![7]), vec![0]);
    assert_eq!(unravel_index(5, &vec![7]), vec![5]);
    assert_eq!(unravel_index(0, &vec![7, 6]), vec![0, 0]);
    assert_eq!(unravel_index(22, &vec![7, 6]), vec![3, 4]);
    assert_eq!(unravel_index(41, &vec![7, 6]), vec![6, 5]);
    assert_eq!(unravel_index(37, &vec![7, 6]), vec![6, 1]);
    assert_eq!(unravel_index(41, &vec![5, 7, 3]), vec![1, 6, 2]);
    assert_eq!(unravel_index(20, &vec![5, 7, 3]), vec![0, 6, 2]);
    assert_eq!(unravel_index(10, &vec![5, 7, 3]), vec![0, 3, 1]);
    assert_eq!(unravel_index(0, &vec![5, 7, 3]), vec![0, 0, 0]);

    // test passed variables won't change its value
    let mut idx = 37;
    assert_eq!(unravel_index(idx, &vec![7, 6]), vec![6, 1]);
    assert_eq!(idx, 37);
}

#[test]
fn test_ravel_index() {
    assert_eq!(ravel_index(&vec![0], &vec![7]), 0);
    assert_eq!(ravel_index(&vec![5], &vec![7]), 5);
    assert_eq!(ravel_index(&vec![0, 0], &vec![7, 6]), 0);
    assert_eq!(ravel_index(&vec![3, 4], &vec![7, 6]), 22);
    assert_eq!(ravel_index(&vec![6, 5], &vec![7, 6]), 41);
    assert_eq!(ravel_index(&vec![6, 1], &vec![7, 6]), 37);
    assert_eq!(ravel_index(&vec![1, 6, 2], &vec![5, 7, 3]), 41);
    assert_eq!(ravel_index(&vec![0, 6, 2], &vec![5, 7, 3]), 20);
    assert_eq!(ravel_index(&vec![0, 3, 1], &vec![5, 7, 3]), 10);
    assert_eq!(ravel_index(&vec![0, 0, 0], &vec![5, 7, 3]), 0);
}