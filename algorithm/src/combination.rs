use std::iter::Iterator;

pub struct IterIndexImpl<'a> {
    is_start: bool,
    is_end: bool,
    index: *mut IterIndex,
    current_val_index: &'a Vec<usize>
}

pub struct IterIndex {
    array_lengths: Vec<usize>,
    current_val_index: Vec<usize>,
}

impl IterIndex {
    pub fn new(array_lengths: Vec<usize>) -> IterIndex {
        IterIndex {
            current_val_index: vec![0; array_lengths.len()],
            array_lengths,
        }
    }

    pub fn iter<'a>(&'a mut self) -> IterIndexImpl<'a> {
        IterIndexImpl {
            is_start: true,
            is_end: false,
            index: self,
            current_val_index: &self.current_val_index
        }
    }
}

impl<'a> Iterator for IterIndexImpl<'a> {
    type Item = &'a Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.is_start {
            self.is_start = false;
            return Some(self.current_val_index);
        }

        if self.is_end {
            return None;
        }

        let mut i = self.current_val_index.len() - 1;
        loop {
            // move to next state & set value of variables to next state value
            unsafe { 
                (&mut *self.index).current_val_index[i] += 1;
                if (*self.index).current_val_index[i] == (*self.index).array_lengths[i] {
                    (&mut *self.index).current_val_index[i] = 0;
                    if i == 0 {
                        self.is_end = true;
                        return None;
                    }

                    i -= 1;
                } else {
                    break;
                }
            }
        }

        Some(self.current_val_index)
    }
}