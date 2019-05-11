pub trait IterableContainer {
    type Element;

    fn get_element(&self, idx: usize) -> &Self::Element;
}

pub struct IterContainer<'a, C: 'a + IterableContainer> {
    current_idx: usize,
    elements: &'a [usize],
    container: &'a C
}

impl<'a, C: 'a + IterableContainer> IterContainer<'a, C> {
    pub fn new(elements: &'a [usize], container: &'a C) -> IterContainer<'a, C> {
        IterContainer {
            current_idx: 0,
            elements,
            container
        }
    }
}

impl<'a, C: 'a + IterableContainer> Iterator for IterContainer<'a, C> {
    type Item = &'a <C as IterableContainer>::Element;

    fn next(&mut self) -> Option<Self::Item> {
        let elem = if self.current_idx < self.elements.len() {
            Some(self.container.get_element(self.current_idx))
        } else {
            None
        };

        self.current_idx += 1;
        elem
    }
}