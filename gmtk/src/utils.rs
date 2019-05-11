use std;

pub struct RefEquality<'a, T: 'a>(pub &'a T);

impl<'a, T> std::hash::Hash for RefEquality<'a, T> {
    fn hash<H>(&self, state: &mut H)
        where H: std::hash::Hasher
    {
        (self.0 as *const T).hash(state)
    }
}

impl<'a, 'b, T> PartialEq<RefEquality<'b, T>> for RefEquality<'a, T> {
    fn eq(&self, other: &RefEquality<T>) -> bool {
        self.0 as *const T == other.0 as *const T
    }
}
impl<'a, T> Eq for RefEquality<'a, T> {}