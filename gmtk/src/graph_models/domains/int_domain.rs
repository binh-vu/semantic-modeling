use graph_models::traits::Domain;

pub struct IntDomain {
    min: i32,
    max: i32,
}

impl IntDomain {
    pub fn new(min: i32, max: i32) -> IntDomain {
        return IntDomain { min, max }
    }
}

impl Domain for IntDomain {
    type Value = i32;

    fn numel(&self) -> usize {
        (self.max - self.min + 1) as usize
    }

    fn get_index(&self, &value: &<Self as Domain>::Value) -> usize {
        assert!(self.min <= value && value <= self.max, format!("Invalid value: {} <= {} <= {}", self.min, value, self.max));
        return (value - self.min) as usize;
    }

    fn get_value(&self, index: usize) -> <Self as Domain>::Value {
        assert!(index <= (self.max - self.min) as usize);
        return self.min + index as i32;
    }
}