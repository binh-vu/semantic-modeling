use tensors::*;
use graph_models::domains::BinaryVectorValue;
use graph_models::traits::Domain;
use graph_models::domains::binary_vector_domain::TBinaryVectorDomain;


#[derive(Serialize, Deserialize)]
pub struct BooleanVectorDomain<T: TensorType=TDefault> {
    pub values: Vec<BinaryVectorValue<T>>,
    pub domain_tensor: DenseTensor<T>,
    pub category: [bool; 2]
}

impl<T: TensorType> BooleanVectorDomain<T> {
    pub fn new() -> BooleanVectorDomain<T> {
        let mut false_tensor = DenseTensor::<T>::zeros(&vec![2]);
        false_tensor.assign(0, 1.0);

        let mut true_tensor = DenseTensor::<T>::zeros(&vec![2]);
        true_tensor.assign(1, 1.0);

        let domain_tensor = DenseTensor::stack_ref(&vec![
            &false_tensor, &true_tensor
        ], 0);

        return BooleanVectorDomain {
            values: vec![
                BinaryVectorValue {
                    tensor: false_tensor,
                    idx: 0
                },
                BinaryVectorValue {
                    tensor: true_tensor,
                    idx: 1
                },
            ],
            domain_tensor,
            category: [false, true]
        }
    }
}

impl<U: TensorType> Domain for BooleanVectorDomain<U> {
    type Value = BinaryVectorValue<U>;

    fn numel(&self) -> usize {
        2
    }

    fn get_index(&self, value: &<Self as Domain>::Value) -> usize {
        return value.idx;
    }

    fn get_value(&self, idx: usize) -> <Self as Domain>::Value {
        return self.values[idx].clone();
    }
}

impl<T: TensorType> TBinaryVectorDomain<bool, T> for BooleanVectorDomain<T> {
    fn get_category(&self, idx: usize) -> &bool {
        &self.category[idx]
    }

    fn get_category_index(&self, category: &bool) -> usize {
        return *category as usize;
    }

    fn has_category(&self, _category: &bool) -> bool {
        true
    }

    fn encode_value(&self, category: &bool) -> BinaryVectorValue<T> {
        self.get_value(*category as usize)
    }

    fn get_domain_tensor(&self) -> &DenseTensor<T> {
        &self.domain_tensor
    }

    fn cuda_(&mut self) {
        unimplemented!()
    }

    fn cpu_(&mut self) {
        unimplemented!()
    }
}