use gmtk::graph_models::*;
use gmtk::tensors::*;
use fnv::FnvHashMap;

pub struct Token<'a, T: 'static + TensorType=TDouble> {
    pub value: BinaryVectorValue<T>,
    pub labeled_value: BinaryVectorValue<T>,
    pub domain: &'a BinaryVectorDomain<String, T>
}

impl<'a, T: 'static + TensorType> Token<'a, T> {
    pub fn new(domain: &'a BinaryVectorDomain<String, T>, value: BinaryVectorValue<T>) -> Token<'a, T> {
        Token {
            labeled_value: value.clone(),
            value,
            domain
        }
    }
}

impl<'a, T: 'static + TensorType> Variable for Token<'a, T> {
    type Value = BinaryVectorValue<T>;

    fn get_id(&self) -> usize {
        (self as *const _) as usize
    }

    fn get_domain_size(&self) -> i64 {
        self.domain.numel() as i64
    }

    fn get_domain(&self) -> &Domain<Value=<Self as Variable>::Value> {
        self.domain
    }

    fn set_value(&mut self, val: <Self as Variable>::Value) {
        self.value = val;
    }

    fn get_value(&self) -> &<Self as Variable>::Value {
        &self.value
    }
}

impl<'a, T: 'static + TensorType> LabeledVariable for Token<'a, T> {
    fn get_label_value(&self) -> &<Self as Variable>::Value {
        &self.labeled_value
    }
}

pub struct Sentence<'a, T: 'static + TensorType=TDouble> {
    pub variables: Vec<Token<'a, T>>,
    pub sentence: Vec<String>,
    pub observed_features: Vec<DenseTensor<T>>
}

pub struct PYGivenX<'a, 'a0: 'a, T: 'static + TensorType=TDouble> {
    pub weights: &'a Weights<T>,
    pub vars: [&'a Token<'a0, T>; 1],
    pub vars_dims: Vec<i64>,
    pub features_tensor: DenseTensor<T>
}

impl<'a, 'a0: 'a, T: 'static + TensorType> PYGivenX<'a, 'a0, T> {
    pub fn new(var: &'a Token<'a0, T>, observed_features: &'a DenseTensor<T>, weights: &'a Weights<T>, domain_tensor: &DenseTensor<T>) -> PYGivenX<'a, 'a0, T> {
        let domain_numel = domain_tensor.size()[0];

        let features_tensor = domain_tensor.view(&[domain_numel, domain_numel, 1])
            .matmul(&observed_features.view(&[1, 1, -1]))
            .view(&[domain_numel, -1]);

        PYGivenX {
            vars: [var],
            vars_dims: vec![var.get_domain_size()],
            weights,
            features_tensor
        }
    }
}

impl<'a, 'a0: 'a, T: 'static + TensorType> Factor<'a, Token<'a0, T>, T> for PYGivenX<'a, 'a0, T> {
    fn get_variables(&self) -> &[&'a Token<'a0, T>] {
        &self.vars
    }

    fn score_assignment(&self, assignment: &FnvHashMap<usize, <Token<'a0, T> as Variable>::Value>) -> f64 {
        self.impl_score_assignment(assignment)
    }

    fn get_scores_tensor(&self) -> DenseTensor<T> {
        self.impl_get_scores_tensor()
    }

    fn compute_gradients(&self, target_assignment: &FnvHashMap<usize, <Token<'a0, T> as Variable>::Value>, inference: &Inference<'a, Token<'a0, T>, T>) -> Vec<(i64, DenseTensor<T>)> {
        self.impl_compute_gradients(target_assignment, inference)
    }

    fn touch(&self, var: &Token<T>) -> bool {
        self.vars[0].get_id() == var.get_id()
    }
}

impl<'a, 'a0: 'a, T: 'static + TensorType> DotTensor1Factor<'a, Token<'a0, T>, T> for PYGivenX<'a, 'a0, T> {
    #[inline]
    fn get_weights(&self) -> &Weights<T> {
        self.weights
    }

    fn get_features_tensor(&self) -> &DenseTensor<T> {
        &self.features_tensor
    }

    fn get_vars_dims(&self) -> &[i64] {
        &self.vars_dims
    }
}

pub struct PYGivenXTemplate<T: 'static + TensorType=TDouble> {
    weights: Vec<Weights<T>>,
    domain_tensor: DenseTensor<T>
}

impl<T: 'static + TensorType> PYGivenXTemplate<T> {
    pub fn default(label_domain: &BinaryVectorDomain<String, T>, word_domain: BinaryVectorDomain<String, T>) -> PYGivenXTemplate<T> {
        let default_weight = Weights::new(DenseTensor::<T>::create_randn(&[label_domain.numel() as i64 * word_domain.numel() as i64]));
        PYGivenXTemplate {
            weights: vec![default_weight],
            domain_tensor: label_domain.get_domain_tensor().clone_reference()
        }
    }
}

impl<'ax, T: 'static + TensorType> FactorTemplate<Sentence<'ax, T>, Token<'ax, T>, T> for PYGivenXTemplate<T> {
    fn get_weights(&self) -> &[Weights<T>] {
        &self.weights
    }

    fn unroll<'a: 'a1, 'a1>(&'a self, example: &'a1 Sentence<'ax, T>) -> Vec<Box<Factor<'a1, Token<'ax, T>, T> + 'a1>> {
        let mut factors: Vec<Box<Factor<'a1, Token<'ax, T>, T> + 'a1>> = Vec::with_capacity(example.variables.len());
        for i in 0..example.variables.len() {
            factors.push(Box::new(PYGivenX::new(
                &example.variables[i],
                &example.observed_features[i],
                &self.weights[0],
                &self.domain_tensor
            )));
        }

        factors
    }
}

pub fn extract_features<T: TensorType>(examples: &mut [Sentence<T>], domain: Option<&BinaryVectorDomain<String, T>>) -> Option<BinaryVectorDomain<String, T>> {
    let mut builders = Vec::with_capacity(examples.iter().map(|e| e.variables.len()).sum());
    for example in examples.iter_mut() {
        for i in 0..example.variables.len() {
            let mut builder = ObservedFeaturesBuilder::<String, T>::new();
            builder += format!("w={}", example.sentence[i]);
            if i > 0 {
                builder += format!("w[-1]={}", example.sentence[i - 1]);
            }
            if i < example.variables.len() - 1 {
                builder += format!("w[+1]={}", example.sentence[i + 1]);
            }
            builders.push(builder);
        }
    }

    let new_domain: Option<BinaryVectorDomain<String, T>> = if domain.is_none() {
        Some(ObservedFeaturesBuilder::<String, T>::create_domain(&mut builders))
    } else {
        None
    };

    {
        let maker_domain = match domain {
            None => new_domain.as_ref().unwrap(),
            Some(x) => x
        };
        let mut counter = 0;
        for example in examples.iter_mut() {
            for _i in 0..example.variables.len() {
                example.observed_features.push(builders[counter].create_tensor(maker_domain));
                counter += 1;
            }
        }
    }

    return new_domain;
}


pub fn build_model<F>(func: F)
    where F: for <'a> Fn(&LogLinearModel<Sentence<'a>, Token<'a>, TDouble>, &[Sentence<'a>], &[Sentence<'a>]) -> ()
{
    let train_sentences = ["I love football", "I hate seafood", "he need you"];
    let test_sentences = ["she need him", "we love seafood"];

    let mut label_domain = BinaryVectorDomain::new(vec!["S".to_owned(), "P".to_owned(), "O".to_owned()]);
    label_domain.compute_domain_tensor();
    let ref_label_domain = &label_domain;

    // make training examples
    let mut train_examples = train_sentences.iter().map(|&s| {
        let variables = s.split(' ').enumerate().map(|(i, _w)| {
            Token::new(ref_label_domain, label_domain.get_value(i))
        }).collect();

        Sentence {
            variables,
            sentence: s.split(' ').map(|w| w.to_owned()).collect(),
            observed_features: Vec::new()
        }
    }).collect::<Vec<Sentence>>();

    let mut test_examples = test_sentences.iter().map(|&s| {
        let variables = s.split(' ').enumerate().map(|(i, _w)| {
            Token::new(ref_label_domain, label_domain.get_value(i))
        }).collect();

        Sentence {
            variables,
            sentence: s.split(' ').map(|w| w.to_owned()).collect(),
            observed_features: Vec::new()
        }
    }).collect::<Vec<Sentence>>();

    let word_domain = extract_features(&mut train_examples, None).unwrap();
    extract_features(&mut test_examples, Some(&word_domain));

    let model = LogLinearModel::<Sentence, Token, TDouble>::new(vec![
        Box::new(PYGivenXTemplate::default(&label_domain, word_domain))
    ]);

    func(&model, &train_examples, &test_examples);
}