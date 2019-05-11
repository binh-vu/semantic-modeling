use optimization::sample_model::*;
use gmtk::optimization::example::NLLExample;
use gmtk::tensors::*;
use gmtk::graph_models::*;
use gmtk::optimization::accumulators::*;
use fnv::FnvHashMap;
use gmtk::optimization::numerical_gradient::compute_approx_gradient;
use gmtk::optimization::batch_example::*;

fn get_log_p<'a>(model: &LogLinearModel<Sentence<'a>, Token<'a>, TDouble>, example: &Sentence<'a>) -> f64 {
    let factors = model.get_factors(example);
    let target_assignment = example.variables.iter().map(|v| (v.get_id(), v.get_label_value().clone())).collect();

    let mut brute_force = BruteForce::new(&example.variables, &factors);
    brute_force.infer();

    - (factors.iter().map(|f| f.score_assignment(&target_assignment)).sum::<f64>() - brute_force.log_z())
}

fn nll_func<'a>(example: &mut NLLExample<Token<'a>, TDouble>) -> f64 {
    let mut loss_accum = ValueAccumulator::new();
    example.accumulate_value(&mut loss_accum);
    return loss_accum.get_value();
}

#[test]
pub fn test_nll_loss_val() {
    build_model(|model, train_examples, test_examples| {
        let mut loss_val_accum = ValueAccumulator::new();
        let mut gradient_accum = Tensor1AccumulatorDict::new();
        for param in model.get_parameters() {
            gradient_accum.track_object(param.id, &param.get_value());
        }

        for example in train_examples.iter().chain(test_examples.iter()) {
            let factors = model.get_factors(&example);
            let inference = BeliefPropagation::new(InferProb::MARGINAL, &example.variables, &factors, 120);
            let mut nll_example = NLLExample::new(&example.variables, &factors, Box::new(inference));

            loss_val_accum.clear();
            nll_example.accumulate_value_and_gradient(&mut loss_val_accum, &mut gradient_accum);
            let log_p = loss_val_accum.get_value();
            assert!(log_p > 0.0);
            assert!(log_p - get_log_p(model, example) < 1e-9);
        }
    });
}

#[test]
pub fn test_nll_gradient() {
    build_model(|model, train_examples, test_examples| {
        let mut loss_val_accum = ValueAccumulator::new();
        let mut gradient_accum = Tensor1AccumulatorDict::new();
        for param in model.get_parameters() {
            gradient_accum.track_object(param.id, &param.get_value());
        }

        for example in train_examples.iter().chain(test_examples.iter()) {
            let factors = model.get_factors(&example);
            let inference = BeliefPropagation::new(InferProb::MARGINAL, &example.variables, &factors, 120);
            let mut nll_example = NLLExample::new(&example.variables, &factors, Box::new(inference));

            loss_val_accum.clear();
            gradient_accum.clear();
            nll_example.accumulate_value_and_gradient(&mut loss_val_accum, &mut gradient_accum);

            for param in model.get_parameters() {
                let gradient = gradient_accum.get_value(&param.id);
                let approx_gradients = compute_approx_gradient(param, || nll_func(&mut nll_example), 1e-5);

                assert_eq!(gradient.size(), approx_gradients.size());
                for i in 0..gradient.size()[0] {
                    assert!(gradient.at(i).get_f64() - approx_gradients.at(i).get_f64() < 1e-6);
                }
            }
            let log_p = loss_val_accum.get_value();
            assert!(log_p > 0.0);
            assert!(log_p - get_log_p(model, example) < 1e-9);
        }
    });
}

#[test]
pub fn test_batch_nll_example() {
    build_model(|model, train_examples, test_examples| {
        let mut loss_val_accum = ValueAccumulator::new();
        let mut gradient_accum = Tensor1AccumulatorDict::new();
        for param in model.get_parameters() {
            gradient_accum.track_object(param.id, &param.get_value());
        }

        let mut factorss = Vec::new();
        let mut nll_examples = Vec::new();

        for example in train_examples.iter().chain(test_examples.iter()) {
            factorss.push(model.get_factors(example));
        }
        for (i, example) in train_examples.iter().chain(test_examples.iter()).enumerate() {
            nll_examples.push(NLLExample::new(&example.variables, &factorss[i], Box::new(
                BeliefPropagation::new(InferProb::MARGINAL, &example.variables, &factorss[i], 120)
            )));
        }

        loss_val_accum.clear();
        gradient_accum.clear();
        for example in nll_examples.iter_mut() {
            example.accumulate_value_and_gradient(&mut loss_val_accum, &mut gradient_accum);
        }
        let total_loss_val = loss_val_accum.get_value();
        let mut gradients: FnvHashMap<i64, DenseTensor<TDouble>> = Default::default();
        for param in model.get_parameters() {
            gradients.insert(param.id, gradient_accum.get_value(&param.id).clone_reference());
        }

        loss_val_accum.clear();
        gradient_accum.clear();
        let mut batch_nll_examples = BatchNLLExample::new(&mut nll_examples);
        batch_nll_examples.accumulate_value_and_gradient(&mut loss_val_accum, &mut gradient_accum);
        assert_eq!(total_loss_val, loss_val_accum.get_value());
        for param in model.get_parameters() {
            assert_eq!(&gradients[&param.id], gradient_accum.get_value(&param.id));
        }
    });
}


#[test]
pub fn test_parallel_batch_nll_example() {
    build_model(|model, train_examples, test_examples| {
        let mut loss_val_accum = ValueAccumulator::new();
        let mut gradient_accum = Tensor1AccumulatorDict::new();
        for param in model.get_parameters() {
            gradient_accum.track_object(param.id, &param.get_value());
        }

        let mut factorss = Vec::new();
        let mut nll_examples = Vec::new();

        for example in train_examples.iter().chain(test_examples.iter()) {
            factorss.push(model.get_factors(example));
        }
        for (i, example) in train_examples.iter().chain(test_examples.iter()).enumerate() {
            nll_examples.push(NLLExample::new(&example.variables, &factorss[i], Box::new(
                BeliefPropagation::new(InferProb::MARGINAL, &example.variables, &factorss[i], 120)
            )));
        }

        loss_val_accum.clear();
        gradient_accum.clear();
        for example in nll_examples.iter_mut() {
            example.accumulate_value_and_gradient(&mut loss_val_accum, &mut gradient_accum);
        }
        let total_loss_val = loss_val_accum.get_value();
        let mut gradients: FnvHashMap<i64, DenseTensor<TDouble>> = Default::default();
        for param in model.get_parameters() {
            gradients.insert(param.id, gradient_accum.get_value(&param.id).clone_reference());
        }

        loss_val_accum.clear();
        gradient_accum.clear();

        let mut parallel_examples = ParallelBatchNLLExample::new(&mut nll_examples);
        parallel_examples.accumulate_value_and_gradient(&mut loss_val_accum, &mut gradient_accum);

        assert!(total_loss_val - loss_val_accum.get_value() < 1e-9);
        for param in model.get_parameters() {
            assert_eq!(&gradients[&param.id], gradient_accum.get_value(&param.id));
        }
    });
}