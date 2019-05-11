use assembling::models::example::MRRExample;
use assembling::models::variable::*;
use assembling::models::annotator::Annotator;
use algorithm::data_structure::graph::Graph;
use assembling::models::templates::*;
use gmtk::prelude::*;
use rayon::prelude::*;
use fnv::FnvHashMap;
use settings::Settings;
use settings::conf_mrf::*;
use rand::prelude::*;
use utils::Timer;

use super::mrr_helper::*;
use assembling::features::*;

pub struct MRRModel<'a> {
    pub dataset: String,
    pub annotator: Annotator<'a>,
    pub model: LogLinearModel<MRRExample, TripleVar>,
    pub tf_domain: TFDomain,
    pub dup_pairwise_domain: DuplicationPairwiseDomain,
    pub pk_pairwise_domain: PkPairwiseDomain,
    pub cooccur_domain: CooccurrenceDomain,
    pub templates_conf: TemplatesConf
}

impl<'a> MRRModel<'a> {
    pub fn new(dataset: &str, annotator: Annotator<'a>, templates_conf: TemplatesConf, model: LogLinearModel<MRRExample, TripleVar>, tf_domain: TFDomain, dup_pairwise_domain: DuplicationPairwiseDomain, pk_pairwise_domain: PkPairwiseDomain, cooccur_domain: CooccurrenceDomain) -> MRRModel<'a> {
        MRRModel {
            dataset: dataset.to_owned(),
            model,
            tf_domain,
            annotator,
            dup_pairwise_domain,
            pk_pairwise_domain,
            cooccur_domain,
            templates_conf
        }
    }

    pub fn predict_sm_probs(&self, sm_id: &str, graphs: Vec<Graph>) -> Vec<(Graph, f64)> {
        graphs.into_par_iter()
            .map(|g| {
                let mut example = self.annotator.create_mrr_example(sm_id, g);
                let score = {
                    self.annotator.annotate(&mut example, &self.tf_domain);
                    let factors = self.model.get_factors(&example);
                    let mut inference = BeliefPropagation::new(InferProb::MARGINAL, &example.variables, &factors, 120);

                    let target_assignment: FnvHashMap<usize, TripleVarValue> = example.variables
                        .iter()
                        .map(|v| (v.get_id(), self.annotator.var_domain.get_value(1))).collect();

                    inference.infer();

                    let log_z = inference.log_z();
                    factors.iter().map(|f| f.score_assignment(&target_assignment)).sum::<f64>() - log_z
                };

                return (example.graph, score.exp());
            })
            .collect::<Vec<_>>()
    }

    pub fn predict_sm_labels(&self, sm_id: &str, graphs: Vec<Graph>) -> Vec<(Graph, Vec<bool>)> {
        graphs.into_par_iter()
            .map(|g| {
                let mut example = self.annotator.create_mrr_example(sm_id, g);

                let map_result = {
                    self.annotator.annotate(&mut example, &self.tf_domain);
                    let factors = self.model.get_factors(&example);
                    let mut inference = BeliefPropagation::new(InferProb::MAP, &example.variables, &factors, 120);
                    inference.infer();

                    inference.map()
                };
                (
                    example.graph,
                    (0..example.variables.len())
                        .map(|i| map_result[&i].idx == 1)
                        .collect::<Vec<_>>()
                )
            })
            .collect::<Vec<_>>()
    }

    pub(super) fn create_loglinear_model(tf_domain: &TFDomain, dup_domain: &DuplicationPairwiseDomain, pk_domain: &PkPairwiseDomain, cooccur_domain: &CooccurrenceDomain, annotator: &Annotator, templates_config: &TemplatesConf) -> LogLinearModel<MRRExample, TripleVar> {
        let primary_key = annotator.primary_key.clone();
        let cardinality = annotator.cardinality.clone();
        let dup_tensors = DuplicationTensor::new(&annotator.stats_predicate, &annotator.local_structure);

        LogLinearModel::new(vec![
            Box::new(TripleTemplate::default(templates_config, tf_domain)),
            Box::new(SubstructureTemplate::default(
                templates_config,
                dup_domain.clone(), pk_domain.clone(), cooccur_domain.clone(),
                annotator.local_structure.clone(), primary_key, cardinality,
                dup_tensors, annotator.attr_scope.clone(), annotator.stype_assistant.clone(),
                annotator.cooccurrence.clone()))
        ])
    }

    pub fn train(mut annotator: Annotator<'a>, train_examples: &mut Vec<MRRExample>, test_examples: &mut Vec<MRRExample>, mrf_config: &MRFConf) -> MRRModel<'a> {
        let mut timer = Timer::start();

        let train_args = &mrf_config.training_args;
        let mut rng = StdRng::from_seed([Settings::get_instance().manual_seed; 32]);

        DenseTensor::<TDefault>::manual_seed(train_args.manual_seed);

        annotator.train(train_examples);

        let tf_domain = annotator.train_annotate(train_examples);
        let dup_pairwise_domain = SubstructureTemplate::get_dup_domain(&annotator);
        let pk_pairwise_domain = SubstructureTemplate::get_pk_domain(&annotator);
        let cooccur_domain = SubstructureTemplate::get_cooccurrence_domain(&annotator);

        test_examples.iter_mut().for_each(|example| annotator.annotate(example, &tf_domain));

        let model = MRRModel::create_loglinear_model(
            &tf_domain,
            &dup_pairwise_domain,
            &pk_pairwise_domain,
            &cooccur_domain,
            &annotator,
            &mrf_config.templates);
        let mut early_stopping = EarlyStopping::new(train_args.early_stopping.min_delta, train_args.early_stopping.patience);

        info!("Train size: {}. Test size: {}. #templates: {}", train_examples.len(), test_examples.len(), model.templates.len());
        debug!("Training with following configuration: {:?}", train_args);
        {
            let train_factorss: Vec<_> = train_examples.iter()
                .map(|e| model.get_factors(e))
                .collect();
            let test_factorss: Vec<_> = test_examples.iter()
                .map(|e| model.get_factors(e))
                .collect();

            debug!("Preprocessing take: {:.5}s", timer.lap().0);

            let mut train_nll_examples = train_examples.iter().enumerate()
                .map(|(i, e)| {
                    NLLExample::new(
                        &e.variables,
                        &train_factorss[i],
                        Box::new(BeliefPropagation::new(InferProb::MARGINAL, &e.variables, &train_factorss[i], 120)
                        ))
                })
                .collect::<Vec<_>>();
            let test_nll_examples = test_examples.iter().enumerate()
                .map(|(i, e)| {
                    NLLExample::new(
                        &e.variables,
                        &test_factorss[i],
                        Box::new(BeliefPropagation::new(InferProb::MARGINAL, &e.variables, &test_factorss[i], 120)
                        ))
                })
                .collect::<Vec<_>>();

            let mut train_map_examples = train_examples.iter().enumerate()
                .map(|(i, e)| {
                    MAPExample::new(
                        &e.variables,
                        &train_factorss[i],
                        Box::new(BeliefPropagation::new(InferProb::MAP, &e.variables, &train_factorss[i], 120)
                        ))
                })
                .collect::<Vec<_>>();
            let mut test_map_examples = test_examples.iter().enumerate()
                .map(|(i, e)| {
                    MAPExample::new(
                        &e.variables,
                        &test_factorss[i],
                        Box::new(BeliefPropagation::new(InferProb::MAP, &e.variables, &test_factorss[i], 120)
                        ))
                })
                .collect::<Vec<_>>();

            let mut optimizer: Box<Optimizer<TDefault>> = match train_args.optimizer {
                OptimizerAlgo::BasicGradientDescent => {
                    Box::new(BasicGradientDescent::new(model.get_parameters(), train_args.optparams.lr))
                }
                OptimizerAlgo::Adam => {
                    let default_params = Adam::<TDefault>::default_params();
                    Box::new(Adam::<TDefault>::new(
                        model.get_parameters(), train_args.optparams.lr,
                        default_params.betas, train_args.optparams.eps,
                        train_args.optparams.weight_decays.clone(),
                        true,
                    ))
                }
                OptimizerAlgo::SGD => {
                    let default_params = SGD::<TDefault>::default_params();
                    Box::new(SGD::<TDefault>::new(
                        model.get_parameters(), train_args.optparams.lr,
                        0.9, train_args.optparams.weight_decays.clone(),
                        default_params.dampening, default_params.nesterov,
                    ))
                }
            };

            let mut average_loss_val = Vec::new();
            for i in 0..train_args.n_epoch {
                trace!("Epoch: {}", i);

                if train_args.shuffle_mini_batch && i < train_args.n_switch {
                    // shuffle examples
                    rng.shuffle(&mut train_nll_examples);
                }

                let mut examples = if i >= train_args.n_switch {
                    // train in batch
                    vec![if train_args.parallel_training {
                        ExampleEnum::ParallelBatchNLL(ParallelBatchNLLExample::new(&mut train_nll_examples))
                    } else {
                        ExampleEnum::BatchNLL(BatchNLLExample::new(&mut train_nll_examples))
                    }]
                } else {
                    // train in mini-batch
                    let _groups = split_random(&mut train_nll_examples, train_args.mini_batch_size);
                    if train_args.parallel_training {
                        _groups.into_iter()
                            .map(|x| ExampleEnum::ParallelBatchNLL(ParallelBatchNLLExample { examples: x }))
                            .collect::<Vec<_>>()
                    } else {
                        _groups.into_iter()
                            .map(|x| ExampleEnum::BatchNLL(BatchNLLExample { examples: x }))
                            .collect::<Vec<_>>()
                    }
                };

                average_loss_val.clear();
                for example in examples.iter_mut() {
                    optimizer.zero_grad();
                    {
                        let (loss_accum, grad_accum) = optimizer.get_loss_and_gradient_accum();
                        example.accumulate_value_and_gradient(loss_accum, grad_accum);
                    }

                    optimizer.average(example.size());
                    optimizer.step();

                    average_loss_val.push(optimizer.get_loss_accum().get_value());
                    trace!("    - Accum loss         = {:.10}", optimizer.get_loss_accum().get_value());
                }

                if optimizer.get_loss_accum().get_value() < 0.0 {
                    // this is invalid/bug in our code
                    error!("Bug in our code");
                    break;
                }

                if average_loss_val.len() > 1 {
                    trace!("    - Average accum loss = {:.10}", average_loss_val.iter().sum::<f64>() / average_loss_val.len() as f64);
                }

                if let Some(best_params) = early_stopping.should_stop_w_weight(average_loss_val.iter().sum::<f64>() / average_loss_val.len() as f64, model.clone_parameters()) {
                    // early stopping
                    debug!(
                        "Stop training because loss increase/doesn't seem to improve much. Recent loss history: {:?}",
                        early_stopping.recent_loss_history()
                    );
                    model.set_parameters(&best_params);
                    break;
                }

                if i % train_args.n_iter_eval == 0 || i == train_args.n_epoch - 1 {
                    let cm_train = evaluate(&mut train_map_examples, &annotator.var_domain).unwrap();
                    let cm_test = evaluate(&mut test_map_examples, &annotator.var_domain);

                    trace!("train (class_idx=0): {:?}", cm_train.precision_recall_fbeta("false", 1.0));
                    trace!("train (class_idx=1): {:?}", cm_train.precision_recall_fbeta("true", 1.0));
                    if cm_test.is_some() {
                        trace!("test  (class_idx=0): {:?}", cm_test.as_ref().unwrap().precision_recall_fbeta("false", 1.0));
                        trace!("test  (class_idx=1): {:?}", cm_test.as_ref().unwrap().precision_recall_fbeta("true", 1.0));
                    }
                }
            }

            if train_args.report_final_loss {
                let cm_train = evaluate(&mut train_map_examples, &annotator.var_domain).unwrap();
                let cm_test = evaluate(&mut test_map_examples, &annotator.var_domain);

                info!("Final average loss = {:.10}", average_loss_val.iter().sum::<f64>() / average_loss_val.len() as f64);
                info!("{}", cm_train.pretty_print("** TRAIN **"));
                if cm_test.is_some() {
                    info!("{}", cm_test.unwrap().pretty_print("** TEST  **"));
                }
            }
        }

        MRRModel::new(
            annotator.dataset,
            annotator,
            mrf_config.templates.clone(),
            model,
            tf_domain,
            dup_pairwise_domain,
            pk_pairwise_domain,
            cooccur_domain,
        )
    }
}