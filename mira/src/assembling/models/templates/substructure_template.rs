use algorithm::data_structure::graph::*;
use assembling::features::structures::duplication_tensor::DuplicationTensor;
use assembling::features::CardinalityFeatures;
use assembling::features::LocalStructure;
use assembling::features::PrimaryKey;
use assembling::models::example::MRRExample;
use assembling::models::factors::sufficient_sub_factor::{SufficientSubFactor, SufficientSubFactorType};
use assembling::models::variable::*;
use gmtk::prelude::*;
use settings::Settings;
use assembling::models::annotator::Annotator;
use std::collections::HashSet;
use assembling::features::*;
use settings::conf_mrf::TemplatesConf;


pub type DuplicationPairwiseDomain = BinaryVectorDomain<String>;
pub type PkPairwiseDomain = BinaryVectorDomain<String>;
pub type CooccurrenceDomain = BinaryVectorDomain<String>;

// TODO: duplication weights should be refactor to be a dictionary
#[derive(Serialize, Deserialize)]
pub struct SubstructureTemplate {
    mrf_max_n_props: usize,
    enable_coocurrence_factors: bool,
    enable_duplication_factors: bool,

    weights: Vec<Weights>,

    #[serde(skip)]
    local_structure: LocalStructure,
    #[serde(skip)]
    primary_key: PrimaryKey,
    #[serde(skip)]
    attr_scope: AttributeScope,
    #[serde(skip)]
    cooccurrence_features: CooccurrenceFeatures,

    #[serde(skip)]
    dup_pairwise_domain: DuplicationPairwiseDomain,
    #[serde(skip)]
    pk_pairwise_domain: PkPairwiseDomain,
    #[serde(skip)]
    cooccurrence_domain: CooccurrenceDomain,

    #[serde(skip)]
    cardinality: CardinalityFeatures,
    #[serde(skip)]
    duplication_tensors: DuplicationTensor,
    #[serde(skip)]
    pairwise_indice_func_tensor: DenseTensor,

    #[serde(skip)]
    stype_assistant: STypeAssistant,

    boolean_domain: BooleanVectorDomain,
}

impl SubstructureTemplate {
    pub fn new(
        templates_config: &TemplatesConf,
        weights: Vec<Weights>,
        local_structure: LocalStructure,
        dup_pairwise_domain: DuplicationPairwiseDomain,
        pk_pairwise_domain: PkPairwiseDomain,
        cooccurrence_domain: CooccurrenceDomain,
        primary_key: PrimaryKey,
        cardinality: CardinalityFeatures,
        duplication_tensors: DuplicationTensor,
        attr_scope: AttributeScope,
        stype_assistant: STypeAssistant,
        cooccurrence_features: CooccurrenceFeatures,
    ) -> SubstructureTemplate {
        let pairwise_indice_func_tensor = DenseTensor::from_ndarray(&[
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0
        ], &[4, 4, 1]);

        SubstructureTemplate {
            enable_coocurrence_factors: templates_config.enable_cooccurrence_factors,
            enable_duplication_factors: templates_config.enable_duplication_factors,
            weights,
            local_structure,
            primary_key,
            cardinality,
            dup_pairwise_domain,
            pk_pairwise_domain,
            cooccurrence_domain,
            duplication_tensors,
            mrf_max_n_props: Settings::get_instance().mrf.max_n_props,
            pairwise_indice_func_tensor,
            attr_scope,
            cooccurrence_features,
            stype_assistant,
            boolean_domain: BooleanVectorDomain::new(),
        }
    }

    pub fn default(
        templates_config: &TemplatesConf,
        dup_pairwise_domain: DuplicationPairwiseDomain,
        pk_pairwise_domain: PkPairwiseDomain,
        cooccurrence_domain: CooccurrenceDomain,
        local_structure: LocalStructure,
        primary_key: PrimaryKey,
        cardinality: CardinalityFeatures,
        duplication_tensors: DuplicationTensor,
        attr_scope: AttributeScope,
        stype_assistant: STypeAssistant,
        cooccurrence_features: CooccurrenceFeatures,
    ) -> SubstructureTemplate {
        let default_weights = vec![
            Weights::new(DenseTensor::zeros(&[2])), // all children weights
            Weights::new(DenseTensor::zeros(&[4 * pk_pairwise_domain.numel() as i64])), // pairwise_pk_weights
            Weights::new(DenseTensor::zeros(&[4 * 2])), // pairwise scope weight
            Weights::new(DenseTensor::zeros(
                &[DuplicationTensor::get_n_features() * dup_pairwise_domain.numel() as i64],
            )), // duplication tensors
            Weights::new(DenseTensor::zeros(&[4 * cooccurrence_domain.numel() as i64])), // cooccurrence weights
            Weights::new(DenseTensor::zeros(&[4 * 2])), // stype assistant weights
        ];

        SubstructureTemplate::new(
            templates_config,
            default_weights,
            local_structure,
            dup_pairwise_domain,
            pk_pairwise_domain,
            cooccurrence_domain,
            primary_key,
            cardinality,
            duplication_tensors,
            attr_scope,
            stype_assistant,
            cooccurrence_features,
        )
    }

    pub fn get_dup_domain<'a>(annotator: &Annotator<'a>) -> DuplicationPairwiseDomain {
        let mut catset: HashSet<String> = Default::default();
        for sm in annotator.sms {
            for n in sm.graph.iter_class_nodes() {
                for e in n.iter_outgoing_edges(&sm.graph) {
                    catset.insert(format!("source={},predicate={}", n.label, e.label));
                }
            }
        }

        let cats: Vec<String> = catset.into_iter().collect();
        DuplicationPairwiseDomain::new(cats)
    }

    pub fn get_cooccurrence_domain<'a>(annotator: &Annotator<'a>) -> CooccurrenceDomain {
        let mut catset: HashSet<String> = Default::default();
        for (class_uri, top_correlated_pred) in &annotator.cooccurrence.top_corelated_pred {
            let node_structure_space = &annotator.local_structure.node_structure_space[class_uri];

            for (pred_a, pred_b) in top_correlated_pred {
                // Note that we need to make it consistent with node structure
                if node_structure_space.has_smaller_index(pred_a, pred_b) {
                    catset.insert(format!("source={},pred_a={},pred_b={}.true", class_uri, pred_a, pred_b));
                    catset.insert(format!("source={},pred_a={},pred_b={}.false", class_uri, pred_a, pred_b));
                } else {
                    catset.insert(format!("source={},pred_a={},pred_b={}.true", class_uri, pred_b, pred_a));
                    catset.insert(format!("source={},pred_a={},pred_b={}.false", class_uri, pred_b, pred_a));
                }
            }
        }

        let cats: Vec<String> = catset.into_iter().collect();
        DuplicationPairwiseDomain::new(cats)
    }

    pub fn get_pk_domain<'a>(annotator: &Annotator<'a>) -> PkPairwiseDomain {
        let mut catset: HashSet<String> = Default::default();
        for sm in annotator.sms {
            for n in sm.graph.iter_class_nodes() {
                if !annotator.primary_key.contains(&n.label) {
                    continue;
                }

                let primary_key = annotator.primary_key.get_primary_key(&n.label);
                if primary_key == "karma:classLink" {
                    // special treat for karma:classLink
                    if n.iter_outgoing_edges(&sm.graph).any(|e| e.label == primary_key) {
                        // has primary key
                        for e in n.iter_outgoing_edges(&sm.graph) {
                            if e.label != primary_key {
                                for cardinality in Cardinality::iterator() {
                                    catset.insert(format!("pk={},attr={},cardinality={}",
                                                          primary_key, e.label, cardinality.as_str()
                                    ));
                                }
                            }
                        }
                    }
                } else {
                    if n.iter_outgoing_edges(&sm.graph).any(|e| e.label == primary_key) {
                        // has primary key
                        for e in n.iter_outgoing_edges(&sm.graph) {
                            if e.label != primary_key {
                                for cardinality in Cardinality::iterator() {
                                    catset.insert(format!("source={},pk={},attr={},cardinality={}",
                                                          n.label, primary_key, e.label, cardinality.as_str()
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }
        let cats: Vec<String> = catset.into_iter().collect();
        PkPairwiseDomain::new(cats)
    }

    fn get_children_weights(&self) -> &Weights {
        &self.weights[0]
    }

    fn get_pk_weights(&self) -> &Weights {
        &self.weights[1]
    }

    fn get_pk_scope_weights(&self) -> &Weights {
        &self.weights[2]
    }

    fn get_duplication_weights(&self) -> &Weights {
        &self.weights[3]
    }

    fn get_cooccurrence_weights(&self) -> &Weights {
        &self.weights[4]
    }

    fn get_stype_assistant_weights(&self) -> &Weights {
        &self.weights[5]
    }

    fn prepare_args(&self, example: &MRRExample, n: &Node) -> Vec<(usize, usize)> {
        // prepare list of variables, and its corresponding index
        let node_structure = &self.local_structure.node_structure_space[&n.label];
        let mut var_with_ids = Vec::new();

        if n.n_incoming_edges > 0 {
            // have parents
            let parent_edge = n.first_incoming_edge(&example.graph).unwrap();
            let parent = parent_edge.get_source_node(&example.graph);
            let corresponding_id = node_structure.get_parent_idx(&parent_edge.label, &parent.label);

            if corresponding_id.is_some() {
                var_with_ids.push((parent_edge.id, *corresponding_id.unwrap()));
            }
        }

        for e in n.iter_outgoing_edges(&example.graph) {
            let target = e.get_target_node(&example.graph);
            let corresponding_id = node_structure.get_child_idx(
                &e.label,
                if target.is_data_node() {
                    "DATA_NODE"
                } else {
                    &target.label
                },
            );

            if corresponding_id.is_some() {
                var_with_ids.push((e.id, *corresponding_id.unwrap()));
            }
        }

        // low index means high frequency, because parent_idx < child_idx, first element always parent
        var_with_ids.sort_by(|a, b| a.1.cmp(&b.1));
        var_with_ids.truncate(self.mrf_max_n_props);

        var_with_ids
    }

    fn create_pairwise_factors<'a: 'a1, 'a1>(&'a self, example: &'a1 MRRExample, triple_vars: &Vec<&'a TripleVar>, variables: &[(usize, usize)], center_node: &Node) -> Vec<Box<SubTensorFactor<'a1, TripleVar> + 'a1>> {
        let mut factors: Vec<Box<SubTensorFactor<'a1, TripleVar> + 'a1>> = Vec::new();
        if !self.primary_key.contains(&center_node.label) {
            return factors;
        }

        let pk = self.primary_key.get_primary_key(&center_node.label);
        let idx = variables
            .iter()
            .enumerate()
            .find(|&(idx, &(eid, pred_idx))| example.graph.get_edge_by_id(eid).label == pk);

        if let Some((idx, &(pk_var_id, pred_idx))) = idx {
            let pk_col = &example.graph.get_edge_by_id(pk_var_id).get_target_node(&example.graph).label;
            let skip = (center_node.n_incoming_edges > 0) as usize;

            for (i, &(var_id, pred_idx)) in variables.iter().enumerate().skip(skip) {
                if i == idx {
                    continue;
                }

                let var = example.graph.get_edge_by_id(var_id);
                let x_target = var.get_target_node(&example.graph);
                let cardin = if x_target.is_data_node() {
                    self.cardinality.get_cardinality(example.sm_idx, pk_col, &x_target.label)
                } else {
                    // if x_target is class node, we will compare between 2 primary keys
                    // TODO: fix me, here we assume the primary keys of next class is correct, but it may not
                    let dpk = self.primary_key.get_primary_key(&x_target.label);
                    let e = x_target
                        .iter_outgoing_edges(&example.graph)
                        .find(|&e| e.label == dpk);

                    if e.is_none() {
                        continue;
                    }

                    self.cardinality.get_cardinality(example.sm_idx, pk_col, &e.unwrap().get_target_node(&example.graph).label)
                };

                let cat = if pk == "karma:classLink" {
                    // special treat for karma:classLink
                    format!("pk={},attr={},cardinality={}", pk, var.label, cardin.as_str())
                } else {
                    format!("source={},pk={},attr={},cardinality={}", &center_node.label, pk, var.label, cardin.as_str())
                };

                if self.pk_pairwise_domain.has_category(&cat) {
                    // TODO: we always assume the first colume is pk, and the second colume is not.
                    // so we need to generate it consistently
                    let features_tensor = self.pairwise_indice_func_tensor.matmul(&self.pk_pairwise_domain.encode_value(&cat).tensor.view(&[1, 1, -1])).view(&[4, -1]);
                    let input_var_idx = if idx <= i {
                        [idx, i]
                    } else {
                        [i, idx]
                    };

                    factors.push(Box::new(SufficientSubFactor::new(
                        SufficientSubFactorType::PairwisePKFactor,
                        triple_vars.clone(),
                        self.get_pk_weights(),
                        input_var_idx.to_vec(),
                        features_tensor,
                    )));
                }
            }
        }

        return factors;
    }

    fn create_pairwise_scope_factors<'a: 'a1, 'a1>(&'a self, example: &'a1 MRRExample, triple_vars: &Vec<&'a TripleVar>, variables: &[(usize, usize)], center_node: &Node) -> Vec<Box<SubTensorFactor<'a1, TripleVar> + 'a1>> {
        let mut factors: Vec<Box<SubTensorFactor<'a1, TripleVar> + 'a1>> = Vec::new();
        let skip = (center_node.n_incoming_edges > 0) as usize;

        // [OLD CODE] we generate N*(N-1) factors (full
        // [NEW CODE] we generate 1 or N factors, where N is #attributes and N > 2 (ring structure)
        let data_var_index = (skip..(variables.len() - 1))
            .filter(|&i| example.graph.get_edge_by_id(variables[i].0).get_target_node(&example.graph).is_data_node())
            .collect::<Vec<_>>();

        if data_var_index.len() == 0 {
            return factors;
        }

        for i in 0..data_var_index.len() - 1 {
            let x_target = example.graph.get_edge_by_id(variables[data_var_index[i]].0).get_target_node(&example.graph);
            let y_target = example.graph.get_edge_by_id(variables[data_var_index[i + 1]].0).get_target_node(&example.graph);

            let val = self.boolean_domain.encode_value(&self.attr_scope.is_same_scope(example.sm_idx, &x_target.label, &y_target.label));
            let features_tensor = self.pairwise_indice_func_tensor.matmul(&val.tensor.view(&[1, 1, -1])).view(&[4, -1]);
            factors.push(Box::new(SufficientSubFactor::new(
                SufficientSubFactorType::PairwiseScopeFactor,
                triple_vars.clone(),
                self.get_pk_scope_weights(),
                vec![data_var_index[i], data_var_index[i + 1]],
                features_tensor,
            )));
        }

        return factors;
    }

    fn create_occurrence_factor<'a: 'a1, 'a1>(&'a self, example: &'a1 MRRExample, triple_vars: &Vec<&'a TripleVar>, variables: &[(usize, usize)], center_node: &Node) -> Vec<Box<SubTensorFactor<'a1, TripleVar> + 'a1>> {
        let mut factors: Vec<Box<SubTensorFactor<'a1, TripleVar> + 'a1>> = Vec::new();
        let skip = (center_node.n_incoming_edges > 0) as usize;

        // generate co-occurrence for n * (n - 1), but we will filter out to keep top n, or min 0.5, because we want it linear with
        // the number of attributes (too many attributes, can make this factor dominant other factors)
        let mut sorted_cooccurrence = vec![];
        for i in skip..variables.len() {
            let e = example.graph.get_edge_by_id(variables[i].0);
            for j in (i + 1)..variables.len() {
                let e_j = example.graph.get_edge_by_id(variables[j].0);
                if e.label == e_j.label {
                    continue;
                }

                if let Some(prob) = self.cooccurrence_features.get_occurrence_prob_min_support(&center_node.label, &e.label, &e_j.label) {
                    sorted_cooccurrence.push((i, j, prob));
                }
            }
        }

        sorted_cooccurrence.sort_by(|a, b| b.partial_cmp(&a).unwrap());
        for &(i, j, score) in &sorted_cooccurrence[..variables.len().min(sorted_cooccurrence.len())] {
            let pred_a = &example.graph.get_edge_by_id(variables[i].0).label;
            let pred_b = &example.graph.get_edge_by_id(variables[j].0).label;

            let mut builder = ObservedFeaturesBuilder::<String>::new();
            builder += (format!("source={},pred_a={},pred_b={}.true", center_node.label, pred_a, pred_b), score);
            builder += (format!("source={},pred_a={},pred_b={}.false", center_node.label, pred_a, pred_b), (1.0 - score).max(0.01));

            let tensor = builder.create_tensor(&self.cooccurrence_domain).view(&[1, 1, -1]);
            let features_tensor = self.pairwise_indice_func_tensor.matmul(&tensor).view(&[4, -1]);

            factors.push(Box::new(SufficientSubFactor::new(
                SufficientSubFactorType::PairwiseCooccurrenceFactor,
                triple_vars.clone(),
                self.get_cooccurrence_weights(),
                vec![i, j],
                features_tensor,
            )));
        }

        factors
    }

    fn create_stype_assistant_factor<'a: 'a1, 'a1>(&'a self, example: &'a1 MRRExample, triple_vars: &Vec<&'a TripleVar>, variables: &[(usize, usize)], center_node: &Node) -> Vec<Box<SubTensorFactor<'a1, TripleVar> + 'a1>> {
        let mut factors: Vec<Box<SubTensorFactor<'a1, TripleVar> + 'a1>> = Vec::new();
        debug_assert!(center_node.n_incoming_edges > 0);

        let parent_edge = example.graph.get_edge_by_id(variables[0].0);
        let parent = parent_edge.get_source_node(&example.graph);

        for (i, &(var_id, pred_idx)) in variables.iter().enumerate().skip(1) {
            let e = example.graph.get_edge_by_id(var_id);
            let target = e.get_target_node(&example.graph);
            if target.is_class_node() {
                continue;
            }

            let potential_gain = self.stype_assistant.get_potential_gain(
                example.sm_idx, &target.label, &center_node.label, &e.label,
                &parent.label, &parent_edge.label
            );
            if let Some(potential_gain) = potential_gain {
                let val = DenseTensor::from_ndarray(&[potential_gain, (1.0 - potential_gain).max(0.01)], &[1, 1,  2]);
                let features_tensor = self.pairwise_indice_func_tensor.matmul(&val).view(&[4, -1]);

//                let mut features_tensor = DenseTensor::<TFloat>::zeros(&[4, 2]);
//                features_tensor.assign(3, [potential_gain, (1.0 - potential_gain).max(0.01)]);

                factors.push(Box::new(SufficientSubFactor::new(
                    SufficientSubFactorType::STypeAssistantFactor,
                    triple_vars.clone(),
                    self.get_stype_assistant_weights(),
                    vec![0, i],
                    features_tensor,
                )));
            }
        }

        factors
    }

    fn create_all_children_wrong_factor<'a: 'a1, 'a1>(&'a self, example: &'a1 MRRExample, triple_vars: &Vec<&'a TripleVar>, variables: &[(usize, usize)], center_node: &Node) -> Box<SubTensorFactor<'a1, TripleVar> + 'a1> {
        let weights = self.get_children_weights();
        let mut features_tensor = DenseTensor::zeros(&[2, 2_i64.pow(variables.len() as u32 - 1), weights.get_value().size()[0]]);
        features_tensor.assign((0, 0, 0), 1.0);
        features_tensor.assign((1, 0, 1), 1.0);

        Box::new(SufficientSubFactor::new(
            SufficientSubFactorType::AllChildrenWrongFactor,
            triple_vars.clone(),
            weights,
            (0..variables.len()).collect(),
            features_tensor,
        ))
    }

    fn create_duplication_factors<'a: 'a1, 'a1>(
        &'a self,
        example: &'a1 MRRExample,
        triple_vars: &Vec<&'a TripleVar>,
        variables: &[(usize, usize)],
        center_node: &Node,
    ) -> Vec<Box<SubTensorFactor<'a1, TripleVar> + 'a1>> {
        let mut dup_range = [0, 1];
        let mut factors: Vec<Box<SubTensorFactor<'a1, TripleVar> + 'a1>> = Vec::new();

        if center_node.n_incoming_edges > 0 {
            // skip parent, start at 1
            dup_range[0] = 1;
            dup_range[1] = 2;
        }

        // dup_range[1] will be 1 if no parent, 2 if has parent
        for (i, &(id, idx)) in variables.iter().enumerate().skip(dup_range[1]) {
            if variables[i - 1].1 == idx {
                dup_range[1] += 1;
            } else {
                if dup_range[1] - dup_range[0] > 1 {
                    let mut tensor = self.duplication_tensors.get_tensor(
                        &center_node.label,
                        variables[i - 1].1,
                        dup_range[1] - dup_range[0],
                    );
                    // MODIFIED here to have different weight for different label
                    let pairwise_cat = format!(
                        "source={},predicate={}",
                        center_node.label,
                        example.graph.get_edge_by_id(variables[i - 1].0).label
                    );
                    let pairwise_val = self.dup_pairwise_domain.encode_value(&pairwise_cat);
                    tensor = tensor
                        .view(&[-1, 6, 1])
                        .matmul(&pairwise_val.tensor.view(&[1, -1]));
                    // END MODIFIED
                    factors.push(Box::new(SufficientSubFactor::new(
                        SufficientSubFactorType::DuplicationFactor,
                        triple_vars.clone(),
                        self.get_duplication_weights(),
                        (dup_range[0]..dup_range[1]).collect(),
                        tensor,
                    )));
                }
                dup_range[0] = i;
                dup_range[1] = i + 1;
            }
        }

        if dup_range[1] - dup_range[0] > 1 {
            let mut tensor = self.duplication_tensors.get_tensor(
                &center_node.label,
                variables[variables.len() - 1].1,
                dup_range[1] - dup_range[0],
            );
            // MODIFIED here to have different weight for different label
            let pairwise_cat = format!(
                "source={},predicate={}",
                center_node.label,
                example
                    .graph
                    .get_edge_by_id(variables[variables.len() - 1].0)
                    .label
            );
            let pairwise_val = self.dup_pairwise_domain.encode_value(&pairwise_cat);
            tensor = tensor
                .view(&[-1, 6, 1])
                .matmul(&pairwise_val.tensor.view(&[1, -1]));
            // END MODIFIED
            factors.push(Box::new(SufficientSubFactor::new(
                SufficientSubFactorType::DuplicationFactor,
                triple_vars.clone(),
                self.get_duplication_weights(),
                (dup_range[0]..dup_range[1]).collect(),
                tensor,
            )));
        }

        return factors;
    }
}

impl FactorTemplate<MRRExample, TripleVar> for SubstructureTemplate {
    fn get_weights(&self) -> &[Weights] {
        &self.weights
    }

    fn unroll<'a: 'a1, 'a1>(
        &'a self,
        example: &'a1 MRRExample,
    ) -> Vec<Box<Factor<'a1, TripleVar> + 'a1>> {
        let mut factors: Vec<Box<Factor<'a1, TripleVar> + 'a1>> =
            Vec::with_capacity(example.variables.len());
        let g = &example.graph;
        for e in g.iter_edges() {
            let target = e.get_target_node(&g);
            if target.n_outgoing_edges == 0 && !example.root_triple_id == e.id {
                // skip if don't have any children or not root triples
                continue;
            }

            if example.root_triple_id == e.id && example.sibling_index.get_n_siblings(target) > 0 {
                let source = e.get_source_node(&g);
                let variables = self.prepare_args(example, source);
                let triple_vars = variables
                    .iter()
                    .map(|idx| &example.variables[idx.0])
                    .collect();

                let mut sub_factors = vec![];
                sub_factors.extend(self.create_pairwise_factors(example, &triple_vars, &variables, source));
                if example.does_sm_has_hierachy {
                    sub_factors.extend(self.create_pairwise_scope_factors(example, &triple_vars, &variables, source));
                }
                sub_factors.extend(self.create_duplication_factors(example, &triple_vars, &variables, source));
                sub_factors.extend(self.create_occurrence_factor(example, &triple_vars, &variables, source));

                let group_factor = GroupTensorFactor::new(
                    triple_vars,
                    sub_factors,
                );

                factors.push(Box::new(group_factor));
            }

            if target.n_outgoing_edges > 1 {
                let variables = self.prepare_args(example, target);
                let triple_vars = variables
                    .iter()
                    .map(|idx| &example.variables[idx.0])
                    .collect();
                let mut sub_factors = vec![];
                sub_factors.extend(self.create_pairwise_factors(example, &triple_vars, &variables, target));
                if example.does_sm_has_hierachy {
                    sub_factors.extend(self.create_pairwise_scope_factors(example, &triple_vars, &variables, target));
                }
                sub_factors.extend(self.create_duplication_factors(example, &triple_vars, &variables, target));
                sub_factors.extend(self.create_occurrence_factor(example, &triple_vars, &variables, target));
                sub_factors.extend(self.create_stype_assistant_factor(example, &triple_vars, &variables, target));
                sub_factors.push(self.create_all_children_wrong_factor(example, &triple_vars, &variables, target));

                let group_factor = GroupTensorFactor::new(
                    triple_vars,
                    sub_factors,
                );

                factors.push(Box::new(group_factor));
            }
        }

        return factors;
    }
}
