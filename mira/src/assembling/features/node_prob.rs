use assembling::models::example::MRRExample;
use algorithm::data_structure::graph::*;
use assembling::features::stats_predicate::StatsPredicate;
use gmtk::tensors::*;
use std::fs::File;
use std::io::*;
use serde_json;
use std::process::Command;
use std::path::PathBuf;
use std::path::Path;
use std::fs::remove_file;
use std::sync::Arc;

#[derive(Clone)]
pub struct NodeProb {
    classifier: Classifier,
    scaler: Scaler,
    stat_predicate: Arc<StatsPredicate>
}

impl NodeProb {
    pub fn new(stat_predicate: Arc<StatsPredicate>) -> NodeProb {
        NodeProb {
            classifier: Classifier::new(),
            scaler: Scaler::new(),
            stat_predicate,
        }
    }

    pub fn reload(&mut self, workdir: &Path) {
        let mut result: PyNodeProbExchange = serde_json::from_reader(
            File::open(workdir.join("node_prob.json"))
                .expect("NodeProb's file not found")
            ).expect("Invalid format");
        result.classifier.update_();
        result.scaler.update_();

        self.classifier = result.classifier;
        self.scaler = result.scaler;
    }

    pub fn train(&mut self, _dataset: &str, workdir: &Path, train_examples: &[MRRExample]) {
        // create training data first;
        let feature = Features::concat(train_examples.iter()
            .map(|e| self.extract_feature_with_label(e))
            .collect());

        let (classifier, scaler) = train(feature, workdir);
        self.classifier = classifier;
        self.scaler = scaler;

        // save model
        self.save(workdir);
    }

    pub fn extract_feature_with_label(&self, example: &MRRExample) -> Features {
        let g = &example.graph;
        let mut features = Features::new(vec!["prob_node", "minimum_merge_cost"]);
        let label = example.label.as_ref().expect("Example need to be labeled before");

        for node in g.iter_class_nodes() {
            let prob_node: f32 = node.iter_outgoing_edges(g)
                .map(|e| e.get_target_node(g))
                .filter(|&n| n.is_data_node())
                .map(|n| example.node_features[n.id].stype_score.unwrap_or(0.0))
                .sum();

            let minimum_merge_cost = g.iter_nodes_by_label(&node.label)
                .map(|n| NodeProb::get_merged_cost(g, node, n, &self.stat_predicate))
                .fold(0./0., f32::min);

            features.add_example(node.id as i32, vec![prob_node, minimum_merge_cost], label.bijection.has_x(node.id));
        }

        features
    }

    pub fn extract_feature_without_label(&self, example: &MRRExample) -> Features {
        let g = &example.graph;
        let mut features = Features::new(vec!["prob_node", "minimum_merge_cost"]);

        for node in g.iter_class_nodes() {
            let prob_node: f32 = node.iter_outgoing_edges(g)
                .map(|e| e.get_target_node(g))
                .filter(|&n| n.is_data_node())
                .map(|n| example.node_features[n.id].stype_score.unwrap_or(0.0))
                .sum();

            let minimum_merge_cost = g.iter_nodes_by_label(&node.label)
                .map(|n| NodeProb::get_merged_cost(g, node, n, &self.stat_predicate))
                .fold(0./0., f32::min);

            features.add_example(node.id as i32, vec![prob_node, minimum_merge_cost], true);
        }

        features
    }

    pub fn update_node_features(&self, example: &mut MRRExample) {
        let features = self.extract_feature_without_label(example);
        let tensor = self.scaler.transform(features.to_tensor());
        let y_pred = self.classifier.predict_proba(&tensor);

        for (node_id, prob) in features.provenance.iter().zip(y_pred.iter()) {
            example.node_features[*node_id as usize].node_prob = Some(*prob);
        }
    }

    fn get_merged_cost(g: &Graph, node_a: &Node, node_b: &Node, stats_predicate: &StatsPredicate) -> f32 {
        if node_a.n_outgoing_edges > node_b.n_outgoing_edges {
            return 1e6;
        }

        // TODO: how about the case that # their nodes is equal ??
        let pseudo_outgoing_links = StatsPredicate::count_edge_label(node_a.iter_outgoing_edges(g).chain(node_b.iter_outgoing_edges(g)));
        let mut total_cost = 0.0;
        for (lbl, count) in pseudo_outgoing_links {
            let cost = match stats_predicate.prob_multi_val(lbl, count) {
                None => 0.0,
                Some(x) => x.max(1e-6).ln()
            };

            total_cost += cost;
        }

        total_cost
    }

    fn save(&self, workdir: &Path) {
        let result = json!({
            "classifier": &self.classifier,
            "scaler": &self.scaler,
            "result": []
        });

        serde_json::to_writer_pretty(File::create(workdir.join("node_prob.json")).unwrap(), &result).unwrap()
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Classifier {
    coef: Vec<f32>,
    intercept: f32,
    #[serde(skip)]
    coef_tensor: DenseTensor
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Scaler {
    mean: Vec<f32>,
    scale: Vec<f32>,
    #[serde(skip)]
    mean_tensor: DenseTensor,
    #[serde(skip)]
    scale_tensor: DenseTensor
}

#[derive(Serialize)]
pub struct Features {
    names: Vec<&'static str>,
    provenance: Vec<i32>,
    labels: Vec<bool>,
    matrix: Vec<Vec<f32>>
}

#[derive(Deserialize, Serialize)]
pub struct PyNodeProbExchange {
    scaler: Scaler,
    classifier: Classifier,
    // #[serde(skip)]
    result: Vec<f32>
}

fn train(feature: Features, workdir: &Path) -> (Classifier, Scaler) {
    // TODO: fix me, won't work if we deliver it as binary object (loss python code)
    let foutput = workdir.join("sm_node_prob_tt8tty1.output.json");
    if foutput.exists() {
        remove_file(&foutput).unwrap();
    }

    let mut writer = BufWriter::new(File::create(workdir.join("sm_node_prob_tt8tty1.json")).unwrap());
    serde_json::to_writer(writer, &feature).unwrap();

    let mut py_node_prob = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    py_node_prob.push("src/assembling/features/node_prob.py");

    let output = Command::new("python")
        .arg(py_node_prob.as_os_str())
        .arg(workdir.to_str().unwrap())
        .output()
        .expect("Fail to execute python process");
    assert!(output.status.success(), "Fail to execute `node_prob` process. Reason: {}", String::from_utf8(output.stderr).unwrap());

    let mut reader = BufReader::new(File::open(foutput).expect("Fail in python process"));
    let mut result: PyNodeProbExchange = serde_json::from_reader(reader).unwrap();
    result.classifier.update_();
    result.scaler.update_();

    // TODO: uncomment to debug, also should unskip result in serialization
    // println!("Predict result: {:?}", &result.classifier.predict_proba(&result.scaler.transform(feature.to_tensor()))[..50]);
    // println!("Result: {:?}", &result.result[..50]);

    (result.classifier, result.scaler)
}

impl Classifier {
    pub fn new() -> Classifier {
        Classifier {
            coef: Vec::new(),
            intercept: 0.0,
            coef_tensor: Default::default()
        }
    }

    pub fn update_(&mut self) {
        self.coef_tensor = DenseTensor::borrow_from_array(&self.coef);
    }

    pub fn predict_proba(&self, tensor: &DenseTensor) -> Vec<f32> {
        (tensor.mv(&self.coef_tensor) + self.intercept).sigmoid().to_1darray()
    }
}

impl Scaler {
    pub fn new() -> Scaler {
        Scaler {
            mean: Vec::new(),
            scale: Vec::new(),
            mean_tensor: Default::default(),
            scale_tensor: Default::default()
        }
    }

    pub fn update_(&mut self) {
        self.mean_tensor = DenseTensor::borrow_from_array(&self.mean);
        self.scale_tensor = DenseTensor::borrow_from_array(&self.scale);
    }

    pub fn transform(&self, mut feature: DenseTensor) -> DenseTensor {
        feature -= &self.mean_tensor;
        feature /= &self.scale_tensor;

        feature
    }
}

impl Features {
    pub fn new(names: Vec<&'static str>) -> Features {
        Features {
            names,
            provenance: Vec::new(),
            labels: Vec::new(),
            matrix: Vec::new()
        }
    }

    pub fn add_example(&mut self, provenance: i32, values: Vec<f32>, label: bool) {
        self.provenance.push(provenance);
        self.matrix.push(values);
        self.labels.push(label);
    }

    pub fn to_tensor(&self) -> DenseTensor {
        let mut data: Vec<f32> = Vec::with_capacity(self.matrix.iter().map(|r| r.len()).sum());
        for row in &self.matrix {
            data.extend(row);
        }

        DenseTensor::from_ndarray(&data, &[self.matrix.len() as i64, self.names.len() as i64])
    }

    pub fn concat(mut features: Vec<Features>) -> Features {
        let mut feature = features.pop().unwrap();

        for mut f in features.into_iter() {
            feature.provenance.append(&mut f.provenance);
            feature.matrix.append(&mut f.matrix);
            feature.labels.append(&mut f.labels);
        }

        feature
    }
}