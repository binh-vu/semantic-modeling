use tensors::*;
use std::collections::HashMap;
use std::ops::*;
use std::cmp;
use fnv::FnvHashMap;
use graph_models::*;
use algorithm::string::center;

pub struct ConfusionMatrix {
    pub matrix: DenseTensor<TLong>,
    pub n_classes: usize,
    pub class_names: Vec<String>,
    pub name2index: HashMap<String, i64>
}

fn _safe_div(a: f64, b: f64) -> f64 {
    if b == 0.0 {
        0.0
    } else {
        (a / b)
    }
}

impl ConfusionMatrix {
    pub fn create(n_classes: usize, class_names: Vec<String>) -> ConfusionMatrix {
        let name2index = class_names.iter()
            .enumerate().map(|(i, class)| (class.clone(), i as i64)).collect::<HashMap<_, _>>();

        ConfusionMatrix {
            matrix: DenseTensor::zeros(&[n_classes as i64, n_classes as i64]),
            n_classes,
            name2index,
            class_names
        }
    }

    pub fn precision_recall_fbeta(&self, class_name: &str, beta: f64) -> (f64, f64, f64) {
        let class_idx = self.name2index[class_name];
        let precision = _safe_div(self.matrix.at((class_idx, class_idx)).get_f64(), self.matrix.at(slice![;, class_idx]).sum().get_f64());
        let recall = _safe_div(self.matrix.at((class_idx, class_idx)).get_f64(), self.matrix.at(slice![class_idx, ;]).sum().get_f64());
        let fbeta = _safe_div((1.0 + beta.powi(2)) * precision * recall, (beta.powi(2)) * precision + recall);

        (precision, recall, fbeta)
    }

    pub fn all_precision_recall_fbeta(&self, beta: f64) -> Vec<(f64, f64, f64)> {
        let mut metrics = Vec::new();
        for class_name in &self.class_names {
            metrics.push(self.precision_recall_fbeta(class_name, beta));
        }
        metrics
    }

    pub fn prettify(&self) -> String {
        let mut str_matrix = vec![vec![String::from("")]];
        str_matrix[0].extend_from_slice(&self.class_names);
        str_matrix[0].push(String::from("supports"));

        let mut cols_width = str_matrix[0].iter().map(|s| s.len()).collect::<Vec<_>>();
        for i in 0..(self.n_classes as i64) {
            let mut row = vec![self.class_names[i as usize].to_owned()];
            row.extend(self.matrix.at(slice![i, ;])
                .to_1darray().iter()
                .map(|v| format!("{}", v)));
            row.push(format!("{}", self.matrix.at(i).sum().get_f64()));

            str_matrix.push(row);
        }

        let mut row = vec![String::from("")];
        row.extend((0..self.n_classes)
            .map(|i| format!("{}", self.matrix.at(slice![;, i as i64]).sum().get_f64())));
        row.push(format!("{}", self.matrix.sum().get_f64()));
        str_matrix.push(row);

        for row in str_matrix.iter() {
            for (i, cell) in row.iter().enumerate() {
                cols_width[i] = cmp::max(cell.len(), cols_width[i]);
            }
        }

        let mut buffer = Vec::new();
        let line_sep = "\n".to_owned() + &(0..cols_width.iter().map(|w| w + 5).sum()).map(|_| "-").collect::<String>() + "\n";
        
        for i in 0..str_matrix.len() {
            let str_row = cols_width.iter().zip(str_matrix[i].iter())
                .map(|(w, s)| format!(" {} ", center(s, *w)))
                .collect::<Vec<_>>().join(" | ");

            buffer.push(format!("  {}  ", str_row));
        }

        return buffer.join(&line_sep);
    }

    pub fn pretty_print(&self, legend: &str) -> String {
        let s = self.prettify();
        let mut outputs = vec!["".to_owned()];
        let fbeta = 1.0;

        outputs.push(center(legend, s.find("\n").unwrap()));
        outputs.push(s);
        
        // report precision, recall, fbeta
        for (i, precision_recall_fbeta) in self.all_precision_recall_fbeta(fbeta).iter().enumerate() {
            let (p, r, f) = precision_recall_fbeta;
            outputs.push(format!("class={} : precision={}, recall={}, f{}={}", self.class_names[i], p, r, fbeta, f));
        }

        outputs.join("\n")
    }
}

impl Add for ConfusionMatrix {
    type Output = ConfusionMatrix;

    fn add(self, rhs: ConfusionMatrix) -> ConfusionMatrix {
        let matrix = self.matrix + rhs.matrix;

        ConfusionMatrix {
            matrix,
            n_classes: rhs.n_classes,
            name2index: rhs.name2index,
            class_names: rhs.class_names
        }
    }
}

impl Add<ConfusionMatrix> for i32 {
    type Output = ConfusionMatrix;

    fn add(self, rhs: ConfusionMatrix) -> ConfusionMatrix {
        rhs
    }
}

pub fn get_confusion_matrix<T: TensorType, D: TBinaryVectorDomain<String, T>>(assignment: &FnvHashMap<usize, BinaryVectorValue<T>>, target_assignment: &FnvHashMap<usize, BinaryVectorValue<T>>, domain: &D) -> ConfusionMatrix {
    let matrix = ConfusionMatrix::create(domain.numel(), (0..domain.numel()).map(|i| domain.get_category(i).to_owned()).collect::<Vec<_>>());

    for (k, v) in target_assignment.iter() {
        let pred_v = &assignment[k];
        matrix.matrix.at((v.idx as i64, pred_v.idx as i64)).add_assign(1.0);
    }

    matrix
}

pub fn get_default_confusion_matrix<T: TensorType, D: TBinaryVectorDomain<String, T>>(domain: &D) -> ConfusionMatrix {
    ConfusionMatrix::create(domain.numel(), (0..domain.numel()).map(|i| domain.get_category(i).to_owned()).collect::<Vec<_>>())
}