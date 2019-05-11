use assembling::models::variable::*;
use assembling::models::annotator::Annotator;
use gmtk::prelude::*;

use super::mrr::*;
use std::path::Path;
use bincode;
use std::fs::File;
use std::io::prelude::*;
use assembling::features::*;
use assembling::models::templates::*;
use settings::conf_mrf::TemplatesConf;

#[derive(Serialize, Deserialize)]
struct MRRSerializeMeta {
    params_size: Vec<usize>,
    tf_domain_size: usize,
    dup_domain_size: usize,
    pk_domain_size: usize,
    cooccurr_domain_size: usize,
    templates_conf_size: usize
}

impl MRRSerializeMeta {
    pub fn get_total_size(&self) -> usize {
        self.tf_domain_size + self.dup_domain_size +
            self.pk_domain_size +
            self.cooccurr_domain_size +
            self.templates_conf_size +
            self.params_size.iter().sum::<usize>()
    }
}


impl<'a> MRRModel<'a> {
    pub fn deserialize(mut annotator: Annotator<'a>, model_file: &Path) -> MRRModel<'a> {
        let mut file = File::open(model_file).unwrap();
        let mut buf = Vec::new();
        file.read_to_end(&mut buf).unwrap();

        // reload the annotator with its previous step
        // TODO: annotator reload node_prob, which may be change in previous run, should
        // the better way to do this
        annotator.reload();

        // deserialize model
        let meta_data_size: u64 = bincode::deserialize(&buf[..8]).unwrap();
        let meta_data: MRRSerializeMeta = bincode::deserialize(&buf[8..(meta_data_size as usize) + 8]).unwrap();
        let mut pivot = meta_data_size as usize + 8;
        let tf_domain: TFDomain = bincode::deserialize(&buf[pivot..pivot + meta_data.tf_domain_size]).unwrap();
        pivot += meta_data.tf_domain_size;
        let dup_pairwise_domain: DuplicationPairwiseDomain = bincode::deserialize(&buf[pivot..pivot + meta_data.dup_domain_size]).unwrap();
        pivot += meta_data.dup_domain_size;
        let pk_pairwise_domain: PkPairwiseDomain = bincode::deserialize(&buf[pivot..pivot + meta_data.pk_domain_size]).unwrap();
        pivot += meta_data.pk_domain_size;
        let cooccur_domain: CooccurrenceDomain = bincode::deserialize(&buf[pivot..pivot + meta_data.cooccurr_domain_size]).unwrap();
        pivot += meta_data.cooccurr_domain_size;
        let templates_conf: TemplatesConf = bincode::deserialize(&buf[pivot..pivot + meta_data.templates_conf_size]).unwrap();
        pivot += meta_data.templates_conf_size;

        let model = MRRModel::create_loglinear_model(&tf_domain, &dup_pairwise_domain, &pk_pairwise_domain, &cooccur_domain, &annotator, &templates_conf);
        {
            let parameters = model.get_parameters();
            assert_eq!(parameters.len(), meta_data.params_size.len());
            for (i, param) in parameters.iter().enumerate() {
                let weight: Weights = bincode::deserialize(&buf[pivot..pivot + meta_data.params_size[i]]).unwrap();
                pivot += meta_data.params_size[i];
                param.copy_(&weight);
            }

            let total_size = 8 + meta_data_size as usize + meta_data.get_total_size();
            assert_eq!(pivot, total_size);
        }

        // re-create it
        MRRModel::new(annotator.dataset, annotator, templates_conf, model, tf_domain, dup_pairwise_domain, pk_pairwise_domain, cooccur_domain)
    }

    pub fn serialize(&self, model_file: &Path) {
        // calculate size of the serialization & metadata
        let tf_domain_size = bincode::serialized_size(&self.tf_domain).unwrap() as usize;
        let dup_domain_size = bincode::serialized_size(&self.dup_pairwise_domain).unwrap() as usize;
        let pk_domain_size = bincode::serialized_size(&self.pk_pairwise_domain).unwrap() as usize;
        let cooccurr_domain_size = bincode::serialized_size(&self.cooccur_domain).unwrap() as usize;
        let templates_conf_size = bincode::serialized_size(&self.templates_conf).unwrap() as usize;

        let mut meta_data = MRRSerializeMeta {
            params_size: Vec::new(),
            tf_domain_size,
            dup_domain_size,
            pk_domain_size,
            cooccurr_domain_size,
            templates_conf_size
        };

        for param in self.model.get_parameters() {
            meta_data.params_size.push(bincode::serialized_size(param).unwrap() as usize);
        }
        let meta_data_size: u64 = bincode::serialized_size(&meta_data).unwrap();

        // start serializing, first size of meta data & metadata
        // then parameters
        let total_size = 8 + meta_data_size as usize + meta_data.get_total_size();

        let mut bytes = Vec::with_capacity(total_size);
        bytes.append(&mut bincode::serialize(&meta_data_size).unwrap());
        bytes.append(&mut bincode::serialize(&meta_data).unwrap());
        bytes.append(&mut bincode::serialize(&self.tf_domain).unwrap());
        bytes.append(&mut bincode::serialize(&self.dup_pairwise_domain).unwrap());
        bytes.append(&mut bincode::serialize(&self.pk_pairwise_domain).unwrap());
        bytes.append(&mut bincode::serialize(&self.cooccur_domain).unwrap());
        bytes.append(&mut bincode::serialize(&self.templates_conf).unwrap());
        for param in self.model.get_parameters() {
            bytes.append(&mut bincode::serialize(&param).unwrap());
        }

        assert_eq!(bytes.len(), total_size);
        let mut file = File::create(model_file).unwrap();
        file.write_all(&bytes).unwrap();
    }
}