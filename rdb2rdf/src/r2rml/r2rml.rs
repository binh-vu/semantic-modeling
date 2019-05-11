use std::path::Path;
use std::fs::File;
use std::io::BufReader;
use serde::Deserialize;
use serde::Serialize;
use std::process::Command;

pub struct R2RML {
    version: String,
    commands: Vec<Command>,
}

impl R2RML {

//    pub fn load_from_file(fpath: &Path) -> R2RML {
//        if fpath.extension().unwrap() == "yml" {
//            let reader = BufReader::new(File::open(fpath).expect("file not found"));
//            return serde_yaml::from_reader(reader).unwrap();
//        }
//
//        panic!("Not support extension: {:?}", fpath.extension().unwrap());
//    }
}