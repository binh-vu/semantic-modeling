use serde_json;
use serde_json::Value;
use std::fs::File;
use std::io::BufReader;
use std::io::prelude::*;
use super::ml2json::xml2json;
use xml::EventReader;
use xml::reader::XmlEvent;
use std::path::Path;
use yaml_rust::YamlLoader;
use utils::ml2json::yml2json;


/// Load JSON data
pub fn load_json(fpath: &Path) -> Value {
    let mut f = File::open(fpath).expect("file not found");
    let mut content = String::new();
    f.read_to_string(&mut content).expect("something wents wrong reading the file");

    let data: Value = serde_json::from_str(&content).unwrap();
    return data;
}

/// Load XML data as SerdeJSON format
pub fn load_xml(fpath: &Path) -> Value {
    let file = File::open(fpath).expect("file not found");
    let file = BufReader::new(file);

    let parser = EventReader::new(file);
    return xml2json(parser);
}

pub fn load_yml(fpath: &Path) -> Value {
    let mut f = File::open(fpath).expect("file not found");
    let mut content = String::new();
    f.read_to_string(&mut content).expect("something wents wrong reading the file");

    let docs = YamlLoader::load_from_str(&content).unwrap();

    return yml2json(docs);
}

/// Load CSV data as SerdeJSON format
pub fn load_csv(fpath: &str, header: bool) {
    let file = File::open(fpath).expect("file not found");
}