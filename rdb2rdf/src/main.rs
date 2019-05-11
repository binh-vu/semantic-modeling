#![allow(unused_imports)]
#![allow(dead_code)]
#![allow(unused_variables)]
extern crate rdb2rdf;
extern crate serde_json;

use rdb2rdf::models::relational_table::RelationalTable;
use rdb2rdf::r2rml::r2rml::R2RML;
use rdb2rdf::utils::data_reader::load_json;
use rdb2rdf::utils::data_reader::load_xml;
use std::path::Path;

fn main() {
//    let rows = vec![1, 2, 3];
    let path = Path::new("/Users/rook/workspace/DataIntegration/source-modeling/data/museum-edm/sources/s03-ima-artists.xml");
    let fpath2model = Path::new("/Users/rook/workspace/DataIntegration/source-modeling/data/museum-edm/models-y2rml/s03-ima-artists.xml");

    let tbl = RelationalTable::load_from_file(path);
    println!("Schema: {:?}", tbl.schema);
    let serialized = serde_json::to_string_pretty(&tbl.schema).unwrap();
    println!("{}", serialized);

//    let mut r2rml = R2RML::load_from_file(fpath2model);

//    for row in &tbl.rows {
//        println!("{:?}", row);
//        println!("{:?}", tbl.schema.normalize(Some(row)));
//    }
//     k, v in
//    println!("{}", x);
}