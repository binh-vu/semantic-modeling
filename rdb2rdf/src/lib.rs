#![allow(unused_imports)]
#![allow(dead_code)]
#![allow(unused_variables)]
extern crate xml;
extern crate yaml_rust;

extern crate serde;
#[macro_use]
extern crate serde_json;
#[macro_use]
extern crate serde_derive;

extern crate prettytable;
extern crate algorithm;

pub mod models;
pub mod utils;
pub mod r2rml;
pub mod ontology;
pub mod prelude;