use rdb2rdf::prelude::SemanticModel;
use std::slice::Iter;
use std::collections::HashMap;

use serde_json::{ Value, from_value };
use serde::Deserialize;
use serde::Deserializer;

#[derive(Clone, Debug)]
pub enum Cardinality {
    OneToN,
    NToOne,
    OneToOne,
    Uncomparable
}

impl Cardinality {
    pub fn from_str(s: &str) -> Cardinality {
        match s {
            "NULL" => Cardinality::Uncomparable,
            "1-TO-1" => Cardinality::OneToOne,
            "1-TO-N" => Cardinality::OneToN,
            "N-TO-1" => Cardinality::NToOne,
            _ => {
                println!("s = {}", s);
                panic!("Invalid cardinality")
            }
        }
    }
    pub fn as_str(&self) -> &'static str {
        match *self {
            Cardinality::OneToN => "1-TO-N",
            Cardinality::NToOne => "N-TO-1",
            Cardinality::OneToOne => "1-TO-1",
            Cardinality::Uncomparable => "NULL"
        }
    }

    pub fn iterator() -> Iter<'static, Cardinality> {
        static CARDINALITIES: [Cardinality; 4] = [Cardinality::OneToOne, Cardinality::NToOne, Cardinality::OneToOne, Cardinality::Uncomparable];
        CARDINALITIES.into_iter()
    }
}

#[derive(Clone, Debug)]
pub enum Cell {
    Number(f32),
    Str(Cardinality)
}

#[derive(Clone)]
struct CardinalityMatrix {
    columns: Vec<String>,
    col2idx: HashMap<String, usize>,
    matrix: Vec<Vec<Cell>>
}

impl CardinalityMatrix {
    pub fn new(columns: Vec<String>, matrix: Vec<Vec<Cell>>) -> CardinalityMatrix {
        let col2idx: HashMap<String, usize> = columns.iter().enumerate()
            .map(|(i, c)| (c.clone(), i)).collect();

        CardinalityMatrix { columns, col2idx, matrix }
    }

    #[inline]
    pub fn get_cardinality(&self, given_col: &str, col: &str) -> &Cell {
        &self.matrix[self.col2idx[given_col]][self.col2idx[col]]
    }
}

#[derive(Clone)]
pub struct CardinalityFeatures {
    sm_cardinalities: HashMap<String, CardinalityMatrix>,
    sm_idx2id: Vec<String>
}

impl CardinalityFeatures {
    pub fn set_sm_index(&mut self, sms: &[SemanticModel]) {
        for sm in sms.iter() {
            self.sm_idx2id.push(sm.id.clone());
        }
        debug_assert_eq!(self.sm_idx2id.len(), self.sm_cardinalities.len());
    }

    pub fn get_cardinality(&self, sm_idx: usize, given_col: &str, col: &str) -> Cardinality {
        let cell = self.sm_cardinalities[&self.sm_idx2id[sm_idx]].get_cardinality(given_col, col);
        match cell {
            Cell::Number(ref val) => {
                if *val > 1.05 {
                    Cardinality::OneToN
                } else {
                    Cardinality::OneToOne
                }
            },
            Cell::Str(car) => car.clone()
        }
    }
}

impl Default for CardinalityFeatures {
    fn default() -> CardinalityFeatures {
        CardinalityFeatures {
            sm_cardinalities: HashMap::new(),
            sm_idx2id: Default::default()
        }
    }
}

impl<'de> Deserialize<'de> for CardinalityFeatures {
    fn deserialize<D>(deserializer: D) -> Result<CardinalityFeatures, D::Error>
        where D: Deserializer<'de> {
        let result = Value::deserialize(deserializer).unwrap();
        if let Value::Object(m) = result {
        let mut fmatrix: HashMap<String, CardinalityMatrix> = Default::default();
            for (sm_id, mut v) in m.into_iter() {
                let matrix: Vec<Vec<Cell>> = v["matrix"].as_array().unwrap().iter()
                    .map(|row| {
                        row.as_array().unwrap().iter().map(|cell| {
                            match cell {
                                &Value::Number(ref x) => Cell::Number(x.as_f64().unwrap() as f32),
                                &Value::String(ref s) => Cell::Str(Cardinality::from_str(&s)),
                                _ => {
                                    panic!("Must be a number or string")
                                }
                            }
                        }).collect()
                    })
                    .collect();

                let cols: Vec<String> = from_value(v["columns"].take()).unwrap();
                fmatrix.insert(sm_id, CardinalityMatrix::new(cols, matrix));
            }

            Ok(CardinalityFeatures {
                sm_idx2id: Vec::with_capacity(fmatrix.len()),
                sm_cardinalities: fmatrix,
            })
        } else {
            panic!("Expect an object");
        }
    }
}
