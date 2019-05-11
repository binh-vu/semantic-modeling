#![allow(dead_code)]
#![allow(unused_imports)]
extern crate gmtk;
extern crate libc;
extern crate bincode;
extern crate time;
extern crate fnv;

use std::thread;
use gmtk::tensors::*;
use std::time::{Duration, Instant};
use std::collections::HashSet;
use time::precise_time_ns;
use gmtk::graph_models::*;
use fnv::FnvHashMap;
use gmtk::optimization::accumulators::ValueAccumulator;
use gmtk::optimization::accumulators::Tensor1AccumulatorDict;
use gmtk::optimization::example::NLLExample;

use gmtk::graph_models::*;
use gmtk::tensors::*;
use std::collections::HashMap;


fn main() {
    let v: DenseTensor = DenseTensor::create_randn(&[10, 10]);
    let handle = thread::spawn(move || {
       println!("Result: {:?}", v.sum());
    });
    handle.join().unwrap();

    println!(">>>> FINISH!!!");
}
