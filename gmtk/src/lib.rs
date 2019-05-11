extern crate libc;
extern crate num_traits;
extern crate serde;
#[macro_use]
extern crate serde_derive;
extern crate bincode;
extern crate uuid;
extern crate fnv;
extern crate rand;
extern crate rayon;
extern crate algorithm;

#[macro_use]
pub mod tensors;
pub mod graph_models;
pub mod optimization;
pub mod utils;
pub mod prelude;
pub mod eval;