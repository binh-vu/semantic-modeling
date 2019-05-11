use time::precise_time_ns;
use std::collections::HashMap;
use std::collections::HashSet;
use std::mem::ManuallyDrop;
use std::ptr;
use std::hash::Hash;

pub struct Timer {
    start: u64,
    current_time: u64
}

impl Timer {
    pub fn start() -> Timer {
        let time = precise_time_ns();
        Timer {
            start: time,
            current_time: time
        }
    }

    pub fn reset(&mut self) {
        self.start = precise_time_ns();
        self.current_time = self.start;
    }

    pub fn lap_and_report(&mut self, message: &str) {
        let current_time = precise_time_ns();
        println!("[TIMER] --- Announcement: {}\n    + Time since last message: {:.5}s\n    + Time since started: {:.5}s", 
            message, (current_time - self.current_time) as f64 * 1e-9, (current_time - self.start) as f64 * 1e-9);
        self.current_time = current_time;
    }

    pub fn lap(&mut self) -> (f64, f64) {
        let current_time = precise_time_ns();
        let since_last_lap = (current_time - self.current_time) as f64 * 1e-9;
        let since_started = (current_time - self.start) as f64 * 1e-9;
        self.current_time = current_time;

        (since_last_lap, since_started)
    }
}

// See: https://users.rust-lang.org/t/hashmap-with-tuple-keys/12711
pub fn dict_get<'a, K1: Eq + Hash, K2: Eq + Hash, V>(map: &'a HashMap<(K1, K2), V>, a: &K1, b: &K2) -> Option<&'a V> {
    unsafe {
        // The 24-byte string headers of `a` and `b` may not be adjacent in
        // memory. Copy them (just the headers) so that they are adjacent. This
        // makes a `(String, String)` backed by the same data as `a` and `b`.
        let k = (ptr::read(a), ptr::read(b));

        // Make sure not to drop the strings, even if `get` panics. The caller
        // or whoever owns `a` and `b` will drop them.
        let k = ManuallyDrop::new(k);

        // Deref `k` to get `&(String, String)` and perform lookup.
        map.get(&k)
    }
}

pub fn dict_has<'a, K1: Eq + Hash, K2: Eq + Hash, V>(map: &'a HashMap<(K1, K2), V>, a: &K1, b: &K2) -> bool {
    unsafe {
        let k = (ptr::read(a), ptr::read(b));
        let k = ManuallyDrop::new(k);
        map.contains_key(&k)
    }
}

pub fn set_has<'a, K1: Eq + Hash, K2: Eq + Hash>(set: &'a HashSet<(K1, K2)>, a: &K1, b: &K2) -> bool {
    unsafe {
        let k = (ptr::read(a), ptr::read(b));
        let k = ManuallyDrop::new(k);
        set.contains(&k)
    }
}