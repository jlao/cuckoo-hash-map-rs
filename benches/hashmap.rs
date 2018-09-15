#[macro_use]
extern crate criterion;
extern crate cuckoo;

use criterion::{Criterion, Fun};
use cuckoo::CuckooHashMap;
use std::collections::HashMap;

fn cuckoo_hashmap_insertion(c: &mut Criterion) {
    let mut cuckoo_map = CuckooHashMap::new();
    let mut cuckoo_val = 0;
    let mut hash_map = HashMap::new();
    let mut hash_val = 0;

    let cuckoo_bench = Fun::new("CuckooHashmap", move |b, _i| {
        b.iter(|| {
            cuckoo_val += 1;
            cuckoo_map.insert(cuckoo_val, cuckoo_val)
        });
    });

    let hashmap_bench = Fun::new("HashMap", move |b, _i| {
        b.iter(|| {
            hash_val += 1;
            hash_map.insert(hash_val, hash_val)
        });
    });

    let functions = vec!(cuckoo_bench, hashmap_bench);
    c.bench_functions("insert", functions, 10);
}

criterion_group!(benches, cuckoo_hashmap_insertion);
criterion_main!(benches);