extern crate cuckoo;
extern crate rand;

use cuckoo::*;

fn main() {
    let mut map = CuckooHashMap::with_capacity(4);
    let mut capacity = map.capacity();
    let mut num_resizes = 0;
    let mut i = 0;

    println!("#resize     #items   capacity   load_factor");
    while num_resizes < 20 {
        map.insert(i, i);
        i += 1;

        if map.capacity() != capacity {
            num_resizes += 1;
            let prev_len = map.len() - 1;
            let lf = prev_len as f64 / capacity as f64;
            println!(
                "{:7} {:10} {:10} {:13.2}",
                num_resizes,
                prev_len,
                capacity,
                lf * 100.0);

            capacity = map.capacity();
        }
    }
}