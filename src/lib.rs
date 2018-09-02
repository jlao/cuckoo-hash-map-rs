extern crate rand;

use std::borrow::Borrow;
use std::hash::Hash;
use std::hash::{BuildHasher, Hasher};
use std::collections::hash_map::RandomState;
use std::fmt::Debug;

#[derive(Debug)]
pub struct CuckooHashMap<K, V>
    where
        K: Hash + Eq + Debug
{
    hash_builder1: RandomState,
    hash_builder2: RandomState,
    buckets: Vec<Bucket<K, V>>,
}

const SLOTS_PER_BUCKET: usize = 4;
const INITIAL_SIZE: usize = 128;

#[derive(Debug)]
struct Bucket<K: Debug, V> {
    slots: [Option<(K, V)>; SLOTS_PER_BUCKET],
}

impl<K, V> Bucket<K, V>
    where K: Debug + Eq
{
    fn new() -> Self {
        Bucket {
            slots: [None, None, None, None]
        }
    }

    fn get_slot(&mut self) -> &mut Option<(K, V)> {
        for i in 0..self.slots.len() {
            if self.slots[i].is_none() {
                return &mut self.slots[i];
            }
        }
        
        &mut self.slots[rand::random::<usize>() % 4]
    }

    pub fn get<Q: ?Sized>(&self, k: &Q) -> Option<&V>
        where
            K: Borrow<Q>,
            Q: Eq,
    {
        self.slots.iter()
            .filter_map(|slot| slot.as_ref())
            .find(|&(key, _)| key.borrow() == k)
            .map(|(_, val)| val)
    }
}

impl<K, V> CuckooHashMap<K, V>
    where
        K: Hash + Eq + Debug,
{
    pub fn new() -> CuckooHashMap<K, V> {
        CuckooHashMap {
            buckets: (0..INITIAL_SIZE).map(|_| Bucket::new()).collect(),
            hash_builder1: RandomState::new(),
            hash_builder2: RandomState::new(),
        }
    }

    pub fn insert(&mut self, k: K, v: V) -> Option<V> {
        let mut existing = (k, v);
        while let Some(new_existing) = self.insert_kicked(existing) {
            existing = new_existing;
        }

        None
    }

    // Inserts v at k. Returns kicked item or None.
    fn insert_kicked(&mut self, kv: (K, V)) -> Option<(K, V)> {
        let (k, v) = kv;
        {
            let h1 = self.hash1(&k);
            let bucket = &mut self.buckets[h1];
            let slot = bucket.get_slot();
            if slot.is_none() {
                *slot = Some((k, v));
                return None;
            }
        }

        let h2 = self.hash2(&k);
        let bucket = &mut self.buckets[h2];
        let slot = bucket.get_slot();
        let existing = slot.take();
        *slot = Some((k, v));
        existing
    }

    pub fn get<Q: ?Sized>(&self, k: &Q) -> Option<&V>
        where
            K: Borrow<Q>,
            Q: Hash + Eq,
    {
        let h1 = self.hash1(k);
        let bucket = &self.buckets[h1];
        let s = bucket.get(k);
        if s.is_some() {
            return s;
        }

        let h2 = self.hash2(k);
        let bucket = &self.buckets[h2];
        bucket.get(k)
    }

    fn hash1<Q: Hash + ?Sized>(&self, k: &Q) -> usize {
        let mut hasher = self.hash_builder1.build_hasher();
        k.hash(&mut hasher);
        hasher.finish() as usize % self.buckets.len()
    }

    fn hash2<Q: Hash + ?Sized>(&self, k: &Q) -> usize {
        let mut hasher = self.hash_builder2.build_hasher();
        k.hash(&mut hasher);
        hasher.finish() as usize % self.buckets.len()
    }
}

#[cfg(test)]
mod tests {
    use super::CuckooHashMap;

    #[test]
    fn basics() {
        let mut map = CuckooHashMap::new();
        map.insert(1, "foo");
        map.insert(2, "bar");

        assert_eq!(Some(&"foo"), map.get(&1));
    }
}
