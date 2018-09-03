extern crate rand;

use std::borrow::Borrow;
use std::hash::Hash;
use std::hash::{BuildHasher, Hasher};
use std::collections::hash_map::RandomState;
use std::fmt::Debug;
use std::mem;

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

    fn get_slot(&mut self, k: &K) -> &mut Option<(K, V)> {
        let mut open_slot = None;

        for i in 0..self.slots.len() {
            if self.slots[i].is_some() {
                if self.slots[i].as_ref().unwrap().0 == *k {
                    return &mut self.slots[i];
                }
            } else {
                open_slot = Some(i);
            }
        }

        if let Some(i) = open_slot {
            return &mut self.slots[i];
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

pub enum Entry<K: Hash + Eq + Debug, V> {
    Occupied(OccupiedEntry<K, V>),
    Vacant(VacantEntry<K, V>),
}

pub struct OccupiedEntry<K: Hash + Eq + Debug, V> {
    map: CuckooHashMap<K, V>
}

pub struct VacantEntry<K: Hash + Eq + Debug, V> {
    map: CuckooHashMap<K, V>
}

enum InsertResult<K, V> {
    None,
    Replaced(V),
    Kicked(K, V),
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
        let mut insert = (k, v);
        loop {
            match self.insert_kicked(insert) {
                InsertResult::None => return None,
                InsertResult::Replaced(v) => return Some(v),
                InsertResult::Kicked(k, v) => {
                    insert = (k, v);
                }
            }
        }
    }

    // Inserts v at k. Returns kicked item or None.
    fn insert_kicked(&mut self, kv: (K, V)) -> InsertResult<K, V> {
        let (k, v) = kv;
        {
            let h1 = self.hash1(&k);
            let bucket = &mut self.buckets[h1];
            let slot = bucket.get_slot(&k);

            match slot {
                None => {
                    *slot = Some((k, v));
                    println!("Inserted into {}", h1);
                    return InsertResult::None;
                },
                Some(existing) => {
                    if existing.0 == k {
                        let prev = mem::replace(&mut existing.1, v);
                        return InsertResult::Replaced(prev);
                    }
                }
            }
        }

        let h2 = self.hash2(&k);
        let bucket = &mut self.buckets[h2];
        let slot = bucket.get_slot(&k);

        match slot {
            None => {
                *slot = Some((k, v));
                return InsertResult::None;
            },
            Some(existing) => {
                if existing.0 == k {
                    let prev = mem::replace(&mut existing.1, v);
                    InsertResult::Replaced(prev)
                } else {
                    let (k, v) = mem::replace(existing, (k, v));
                    InsertResult::Kicked(k, v)
                }
            }
        }
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
        map.insert(1, "foo".to_string());
        map.insert(2, "bar".to_string());

        assert_eq!(Some(&"foo".to_string()), map.get(&1));
    }

    #[test]
    fn insert() {
        let mut map = CuckooHashMap::new();
        map.insert(1, "foo".to_string());
        assert_eq!(Some(&"foo".to_string()), map.get(&1));

        let prev = map.insert(1, "bar".to_string());
        assert_eq!(Some("foo".to_string()), prev);
        assert_eq!(Some(&"bar".to_string()), map.get(&1));
    }
}
