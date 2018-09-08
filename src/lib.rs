extern crate rand;

use std::borrow::Borrow;
use std::hash::Hash;
use std::hash::{BuildHasher, Hasher};
use std::collections::hash_map::RandomState;
use std::ops::{Deref, Index};

#[derive(Debug)]
pub struct CuckooHashMap<K, V>
    where
        K: Hash + Eq
{
    state: RandomState,
    table: Table<K, V>,
}

#[derive(Debug)]
struct HashKey {
    hash: usize,
    partial: u8,
}

impl HashKey {
    fn new(hash: usize, partial: u8) -> HashKey {
        HashKey {
            hash,
            partial,
        }
    }

    fn index(&self, table_size: usize) -> usize {
        self.hash % table_size
    }

    fn alt_index(&self, table_size: usize) -> usize {
        let index = self.index(table_size);
        (index ^ (self.partial as usize).wrapping_mul(0xc6a4a7935bd1e995)) % table_size
    }
}

fn create_partial(hash: usize) -> u8 {
    let hash = (hash as u32) ^ ((hash >> 32) as u32);
    let hash = (hash as u16) ^ ((hash >> 16) as u16);
    let hash = (hash as u8) ^ ((hash >> 8) as u8);
    if hash > 0 { hash } else { 1 }
}

#[derive(Debug)]
struct Table<K, V>
    where
        K: Eq,
{
    buckets: Vec<Bucket<K, V>>,
}

impl<K, V> Table<K, V>
    where
        K: Eq,
{
    fn new() -> Table<K, V> {
        Table {
            buckets: (0..INITIAL_SIZE).map(|_| Bucket::new()).collect(),
        }
    }

    fn size(&self) -> usize {
        self.buckets.len()
    }
}

fn find_slot<K, V, M, F>(table: M, hashkey: &HashKey, is_match: F)
    -> Slot<M>
    where
        K: Eq,
        M: Deref<Target = Table<K, V>>,
        F: Fn(&K) -> bool,
{
    // Look in first bucket
    let b1_index = hashkey.index(table.size());
    println!("Looking in bucket {}", b1_index);
    let (b1_slot, b1_slot_status) =
        table.buckets[b1_index].find_slot(hashkey, &is_match);

    if b1_slot_status == SlotStatus::Match {
        return Slot::new(b1_index, b1_slot, SlotStatus::Match, table);
    }

    // Look in second bucket
    let b2_index = hashkey.alt_index(table.size());
    println!("Looking in bucket {}", b2_index);
    let (b2_slot, b2_slot_status) =
        table.buckets[b2_index].find_slot(hashkey, &is_match);

    match b2_slot_status {
        SlotStatus::Match => {
            Slot::new(b2_index, b2_slot, SlotStatus::Match, table)
        },
        SlotStatus::Open => {
            if b1_slot_status == SlotStatus::Open {
                // Prefer open slots from first bucket
                Slot::new(b1_index, b1_slot, SlotStatus::Open, table)
            } else {
                Slot::new(b2_index, b2_slot, SlotStatus::Open, table)
            }
        },
        SlotStatus::Occupied => {
            Slot::new(b2_index, b2_slot, SlotStatus::Occupied, table)
        }
    }
}

fn find_alt_slot<K, V, M, F>(table: M, hashkey: &HashKey, is_match: F)
    -> Slot<M>
    where
        K: Eq,
        M: Deref<Target = Table<K, V>>,
        F: Fn(&K) -> bool,
{
    let bucket = hashkey.alt_index(table.size());
    let (slot, slot_status) =
        table.buckets[bucket].find_slot(hashkey, &is_match);

    Slot::new(bucket, slot, slot_status, table)
}

const SLOTS_PER_BUCKET: usize = 4;
const INITIAL_SIZE: usize = 128;

struct Slot<M> {
    bucket: usize,
    slot: usize,
    slot_status: SlotStatus,
    table: M,
}

impl<M> Slot<M> {
    fn new(bucket: usize, slot: usize, slot_status: SlotStatus, table: M)
        -> Slot<M>
    {
        Slot { bucket, slot, slot_status, table }
    }
}

enum InsertResult<K, V> {
    Open,
    Match { prev: V },
    Displaced(Displaced<K, V>),
}

struct Displaced<K, V> {
    key: K,
    val: V,
    partial: PartialKey,
    bucket: usize,
}

impl<K, V> Displaced<K, V> {
    fn hashkey(&self) -> HashKey {
        HashKey::new(self.bucket, self.partial)
    }
}

impl<'m, K: 'm, V: 'm> Slot<&'m Table<K, V>>
    where
        K: Eq,
{
    fn raw_slot(&self) -> &'m Option<(K, V)> {
        &self.table.buckets[self.bucket].slots[self.slot]
    }
}

impl<'m, K: 'm, V: 'm> Slot<&'m mut Table<K, V>>
    where
        K: Eq,
{
    fn val(&self) -> &V {
        &self.table.buckets[self.bucket].slots[self.slot].as_ref().unwrap().1
    }

    fn val_mut(&mut self) -> &mut V {
        &mut self.table.buckets[self.bucket].slots[self.slot].as_mut().unwrap().1
    }

    fn into_val_mut(self) -> &'m mut V {
        &mut self.table.buckets[self.bucket].slots[self.slot].as_mut().unwrap().1
    }

    fn raw_slot_mut(&mut self) -> &mut RawSlot<K, V> {
        &mut self.table.buckets[self.bucket].slots[self.slot]
    }

    fn partial_mut(&mut self) -> &mut PartialKey {
        &mut self.table.buckets[self.bucket].partials[self.slot]
    }

    fn insert(&mut self, hashkey: &HashKey, key: K, val: V) -> InsertResult<K, V>
    {
        let prev_partial = *self.partial_mut();
        *self.partial_mut() = hashkey.partial;
        match self.slot_status {
            SlotStatus::Open => {
                *self.raw_slot_mut() = Some((key, val));
                InsertResult::Open
            },
            SlotStatus::Match => {
                let (key, prev) = self.raw_slot_mut().take().unwrap();
                *self.raw_slot_mut() = Some((key, val));
                InsertResult::Match { prev }
            },
            SlotStatus::Occupied => {
                let (pkey, pval) = self.raw_slot_mut().take().unwrap();
                *self.raw_slot_mut() = Some((key, val));
                InsertResult::Displaced(Displaced {
                    key: pkey,
                    val: pval,
                    partial: prev_partial,
                    bucket: self.bucket,
                })
            }
        }
    }

    fn remove(&mut self) -> Option<(K, V)> {
        debug_assert_eq!(SlotStatus::Match, self.slot_status);
        self.raw_slot_mut().take()
    }
}

fn insert_loop<K: Eq, V>(
    slot: &mut Slot<&mut Table<K, V>>,
    hashkey: &HashKey,
    key: K,
    val: V) -> Option<V>
{
    let mut insert_result = slot.insert(&hashkey, key, val);

    match insert_result {
        InsertResult::Open => return None,
        InsertResult::Match { prev } => return Some(prev),
        _ => {},
    }

    while let InsertResult::Displaced(displaced) = insert_result {
        let hashkey = displaced.hashkey();
        let mut slot = find_alt_slot(
            &mut *slot.table,
            &hashkey,
            |k| k == &displaced.key);
        insert_result = slot.insert(&hashkey, displaced.key, displaced.val);
    }

    None
}

#[derive(PartialEq, Eq, Debug)]
enum SlotStatus {
    Open,
    Match,
    Occupied,
}

type RawSlot<K, V> = Option<(K, V)>;
type PartialKey = u8;

#[derive(Debug)]
struct Bucket<K, V> {
    partials: [PartialKey; SLOTS_PER_BUCKET],
    slots: [RawSlot<K, V>; SLOTS_PER_BUCKET],
}

impl<K, V> Bucket<K, V>
    where K: Eq
{
    fn new() -> Self {
        Bucket {
            partials: [0; SLOTS_PER_BUCKET],
            slots: [None, None, None, None],
        }
    }

    fn find_slot<F>(&self, hashkey: &HashKey, is_match: &F) -> (usize, SlotStatus)
        where
            F: Fn(&K) -> bool
    {
        let mut open_slot = None;

        for i in 0..self.slots.len() {
            println!("Looking in slot {}", i);

            if self.partials[i] == 0 {
                open_slot = Some(i);
                println!("Partial is 0 in bucket {}", i);
                continue;
            }

            if self.partials[i] != hashkey.partial {
                println!("Partial is {} in bucket {}", hashkey.partial, i);
                continue;
            }

            match &self.slots[i] {
                None if open_slot.is_none() => {
                    open_slot = Some(i);
                    continue;
                },
                None => {},
                Some((k, _)) => {
                    if is_match(k) {
                        return (i, SlotStatus::Match);
                    }
                }
            }
        }

        if let Some(i) = open_slot {
            return (i, SlotStatus::Open);
        }

        (rand::random::<usize>() % SLOTS_PER_BUCKET, SlotStatus::Occupied)
    }
}

impl<K, V> CuckooHashMap<K, V>
    where
        K: Hash + Eq,
{
    pub fn new() -> CuckooHashMap<K, V> {
        CuckooHashMap {
            state: RandomState::new(),
            table: Table::new(),
        }
    }

    pub fn entry(&mut self, key: K) -> Entry<K, V> {
        let hashkey = self.hash(&key);
        let slot = find_slot(
            &mut self.table,
            &hashkey,
            |k| k == &key);
        
        match slot.slot_status {
            SlotStatus::Open | SlotStatus::Occupied => {
                Entry::Vacant(VacantEntry {
                    key,
                    hashkey,
                    slot,
                })
            },
            SlotStatus::Match => {
                Entry::Occupied(OccupiedEntry {
                    key,
                    slot,
                })
            }
        }
    }

    pub fn insert(&mut self, key: K, val: V) -> Option<V> {
        let hashkey = self.hash(&key);
        let mut slot = find_slot(&mut self.table, &hashkey, |k| k == &key);
        insert_loop(
            &mut slot,
            &hashkey,
            key,
            val)
    }

    pub fn get<'a, Q>(&'a self, key: &Q) -> Option<&'a V>
        where
            K: Borrow<Q>,
            Q: Hash + Eq + ?Sized,
    {
        let hashkey = self.hash(key);
        let slot = find_slot(&self.table, &hashkey, |k| k.borrow() == key);
        if slot.slot_status == SlotStatus::Match {
            slot.raw_slot().as_ref().map(|(_, v)| v)
        } else {
            None
        }
    }

    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
        where
            K: Borrow<Q>,
            Q: Hash + Eq,
    {
        let hashkey = self.hash(key);
        let mut slot = find_slot(&mut self.table, &hashkey, |k| k.borrow() == key);
        if slot.slot_status == SlotStatus::Match {
            slot.remove().map(|(_, v)| v)
        } else {
            None
        }
    }

    pub fn remove_entry<Q>(&mut self, key: &Q) -> Option<(K, V)>
        where
            K: Borrow<Q>,
            Q: Hash + Eq,
    {
        let hashkey = self.hash(key);
        let mut slot = find_slot(&mut self.table, &hashkey, |k| k.borrow() == key);
        if slot.slot_status == SlotStatus::Match {
            slot.remove()
        } else {
            None
        }
    }

    fn hash<Q: Hash + ?Sized>(&self, key: &Q) -> HashKey {
        let mut hasher = self.state.build_hasher();
        key.hash(&mut hasher);
        let hash = hasher.finish() as usize;
        let partial = create_partial(hash);
        HashKey {
            hash,
            partial,
        }
    }
}

pub enum Entry<'a, K, V>
    where
        K: 'a + Hash + Eq,
        V: 'a,
{
    Occupied(OccupiedEntry<'a, K, V>),
    Vacant(VacantEntry<'a, K, V>),
}

impl<'a, K, V> Entry<'a, K, V>
    where
        K: 'a + Hash + Eq,
        V: 'a,
{
    /// Ensures a value is in the entry by inserting the default if empty, and returns
    /// a mutable reference to the value in the entry.
    ///
    /// # Examples
    ///
    /// ```
    /// use cuckoo::CuckooHashMap;
    ///
    /// let mut map: CuckooHashMap<&str, u32> = CuckooHashMap::new();
    /// map.entry("poneyland").or_insert(12);
    ///
    /// assert_eq!(map["poneyland"], 12);
    ///
    /// *map.entry("poneyland").or_insert(12) += 10;
    /// assert_eq!(map["poneyland"], 22);
    /// ```
    pub fn or_insert(self, default: V) -> &'a mut V {
        match self {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.insert(default),
        }
    }

    pub fn or_insert_with<F: FnOnce() -> V>(self, default: F) -> &'a mut V {
        match self {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.insert(default()),
        }
    }

    pub fn key(&self) -> &K {
        match *self {
            Entry::Occupied(ref entry) => entry.key(),
            Entry::Vacant(ref entry) => entry.key(),
        }
    }

    pub fn and_modify<F>(self, f: F) -> Self
        where
            F: FnOnce(&mut V),
    {
        match self {
            Entry::Occupied(mut entry) => {
                f(entry.get_mut());
                Entry::Occupied(entry)
            },
            Entry::Vacant(entry) => Entry::Vacant(entry),
        }
    }
}

impl<'a, K: Hash + Eq, V: Default> Entry<'a, K, V> {
    pub fn or_default(self) -> &'a mut V {
        unimplemented!()
    }
}

pub struct OccupiedEntry<'a, K, V>
    where
        K: 'a + Eq,
        V: 'a,
{
    key: K,
    slot: Slot<&'a mut Table<K, V>>,
}

impl<'a, K, V> OccupiedEntry<'a, K, V>
    where
        K: 'a + Eq,
        V: 'a,
{
    pub fn key(&self) -> &K {
        &self.key
    }

    pub fn remove_entry(self) -> (K, V) {
        let mut slot = self.slot;
        slot.remove().unwrap()
    }

    pub fn get(&self) -> &V {
        self.slot.val()
    }

    pub fn get_mut(&mut self) -> &mut V {
        self.slot.val_mut()
    }

    pub fn into_mut(self) -> &'a mut V {
        self.slot.into_val_mut()
    }

    pub fn insert(&mut self, value: V) -> V {
        std::mem::replace(self.get_mut(), value)
    }

    pub fn remove(self) -> V {
        self.remove_entry().1
    }
}

pub struct VacantEntry<'a, K, V>
    where
        K: 'a + Hash + Eq,
        V: 'a,
{
    key: K,
    hashkey: HashKey,
    slot: Slot<&'a mut Table<K, V>>,
}

impl<'a, K, V> VacantEntry<'a, K, V>
    where
        K: 'a + Hash + Eq,
        V: 'a,
{
    pub fn key(&self) -> &K {
        &self.key
    }

    pub fn into_key(self) -> K {
        self.key
    }

    pub fn insert(self, value: V) -> &'a mut V {
        debug_assert_eq!(
            self.slot.slot_status,
            SlotStatus::Open,
            "Expected slot in VacantEntry to be Open");
        
        let mut slot = self.slot;
        insert_loop(
            &mut slot,
            &self.hashkey,
            self.key,
            value);
        slot.into_val_mut()
    }
}

impl<'a, K, Q: ?Sized, V> Index<&'a Q> for CuckooHashMap<K, V>
where
    K: Eq + Hash + Borrow<Q>,
    Q: Eq + Hash,
{
    type Output = V;

    fn index(&self, key: &Q) -> &V {
        self.get(key).expect("no entry found for key")
    }
}

#[cfg(test)]
mod internal_tests {
    use super::*;

    #[test]
    fn table_basics() {
        let mut table = Table::new();
        let hashkey = HashKey::new(7832, create_partial(7832));
        println!("hashkey = {:?}", hashkey);

        {
            let mut slot = find_slot(&mut table, &hashkey, |k| *k == 4);
            println!("Slot status = {:?}", slot.slot_status);
            slot.insert(&hashkey, 4, "hello".to_string());
        }

        let slot = find_slot(&table, &hashkey, |k| {
            println!("k = {:?}", *k);
            *k == 4
        });
        println!("Slot status = {:?}", slot.slot_status);
        assert_eq!(SlotStatus::Match, slot.slot_status);

        let v = slot.raw_slot();
        assert_eq!(&Some((4, "hello".to_string())), v);
    }

    #[test]
    fn hashkey_invertable() {
        let table_size = 128;
        let hashkey = HashKey::new(74839, create_partial(74839));
        let index = hashkey.index(table_size);
        let alt_index = hashkey.alt_index(table_size);

        let hashkey = HashKey::new(alt_index, hashkey.partial);
        assert_eq!(index, hashkey.alt_index(table_size));
    }
/*
*/
/*
    #[test]
    fn vacant_entry_key() {
        let mut map: CuckooHashMap<i32, String> = CuckooHashMap::new();
        let entry = map.entry(1);
        if let Entry::Vacant(vacant) = entry {
            assert_eq!(&1, vacant.key());
            assert_eq!(1, vacant.into_key());
        } else {
            panic!("Expected vacant entry")
        }
    }

    #[test]
    fn vacant_entry_insert() {
        let mut map = CuckooHashMap::new();
        let entry = map.entry(1);
        if let Entry::Vacant(vacant) = entry {
            assert_eq!(
                &"value".to_string(),
                vacant.insert("value".to_string()));
        } else {
            panic!("Expected vacant entry")
        }
    }
*/
}