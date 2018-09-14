extern crate rand;

use std::borrow::Borrow;
use std::collections::hash_map::RandomState;
use std::fmt;
use std::hash::Hash;
use std::hash::{BuildHasher, Hasher};
use std::iter;
use std::ops::{Deref, Index};
use std::slice;

const MAX_DISPLACEMENTS: u8 = 100;

#[derive(Debug)]
pub struct CuckooHashMap<K, V, S = RandomState>
where
    K: Eq + Hash,
    S: BuildHasher,
{
    state: S,
    table: Table<K, V>,
}

#[derive(Debug)]
struct HashKey {
    hash: usize,
    partial: u8,
}

impl HashKey {
    fn new<Q: Hash + ?Sized, S: BuildHasher>(key: &Q, state: &S) -> HashKey {
        let hash = hash(key, state);
        Self::from_hash(hash)
    }

    fn from_hash(hash: usize) -> HashKey {
        HashKey {
            hash,
            partial: Self::create_partial(hash),
        }
    }

    fn with_partial(hash: usize, partial: u8) -> HashKey {
        HashKey { hash, partial }
    }

    fn index(&self, table_size: usize) -> usize {
        self.hash % table_size
    }

    fn alt_index(&self, table_size: usize) -> usize {
        let index = self.index(table_size);
        (index ^ (self.partial as usize).wrapping_mul(0xc6a4a7935bd1e995)) % table_size
    }

    fn create_partial(hash: usize) -> u8 {
        let hash = (hash as u32) ^ ((hash >> 32) as u32);
        let hash = (hash as u16) ^ ((hash >> 16) as u16);
        let hash = (hash as u8) ^ ((hash >> 8) as u8);
        if hash > 0 {
            hash
        } else {
            1
        }
    }
}

#[derive(Debug)]
struct Table<K, V>
where
    K: Eq + Hash,
{
    size: usize,
    buckets: Vec<Bucket<K, V>>,
}

impl<K, V> Table<K, V>
where
    K: Eq + Hash,
{
    fn new(num_buckets: usize) -> Table<K, V> {
        Table {
            size: 0,
            buckets: (0..num_buckets).map(|_| Bucket::new()).collect(),
        }
    }

    fn size(&self) -> usize {
        self.buckets.len()
    }

    fn iter<'a>(&'a self) -> TableIter<'a, K, V> {
        TableIter {
            bucket: 0,
            slot: 0,
            table: self,
        }
    }

    fn iter_mut(&mut self) -> TableIterMut<K, V> {
        TableIterMut {
            buckets: self.buckets.iter_mut().fuse(),
            slots: None,
        }
    }

    fn into_iter(self) -> TableIntoIter<K, V> {
        TableIntoIter {
            bucket: 0,
            slot: 0,
            table: self,
        }
    }

    fn drain(&mut self) -> TableDrain<K, V> {
        self.size = 0;
        TableDrain {
            bucket: 0,
            slot: 0,
            table: self,
        }
    }

    fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&K, &mut V) -> bool
    {
        for bucket in 0..self.buckets.len() {
            for slot in 0..SLOTS_PER_BUCKET {
                let partial = self.buckets[bucket].partials[slot];
                if partial == 0 {
                    continue;
                }

                let should_remove = {
                    let (k, v) = self.buckets[bucket].slots[slot].as_mut().unwrap();
                    !f(k, v)
                };

                if should_remove {
                    self.size -= 1;
                    self.buckets[bucket].partials[slot] = 0;
                    self.buckets[bucket].slots[slot].take();
                }
            }
        }
    }

    fn insert<S>(&mut self, state: &S, hashkey: &HashKey, key: K, val: V) -> Option<V>
    where
        S: BuildHasher,
    {
        let mut slot = find_slot(&mut *self, hashkey, |k| k == &key);
        slot.insert(state, hashkey, key, val)
    }

}

fn hash<Q: Hash + ?Sized, S: BuildHasher>(key: &Q, state: &S) -> usize {
    let mut hasher = state.build_hasher();
    key.hash(&mut hasher);
    hasher.finish() as usize
}

struct TableIter<'a, K: 'a + Eq + Hash, V: 'a> {
    bucket: usize,
    slot: usize,
    table: &'a Table<K, V>
}

impl<'a, K: 'a + Eq + Hash, V: 'a> Iterator for TableIter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        while self.bucket < self.table.size() {
            while self.slot < SLOTS_PER_BUCKET {
                if self.table.buckets[self.bucket].partials[self.slot] == 0 {
                    self.slot += 1;
                    continue;
                }

                let slot = &self.table.buckets[self.bucket].slots[self.slot];
                self.slot += 1;
                let (k, v) = slot.as_ref().unwrap();
                return Some((k, v));
            }

            self.bucket += 1;
            self.slot = 0;
        }

        None
    }
}

struct TableIntoIter<K: Eq + Hash, V> {
    bucket: usize,
    slot: usize,
    table: Table<K, V>
}

impl<K: Eq + Hash, V> Iterator for TableIntoIter<K, V> {
    type Item = (K, V);

    fn next(&mut self) -> Option<Self::Item> {
        while self.bucket < self.table.size() {
            while self.slot < SLOTS_PER_BUCKET {
                if self.table.buckets[self.bucket].partials[self.slot] == 0 {
                    self.slot += 1;
                    continue;
                }

                let item = self.table.buckets[self.bucket].slots[self.slot].take();
                self.slot += 1;
                return item;
            }

            self.bucket += 1;
            self.slot = 0;
        }

        None
    }
}

struct TableIterMut<'a, K: 'a + Eq, V: 'a> {
    buckets: iter::Fuse<slice::IterMut<'a, Bucket<K, V>>>,
    slots: Option<iter::Fuse<slice::IterMut<'a, RawSlot<K, V>>>>,
}

impl<'a, K: 'a + Eq, V: 'a> Iterator for TableIterMut<'a, K, V> {
    type Item = (&'a K, &'a mut V);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(ref mut slots) = self.slots {
                while let Some(raw_slot) = slots.next() {
                    if let Some((k, v)) = raw_slot {
                        return Some((k, v));
                    }
                }
            }

            if let Some(bucket) = self.buckets.next() {
                self.slots = Some(bucket.slots.iter_mut().fuse());
            } else {
                self.slots = None;
                return None;
            }
        }
    }
}

struct TableDrain<'a, K: 'a + Eq + Hash, V: 'a> {
    bucket: usize,
    slot: usize,
    table: &'a mut Table<K, V>
}

impl<'a, K: 'a + Eq + Hash, V: 'a> Iterator for TableDrain<'a, K, V> {
    type Item = (PartialKey, K, V);

    fn next(&mut self) -> Option<Self::Item> {
        while self.bucket < self.table.size() {
            while self.slot < SLOTS_PER_BUCKET {
                let bucket = &mut self.table.buckets[self.bucket];
                let partial = &mut bucket.partials[self.slot];

                if *partial == 0 {
                    self.slot += 1;
                    continue;
                }

                let return_partial = *partial;
                *partial = 0;
                let (k, v) = bucket.slots[self.slot].take().unwrap();
                self.slot += 1;
                return Some((return_partial, k, v));
            }

            self.bucket += 1;
            self.slot = 0;
        }

        None
    }
}

// The TableDrain iterator lazily takes values out of the table. If you do not
// exhaust it, there will be left over items in the table. Thus, when the
// iterator is dropped, we call drop on any remaining items that were not
// iterated over to prevent memory leaks.
impl<'a, K: 'a + Eq + Hash, V: 'a> Drop for TableDrain<'a, K, V> {
    fn drop(&mut self) {
        self.for_each(drop);
    }
}

fn find_slot<K, V, M, F>(table: M, hashkey: &HashKey, is_match: F) -> Slot<M>
where
    K: Eq + Hash,
    M: Deref<Target = Table<K, V>>,
    F: Fn(&K) -> bool,
{
    // Look in first bucket
    let b1_index = hashkey.index(table.size());
    let (b1_slot, b1_slot_status) = table.buckets[b1_index].find_slot(hashkey, &is_match);

    if b1_slot_status == SlotStatus::Match {
        return Slot::Match(MatchSlot {
            bucket: b1_index,
            slot: b1_slot,
            table,
        });
    }

    // Look in second bucket
    let b2_index = hashkey.alt_index(table.size());
    let (b2_slot, b2_slot_status) = table.buckets[b2_index].find_slot(hashkey, &is_match);

    match b2_slot_status {
        SlotStatus::Match => {
            return Slot::Match(MatchSlot {
                bucket: b2_index,
                slot: b2_slot,
                table,
            })
        }
        SlotStatus::Open => {
            if b1_slot_status == SlotStatus::Open {
                // Prefer open slots from first bucket
                Slot::Vacant(VacantSlot {
                    bucket: b1_index,
                    slot: b1_slot,
                    table,
                })
            } else {
                Slot::Vacant(VacantSlot {
                    bucket: b2_index,
                    slot: b2_slot,
                    table,
                })
            }
        }
        SlotStatus::Occupied => Slot::Vacant(VacantSlot {
            bucket: b2_index,
            slot: b2_slot,
            table,
        }),
    }
}

fn find_alt_slot<K, V, M, F>(table: M, hashkey: &HashKey, is_match: F) -> Slot<M>
where
    K: Eq + Hash,
    M: Deref<Target = Table<K, V>>,
    F: Fn(&K) -> bool,
{
    let bucket = hashkey.alt_index(table.size());
    let (slot, slot_status) = table.buckets[bucket].find_slot(hashkey, &is_match);

    match slot_status {
        SlotStatus::Match => Slot::Match(MatchSlot {
            bucket,
            slot,
            table,
        }),
        _ => Slot::Vacant(VacantSlot {
            bucket,
            slot,
            table,
        }),
    }
}

const SLOTS_PER_BUCKET: usize = 4;
const INITIAL_SIZE: usize = 128;

enum Slot<M> {
    Vacant(VacantSlot<M>),
    Match(MatchSlot<M>),
}

struct MatchSlot<M> {
    bucket: usize,
    slot: usize,
    table: M,
}

impl<'m, K: 'm + Eq + Hash, V: 'm> MatchSlot<&'m Table<K, V>>
{
    fn raw_slot(&self) -> &'m Option<(K, V)> {
        &self.table.buckets[self.bucket].slots[self.slot]
    }
}

impl<'m, K: 'm + Eq + Hash, V: 'm> MatchSlot<&'m mut Table<K, V>>
{
    fn val(&self) -> &V {
        &self.table.buckets[self.bucket].slots[self.slot]
            .as_ref()
            .unwrap()
            .1
    }

    fn val_mut(&mut self) -> &mut V {
        &mut self.table.buckets[self.bucket].slots[self.slot]
            .as_mut()
            .unwrap()
            .1
    }

    fn into_val_mut(self) -> &'m mut V {
        &mut self.table.buckets[self.bucket].slots[self.slot]
            .as_mut()
            .unwrap()
            .1
    }

    fn remove(&mut self) -> Option<(K, V)> {
        self.table.size -= 1;
        *self.partial_mut() = 0;
        self.table.buckets[self.bucket].slots[self.slot].take()
    }

    fn partial_mut(&mut self) -> &mut PartialKey {
        &mut self.table.buckets[self.bucket].partials[self.slot]
    }

    fn raw_slot_mut(&mut self) -> &mut RawSlot<K, V> {
        &mut self.table.buckets[self.bucket].slots[self.slot]
    }

    fn insert(&mut self, hashkey: &HashKey, val: V) -> V {
        *self.partial_mut() = hashkey.partial;
        let (key, prev) = self.raw_slot_mut().take().unwrap();
        *self.raw_slot_mut() = Some((key, val));
        prev
    }
}

struct VacantSlot<M> {
    bucket: usize,
    slot: usize,
    table: M,
}

impl<'m, K: 'm + Eq + Hash, V: 'm> VacantSlot<&'m mut Table<K, V>>
{
    fn into_val_mut(self) -> &'m mut V {
        &mut self.table.buckets[self.bucket].slots[self.slot]
            .as_mut()
            .unwrap()
            .1
    }

    fn raw_slot_mut(&mut self) -> &mut RawSlot<K, V> {
        &mut self.table.buckets[self.bucket].slots[self.slot]
    }

    fn partial_mut(&mut self) -> &mut PartialKey {
        &mut self.table.buckets[self.bucket].partials[self.slot]
    }

    fn insert<S>(&mut self, state: &S, hashkey: &HashKey, key: K, val: V)
    where
        S: BuildHasher,
    {
        let insert_result = self.insert_and_displace(hashkey, key, val);
        self.table.size += 1;

        if let InsertResult::Displaced(displaced) = insert_result {
            let hashkey = displaced.hashkey();

            let result = {
                let mut slot = find_alt_slot(&mut *self.table, &hashkey, |k| k == &displaced.key);
                insert_loop(&mut slot, &hashkey, displaced.key, displaced.val, MAX_DISPLACEMENTS - 1)
            };

            if let Err(err) = result {
                // resize
                resize(self.table, state);
                let hashkey = HashKey::with_partial(hash(&err.key, state), err.partial);
                self.table.insert(state, &hashkey, err.key, err.val);
            }
        }
    }

    fn insert_and_displace(&mut self, hashkey: &HashKey, key: K, val: V) -> InsertResult<K, V> {
        let prev_partial = std::mem::replace(self.partial_mut(), hashkey.partial);
        std::mem::replace(self.raw_slot_mut(), Some((key, val))).map_or_else(
            || InsertResult::Open,
            |(pkey, pval)| {
                InsertResult::Displaced(Displaced {
                    key: pkey,
                    val: pval,
                    partial: prev_partial,
                    bucket: self.bucket,
                })
            },
        )
    }
}

fn resize<K, V, S>(table: &mut Table<K, V>, state: &S)
where
    K: Eq + Hash,
    S: BuildHasher,
{
    let mut new_table = Table::new(table.buckets.len() * 2);

    for (partial, key, val) in table.drain() {
        let hash = hash(&key, state);
        let hashkey = HashKey::with_partial(hash, partial);
        new_table.insert(state, &hashkey, key, val);
    }

    std::mem::replace(table, new_table);
}

fn insert_loop<K, V>(
    slot: &mut Slot<&mut Table<K, V>>,
    hashkey: &HashKey,
    key: K,
    val: V,
    max_displacements: u8,
) -> Result<(), ResizeError<K, V>>
where
    K: Eq + Hash,
{
    let mut num_displacements = 0;

    match slot {
        Slot::Match(_) => unreachable!(),
        Slot::Vacant(vslot) => {
            let mut insert_result = vslot.insert_and_displace(hashkey, key, val);

            while let InsertResult::Displaced(displaced) = insert_result {
                num_displacements += 1;
                if num_displacements >= max_displacements {
                    return Err(ResizeError {
                        key: displaced.key,
                        val: displaced.val,
                        partial: displaced.partial,
                    });
                }

                let hashkey = displaced.hashkey();
                let mut slot = find_alt_slot(&mut *vslot.table, &hashkey, |k| k == &displaced.key);

                match slot {
                    Slot::Match(_) => panic!("This shouldn't happen"),
                    Slot::Vacant(ref mut vslot) => {
                        insert_result =
                            vslot.insert_and_displace(&hashkey, displaced.key, displaced.val);
                    }
                }
            }

            Ok(())
        }
    }
}

enum InsertResult<K, V> {
    Open,
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
        HashKey::with_partial(self.bucket, self.partial)
    }
}

impl<'m, K: 'm, V: 'm, M: 'm> Slot<M>
where
    K: Eq + Hash,
    M: Deref<Target = Table<K, V>>,
{
    fn is_match(&self) -> bool {
        match self {
            Slot::Match(_) => true,
            _ => false,
        }
    }

    fn is_vacant(&self) -> bool {
        match self {
            Slot::Vacant(_) => true,
            _ => false,
        }
    }
}

impl<'m, K: 'm, V: 'm> Slot<&'m mut Table<K, V>>
where
    K: Eq + Hash,
{
    fn insert<S: BuildHasher>(&mut self, state: &S, hashkey: &HashKey, key: K, val: V) -> Option<V> {
        match self {
            Slot::Match(mslot) => Some(mslot.insert(hashkey, val)),
            Slot::Vacant(vslot) => {
                vslot.insert(state, hashkey, key, val);
                None
            }
        }
    }
}

struct ResizeError<K, V> {
    key: K,
    val: V,
    partial: PartialKey,
}

impl<K, V> fmt::Debug for ResizeError<K, V> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.debug_struct("ResizeError").finish()
    }
}

type RawSlot<K, V> = Option<(K, V)>;
type PartialKey = u8;

#[derive(Debug)]
struct Bucket<K, V> {
    partials: [PartialKey; SLOTS_PER_BUCKET],
    slots: [RawSlot<K, V>; SLOTS_PER_BUCKET],
}

#[derive(PartialEq, Eq)]
enum SlotStatus {
    Open,
    Match,
    Occupied,
}

impl<K, V> Bucket<K, V>
where
    K: Eq,
{
    fn new() -> Self {
        Bucket {
            partials: [0; SLOTS_PER_BUCKET],
            slots: [None, None, None, None],
        }
    }

    fn find_slot<F>(&self, hashkey: &HashKey, is_match: &F) -> (usize, SlotStatus)
    where
        F: Fn(&K) -> bool,
    {
        let mut open_slot = None;

        for i in 0..self.slots.len() {
            if self.partials[i] == 0 {
                open_slot = Some(i);
                continue;
            }

            if self.partials[i] != hashkey.partial {
                continue;
            }

            let (k, _) = &self.slots[i].as_ref().unwrap();
            if is_match(k) {
                return (i, SlotStatus::Match);
            }
        }

        if let Some(i) = open_slot {
            return (i, SlotStatus::Open);
        }

        (
            rand::random::<usize>() % SLOTS_PER_BUCKET,
            SlotStatus::Occupied,
        )
    }
}

impl<K, V> CuckooHashMap<K, V, RandomState>
where
    K: Hash + Eq,
{
    pub fn new() -> CuckooHashMap<K, V, RandomState> {
        CuckooHashMap::with_hasher(RandomState::new())
    }
}

impl<K, V, S> CuckooHashMap<K, V, S>
where
    K: Hash + Eq,
    S: BuildHasher,
{
    /// Creates an empty `HashMap` which will use the given hash builder to hash
    /// keys.
    ///
    /// The created map has the default initial capacity.
    ///
    /// Warning: `hash_builder` is normally randomly generated, and
    /// is designed to allow HashMaps to be resistant to attacks that
    /// cause many collisions and very poor performance. Setting it
    /// manually using this function can expose a DoS attack vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use cuckoo::CuckooHashMap;
    /// use std::collections::hash_map::RandomState;
    ///
    /// let s = RandomState::new();
    /// let mut map = CuckooHashMap::with_hasher(s);
    /// map.insert(1, 2);
    /// ```
    pub fn with_hasher(hash_builder: S) -> CuckooHashMap<K, V, S> {
        CuckooHashMap {
            state: hash_builder,
            table: Table::new(INITIAL_SIZE),
        }
    }

    /// Returns a reference to the map's [`BuildHasher`].
    ///
    /// [`BuildHasher`]: ../../std/hash/trait.BuildHasher.html
    ///
    /// # Examples
    ///
    /// ```
    /// use cuckoo::CuckooHashMap;
    /// use std::collections::hash_map::RandomState;
    ///
    /// let hasher = RandomState::new();
    /// let map: CuckooHashMap<i32, i32> = CuckooHashMap::with_hasher(hasher);
    /// let hasher: &RandomState = map.hasher();
    /// ```
    pub fn hasher(&self) -> &S {
        &self.state
    }

    /// Gets the given key's corresponding entry in the map for in-place manipulation.
    ///
    /// # Examples
    ///
    /// ```
    /// use cuckoo::CuckooHashMap;
    ///
    /// let mut letters = CuckooHashMap::new();
    ///
    /// for ch in "a short treatise on fungi".chars() {
    ///     let counter = letters.entry(ch).or_insert(0);
    ///     *counter += 1;
    /// }
    ///
    /// assert_eq!(letters[&'s'], 2);
    /// assert_eq!(letters[&'t'], 3);
    /// assert_eq!(letters[&'u'], 1);
    /// assert_eq!(letters.get(&'y'), None);
    /// ```
    pub fn entry(&mut self, key: K) -> Entry<K, V, S> {
        let hashkey = HashKey::new(&key, &self.state);
        let slot = find_slot(&mut self.table, &hashkey, |k| k == &key);

        match slot {
            Slot::Vacant(vslot) => Entry::Vacant(VacantEntry {
                key,
                hashkey,
                slot: vslot,
                state: &self.state,
            }),
            Slot::Match(mslot) => Entry::Occupied(OccupiedEntry { key, slot: mslot }),
        }
    }

    /// Inserts a key-value pair into the map.
    ///
    /// If the map did not have this key present, [`None`] is returned.
    ///
    /// If the map did have this key present, the value is updated, and the old
    /// value is returned. The key is not updated, though; this matters for
    /// types that can be `==` without being identical. See the [module-level
    /// documentation] for more.
    ///
    /// [`None`]: ../../std/option/enum.Option.html#variant.None
    /// [module-level documentation]: index.html#insert-and-complex-keys
    ///
    /// # Examples
    ///
    /// ```
    /// use cuckoo::CuckooHashMap;
    ///
    /// let mut map = CuckooHashMap::new();
    /// assert_eq!(map.insert(37, "a"), None);
    /// //assert_eq!(map.is_empty(), false); // TODO re-enable when is_empty() is implemented
    ///
    /// map.insert(37, "b");
    /// assert_eq!(map.insert(37, "c"), Some("b"));
    /// assert_eq!(map[&37], "c");
    /// ```
    pub fn insert(&mut self, key: K, val: V) -> Option<V> {
        let hashkey = HashKey::new(&key, &self.state);
        self.table.insert(&self.state, &hashkey, key, val)
    }

    /// Returns a reference to the value corresponding to the key.
    ///
    /// The key may be any borrowed form of the map's key type, but
    /// [`Hash`] and [`Eq`] on the borrowed form *must* match those for
    /// the key type.
    ///
    /// [`Eq`]: ../../std/cmp/trait.Eq.html
    /// [`Hash`]: ../../std/hash/trait.Hash.html
    ///
    /// # Examples
    ///
    /// ```
    /// use cuckoo::CuckooHashMap;
    ///
    /// let mut map = CuckooHashMap::new();
    /// map.insert(1, "a");
    /// assert_eq!(map.get(&1), Some(&"a"));
    /// assert_eq!(map.get(&2), None);
    /// ```
    pub fn get<'a, Q>(&'a self, key: &Q) -> Option<&'a V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let slot = Self::find_slot(&self.table, &self.state, key);
        match slot {
            Slot::Match(match_slot) => match_slot.raw_slot().as_ref().map(|(_, v)| v),
            _ => None,
        }
    }

    /// Returns a mutable reference to the value corresponding to the key.
    ///
    /// The key may be any borrowed form of the map's key type, but
    /// [`Hash`] and [`Eq`] on the borrowed form *must* match those for
    /// the key type.
    ///
    /// [`Eq`]: ../../std/cmp/trait.Eq.html
    /// [`Hash`]: ../../std/hash/trait.Hash.html
    ///
    /// # Examples
    ///
    /// ```
    /// use cuckoo::CuckooHashMap;
    ///
    /// let mut map = CuckooHashMap::new();
    /// map.insert(1, "a");
    /// if let Some(x) = map.get_mut(&1) {
    ///     *x = "b";
    /// }
    /// assert_eq!(map[&1], "b");
    /// ```
    pub fn get_mut<Q: ?Sized>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        let slot = Self::find_slot(&mut self.table, &self.state, key);
        match slot {
            Slot::Match(match_slot) => Some(match_slot.into_val_mut()),
            _ => None,
        }
    }

    /// Returns true if the map contains a value for the specified key.
    ///
    /// The key may be any borrowed form of the map's key type, but
    /// [`Hash`] and [`Eq`] on the borrowed form *must* match those for
    /// the key type.
    ///
    /// [`Eq`]: ../../std/cmp/trait.Eq.html
    /// [`Hash`]: ../../std/hash/trait.Hash.html
    ///
    /// # Examples
    ///
    /// ```
    /// use cuckoo::CuckooHashMap;
    ///
    /// let mut map = CuckooHashMap::new();
    /// map.insert(1, "a");
    /// assert_eq!(map.contains_key(&1), true);
    /// assert_eq!(map.contains_key(&2), false);
    /// ```
    pub fn contains_key<Q: ?Sized>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        let slot = Self::find_slot(&self.table, &self.state, key);
        match slot {
            Slot::Match(_) => true,
            _ => false,
        }
    }

    /// Removes a key from the map, returning the stored key and value if the
    /// key was previously in the map.
    ///
    /// The key may be any borrowed form of the map's key type, but
    /// [`Hash`] and [`Eq`] on the borrowed form *must* match those for
    /// the key type.
    ///
    /// [`Eq`]: ../../std/cmp/trait.Eq.html
    /// [`Hash`]: ../../std/hash/trait.Hash.html
    ///
    /// # Examples
    ///
    /// ```
    /// use cuckoo::CuckooHashMap;
    ///
    /// let mut map = CuckooHashMap::new();
    /// map.insert(1, "a");
    /// assert_eq!(map.remove_entry(&1), Some((1, "a")));
    /// assert_eq!(map.remove(&1), None);
    /// ```
    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.remove_entry(key).map(|(_, v)| v)
    }

    /// Removes a key from the map, returning the stored key and value if the
    /// key was previously in the map.
    ///
    /// The key may be any borrowed form of the map's key type, but
    /// [`Hash`] and [`Eq`] on the borrowed form *must* match those for
    /// the key type.
    ///
    /// [`Eq`]: ../../std/cmp/trait.Eq.html
    /// [`Hash`]: ../../std/hash/trait.Hash.html
    ///
    /// # Examples
    ///
    /// ```
    /// use cuckoo::CuckooHashMap;
    ///
    /// let mut map = CuckooHashMap::new();
    /// map.insert(1, "a");
    /// assert_eq!(map.remove_entry(&1), Some((1, "a")));
    /// assert_eq!(map.remove(&1), None);
    /// ```
    pub fn remove_entry<Q>(&mut self, key: &Q) -> Option<(K, V)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        let mut slot = Self::find_slot(&mut self.table, &self.state, key);
        match slot {
            Slot::Match(ref mut match_slot) => match_slot.remove(),
            _ => None,
        }
    }

    /// Returns the number of elements in the map.
    ///
    /// # Examples
    ///
    /// ```
    /// use cuckoo::CuckooHashMap;
    ///
    /// let mut a = CuckooHashMap::new();
    /// assert_eq!(a.len(), 0);
    /// a.insert(1, "a");
    /// assert_eq!(a.len(), 1);
    /// ```
    pub fn len(&self) -> usize {
        self.table.size
    }

    /// Returns true if the map contains no elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use cuckoo::CuckooHashMap;
    ///
    /// let mut a = CuckooHashMap::new();
    /// assert!(a.is_empty());
    /// a.insert(1, "a");
    /// assert!(!a.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// An iterator visiting all key-value pairs in arbitrary order.
    /// The iterator element type is `(&'a K, &'a V)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use cuckoo::CuckooHashMap;
    ///
    /// let mut map = CuckooHashMap::new();
    /// map.insert("a", 1);
    /// map.insert("b", 2);
    /// map.insert("c", 3);
    ///
    /// for (key, val) in map.iter() {
    ///     println!("key: {} val: {}", key, val);
    /// }
    /// ```
    pub fn iter(&self) -> Iter<K, V> {
        Iter {
            inner: self.table.iter(),
        }
    }

    /// An iterator visiting all keys in arbitrary order.
    /// The iterator element type is `&'a K`.
    ///
    /// # Examples
    ///
    /// ```
    /// use cuckoo::CuckooHashMap;
    ///
    /// let mut map = CuckooHashMap::new();
    /// map.insert("a", 1);
    /// map.insert("b", 2);
    /// map.insert("c", 3);
    ///
    /// for key in map.keys() {
    ///     println!("{}", key);
    /// }
    /// ```
    pub fn keys(&self) -> Keys<K, V> {
        Keys { inner: self.iter() }
    }

    /// An iterator visiting all values in arbitrary order.
    /// The iterator element type is `&'a V`.
    ///
    /// # Examples
    ///
    /// ```
    /// use cuckoo::CuckooHashMap;
    ///
    /// let mut map = CuckooHashMap::new();
    /// map.insert("a", 1);
    /// map.insert("b", 2);
    /// map.insert("c", 3);
    ///
    /// for val in map.values() {
    ///     println!("{}", val);
    /// }
    /// ```
    pub fn values(&self) -> Values<K, V> {
        Values { inner: self.iter() }
    }

    /// An iterator visiting all key-value pairs in arbitrary order,
    /// with mutable references to the values.
    /// The iterator element type is `(&'a K, &'a mut V)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use cuckoo::CuckooHashMap;
    ///
    /// let mut map = CuckooHashMap::new();
    /// map.insert("a", 1);
    /// map.insert("b", 2);
    /// map.insert("c", 3);
    ///
    /// // Update all values
    /// for (_, val) in map.iter_mut() {
    ///     *val *= 2;
    /// }
    ///
    /// for (key, val) in &map {
    ///     println!("key: {} val: {}", key, val);
    /// }
    /// ```
    pub fn iter_mut(&mut self) -> IterMut<K, V> {
        IterMut {
            inner: self.table.iter_mut(),
        }
    }

    /// An iterator visiting all values mutably in arbitrary order.
    /// The iterator element type is `&'a mut V`.
    ///
    /// # Examples
    ///
    /// ```
    /// use cuckoo::CuckooHashMap;
    ///
    /// let mut map = CuckooHashMap::new();
    ///
    /// map.insert("a", 1);
    /// map.insert("b", 2);
    /// map.insert("c", 3);
    ///
    /// for val in map.values_mut() {
    ///     *val = *val + 10;
    /// }
    ///
    /// for val in map.values() {
    ///     println!("{}", val);
    /// }
    /// ```
    pub fn values_mut(&mut self) -> ValuesMut<K, V> {
        ValuesMut { inner: self.iter_mut() }
    }

    /// Clears the map, returning all key-value pairs as an iterator. Keeps the
    /// allocated memory for reuse.
    ///
    /// # Examples
    ///
    /// ```
    /// use cuckoo::CuckooHashMap;
    ///
    /// let mut a = CuckooHashMap::new();
    /// a.insert(1, "a");
    /// a.insert(2, "b");
    ///
    /// for (k, v) in a.drain().take(1) {
    ///     assert!(k == 1 || k == 2);
    ///     assert!(v == "a" || v == "b");
    /// }
    ///
    /// assert!(a.is_empty());
    /// ```
    pub fn drain(&mut self) -> Drain<K, V> {
        Drain { inner: self.table.drain() }
    }

    /// Clears the map, removing all key-value pairs. Keeps the allocated memory
    /// for reuse.
    ///
    /// # Examples
    ///
    /// ```
    /// use cuckoo::CuckooHashMap;
    ///
    /// let mut a = CuckooHashMap::new();
    /// a.insert(1, "a");
    /// a.clear();
    /// assert!(a.is_empty());
    /// ```
    pub fn clear(&mut self) {
        self.drain();
    }

    /// Retains only the elements specified by the predicate.
    ///
    /// In other words, remove all pairs `(k, v)` such that `f(&k,&mut v)` returns `false`.
    ///
    /// # Examples
    ///
    /// ```
    /// use cuckoo::CuckooHashMap;
    ///
    /// let mut map: CuckooHashMap<i32, i32> = (0..8).map(|x|(x, x*10)).collect();
    /// map.retain(|&k, _| k % 2 == 0);
    /// assert_eq!(map.len(), 4);
    /// ```
    pub fn retain<F>(&mut self, f: F)
        where F: FnMut(&K, &mut V) -> bool
    {
        self.table.retain(f);
    }

    fn find_slot<M, Q>(table: M, state: &S, key: &Q) -> Slot<M>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
        M: Deref<Target = Table<K, V>>,
    {
        let hashkey = HashKey::new(key, state);
        find_slot(table, &hashkey, |k| k.borrow() == key)
    }
}

impl<'a, K, V, S> IntoIterator for &'a CuckooHashMap<K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher,
{
    type Item = (&'a K, &'a V);
    type IntoIter = Iter<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, K, V, S> IntoIterator for &'a mut CuckooHashMap<K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher,
{
    type Item = (&'a K, &'a mut V);
    type IntoIter = IterMut<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<K, V, S> IntoIterator for CuckooHashMap<K, V, S>
    where K: Eq + Hash,
          S: BuildHasher
{
    type Item = (K, V);
    type IntoIter = IntoIter<K, V>;

    /// Creates a consuming iterator, that is, one that moves each key-value
    /// pair out of the map in arbitrary order. The map cannot be used after
    /// calling this.
    ///
    /// # Examples
    ///
    /// ```
    /// use cuckoo::CuckooHashMap;
    ///
    /// let mut map = CuckooHashMap::new();
    /// map.insert("a", 1);
    /// map.insert("b", 2);
    /// map.insert("c", 3);
    ///
    /// // Not possible with .iter()
    /// let vec: Vec<(&str, i32)> = map.into_iter().collect();
    /// ```
    fn into_iter(self) -> IntoIter<K, V> {
        IntoIter { inner: self.table.into_iter() }
    }
}

pub struct IntoIter<K: Eq + Hash, V> {
    inner: TableIntoIter<K, V>,
}

impl<K: Eq + Hash, V> Iterator for IntoIter<K, V> {
    type Item = (K, V);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

impl<K, V, S> Extend<(K, V)> for CuckooHashMap<K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher,
{
    fn extend<T: IntoIterator<Item = (K, V)>>(&mut self, iter: T) {
        for (k, v) in iter {
            self.insert(k, v);
        }
    }
}

impl<'a, K, V, S> Extend<(&'a K, &'a V)> for CuckooHashMap<K, V, S>
where
    K: Eq + Hash + Copy,
    V: Copy,
    S: BuildHasher,
{
    fn extend<T: IntoIterator<Item = (&'a K, &'a V)>>(&mut self, iter: T) {
        self.extend(iter.into_iter().map(|(&k, &v)| (k, v)))
    }
}

impl<K, V, S> iter::FromIterator<(K, V)> for CuckooHashMap<K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher + Default,
{
    fn from_iter<T: IntoIterator<Item = (K, V)>>(iter: T) -> CuckooHashMap<K, V, S> {
        let mut map = CuckooHashMap::with_hasher(Default::default());
        map.extend(iter);
        map
    }
}

pub struct Iter<'a, K: 'a + Eq + Hash, V: 'a> {
    inner: TableIter<'a, K, V>,
}

impl<'a, K: 'a + Eq + Hash, V: 'a> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

pub struct Keys<'a, K: 'a + Eq + Hash, V: 'a> {
    inner: Iter<'a, K, V>,
}

impl<'a, K: 'a + Eq + Hash, V: 'a> Iterator for Keys<'a, K, V> {
    type Item = &'a K;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(k, _)| k)
    }
}

pub struct Values<'a, K: 'a + Eq + Hash, V: 'a> {
    inner: Iter<'a, K, V>,
}

impl<'a, K: 'a + Eq + Hash, V: 'a> Iterator for Values<'a, K, V> {
    type Item = &'a V;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(_, v)| v)
    }
}

pub struct IterMut<'a, K: 'a + Eq + Hash, V: 'a> {
    inner: TableIterMut<'a, K, V>,
}

impl<'a, K: 'a + Eq + Hash, V: 'a> Iterator for IterMut<'a, K, V> {
    type Item = (&'a K, &'a mut V);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

pub struct ValuesMut<'a, K: 'a + Eq + Hash, V: 'a> {
    inner: IterMut<'a, K, V>,
}

impl<'a, K: 'a + Eq + Hash, V: 'a> Iterator for ValuesMut<'a, K, V> {
    type Item = &'a mut V;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(_, v)| v)
    }
}

pub struct Drain<'a, K: 'a + Eq + Hash, V: 'a> {
    inner: TableDrain<'a, K, V>,
}

impl<'a, K: 'a + Eq + Hash, V: 'a> Iterator for Drain<'a, K, V> {
    type Item = (K, V);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(_, k, v)| (k, v))
    }
}

pub enum Entry<'a, K, V, S>
where
    K: 'a + Hash + Eq,
    V: 'a,
    S: 'a + BuildHasher,
{
    Occupied(OccupiedEntry<'a, K, V>),
    Vacant(VacantEntry<'a, K, V, S>),
}

impl<'a, K, V, S> Entry<'a, K, V, S>
where
    K: 'a + Hash + Eq,
    V: 'a,
    S: BuildHasher,
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

    /// Ensures a value is in the entry by inserting the result of the default function if empty,
    /// and returns a mutable reference to the value in the entry.
    ///
    /// # Examples
    ///
    /// ```
    /// use cuckoo::CuckooHashMap;
    ///
    /// let mut map: CuckooHashMap<&str, String> = CuckooHashMap::new();
    /// let s = "hoho".to_string();
    ///
    /// map.entry("poneyland").or_insert_with(|| s);
    ///
    /// assert_eq!(map["poneyland"], "hoho".to_string());
    /// ```
    pub fn or_insert_with<F: FnOnce() -> V>(self, default: F) -> &'a mut V {
        match self {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.insert(default()),
        }
    }

    /// Returns a reference to this entry's key.
    ///
    /// # Examples
    ///
    /// ```
    /// use cuckoo::CuckooHashMap;
    ///
    /// let mut map: CuckooHashMap<&str, u32> = CuckooHashMap::new();
    /// assert_eq!(map.entry("poneyland").key(), &"poneyland");
    /// ```
    pub fn key(&self) -> &K {
        match *self {
            Entry::Occupied(ref entry) => entry.key(),
            Entry::Vacant(ref entry) => entry.key(),
        }
    }

    /// Provides in-place mutable access to an occupied entry before any
    /// potential inserts into the map.
    ///
    /// # Examples
    ///
    /// ```
    /// use cuckoo::CuckooHashMap;
    ///
    /// let mut map: CuckooHashMap<&str, u32> = CuckooHashMap::new();
    ///
    /// map.entry("poneyland")
    ///    .and_modify(|e| { *e += 1 })
    ///    .or_insert(42);
    /// assert_eq!(map["poneyland"], 42);
    ///
    /// map.entry("poneyland")
    ///    .and_modify(|e| { *e += 1 })
    ///    .or_insert(42);
    /// assert_eq!(map["poneyland"], 43);
    /// ```
    pub fn and_modify<F>(self, f: F) -> Self
    where
        F: FnOnce(&mut V),
    {
        match self {
            Entry::Occupied(mut entry) => {
                f(entry.get_mut());
                Entry::Occupied(entry)
            }
            Entry::Vacant(entry) => Entry::Vacant(entry),
        }
    }
}

impl<'a, K: Hash + Eq, V: Default, S: BuildHasher> Entry<'a, K, V, S> {
    /// Ensures a value is in the entry by inserting the default value if empty,
    /// and returns a mutable reference to the value in the entry.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() {
    /// use cuckoo::CuckooHashMap;
    ///
    /// let mut map: CuckooHashMap<&str, Option<u32>> = CuckooHashMap::new();
    /// map.entry("poneyland").or_default();
    ///
    /// assert_eq!(map["poneyland"], None);
    /// # }
    /// ```
    pub fn or_default(self) -> &'a mut V {
        match self {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.insert(Default::default()),
        }
    }
}

pub struct OccupiedEntry<'a, K, V>
where
    K: 'a + Eq + Hash,
    V: 'a,
{
    key: K,
    slot: MatchSlot<&'a mut Table<K, V>>,
}

impl<'a, K, V> OccupiedEntry<'a, K, V>
where
    K: 'a + Eq + Hash,
    V: 'a,
{
    /// Gets a reference to the key in the entry.
    ///
    /// # Examples
    ///
    /// ```
    /// use cuckoo::CuckooHashMap;
    ///
    /// let mut map: CuckooHashMap<&str, u32> = CuckooHashMap::new();
    /// map.entry("poneyland").or_insert(12);
    /// assert_eq!(map.entry("poneyland").key(), &"poneyland");
    /// ```
    pub fn key(&self) -> &K {
        &self.key
    }

    /// Take the ownership of the key and value from the map.
    ///
    /// # Examples
    ///
    /// ```
    /// use cuckoo::{CuckooHashMap, Entry};
    ///
    /// let mut map: CuckooHashMap<&str, u32> = CuckooHashMap::new();
    /// map.entry("poneyland").or_insert(12);
    ///
    /// if let Entry::Occupied(o) = map.entry("poneyland") {
    ///     // We delete the entry from the map.
    ///     o.remove_entry();
    /// }
    ///
    /// assert_eq!(map.contains_key("poneyland"), false);
    /// ```
    pub fn remove_entry(self) -> (K, V) {
        let mut slot = self.slot;
        slot.remove().unwrap()
    }

    /// Gets a reference to the value in the entry.
    ///
    /// # Examples
    ///
    /// ```
    /// use cuckoo::{CuckooHashMap, Entry};
    ///
    /// let mut map: CuckooHashMap<&str, u32> = CuckooHashMap::new();
    /// map.entry("poneyland").or_insert(12);
    ///
    /// if let Entry::Occupied(o) = map.entry("poneyland") {
    ///     assert_eq!(o.get(), &12);
    /// }
    /// ```
    pub fn get(&self) -> &V {
        self.slot.val()
    }

    /// Gets a mutable reference to the value in the entry.
    ///
    /// If you need a reference to the `OccupiedEntry` which may outlive the
    /// destruction of the `Entry` value, see [`into_mut`].
    ///
    /// [`into_mut`]: #method.into_mut
    ///
    /// # Examples
    ///
    /// ```
    /// use cuckoo::{CuckooHashMap, Entry};
    ///
    /// let mut map: CuckooHashMap<&str, u32> = CuckooHashMap::new();
    /// map.entry("poneyland").or_insert(12);
    ///
    /// assert_eq!(map["poneyland"], 12);
    /// if let Entry::Occupied(mut o) = map.entry("poneyland") {
    ///     *o.get_mut() += 10;
    ///     assert_eq!(*o.get(), 22);
    ///
    ///     // We can use the same Entry multiple times.
    ///     *o.get_mut() += 2;
    /// }
    ///
    /// assert_eq!(map["poneyland"], 24);
    /// ```
    pub fn get_mut(&mut self) -> &mut V {
        self.slot.val_mut()
    }

    /// Converts the OccupiedEntry into a mutable reference to the value in the entry
    /// with a lifetime bound to the map itself.
    ///
    /// If you need multiple references to the `OccupiedEntry`, see [`get_mut`].
    ///
    /// [`get_mut`]: #method.get_mut
    ///
    /// # Examples
    ///
    /// ```
    /// use cuckoo::{CuckooHashMap, Entry};
    ///
    /// let mut map: CuckooHashMap<&str, u32> = CuckooHashMap::new();
    /// map.entry("poneyland").or_insert(12);
    ///
    /// assert_eq!(map["poneyland"], 12);
    /// if let Entry::Occupied(o) = map.entry("poneyland") {
    ///     *o.into_mut() += 10;
    /// }
    ///
    /// assert_eq!(map["poneyland"], 22);
    /// ```
    pub fn into_mut(self) -> &'a mut V {
        self.slot.into_val_mut()
    }

    /// Sets the value of the entry, and returns the entry's old value.
    ///
    /// # Examples
    ///
    /// ```
    /// use cuckoo::{CuckooHashMap, Entry};
    ///
    /// let mut map: CuckooHashMap<&str, u32> = CuckooHashMap::new();
    /// map.entry("poneyland").or_insert(12);
    ///
    /// if let Entry::Occupied(mut o) = map.entry("poneyland") {
    ///     assert_eq!(o.insert(15), 12);
    /// }
    ///
    /// assert_eq!(map["poneyland"], 15);
    /// ```
    pub fn insert(&mut self, value: V) -> V {
        std::mem::replace(self.get_mut(), value)
    }

    /// Takes the value out of the entry, and returns it.
    ///
    /// # Examples
    ///
    /// ```
    /// use cuckoo::{CuckooHashMap, Entry};
    ///
    /// let mut map: CuckooHashMap<&str, u32> = CuckooHashMap::new();
    /// map.entry("poneyland").or_insert(12);
    ///
    /// if let Entry::Occupied(o) = map.entry("poneyland") {
    ///     assert_eq!(o.remove(), 12);
    /// }
    ///
    /// assert_eq!(map.contains_key("poneyland"), false);
    /// ```
    pub fn remove(self) -> V {
        self.remove_entry().1
    }
}

pub struct VacantEntry<'a, K, V, S>
where
    K: 'a + Hash + Eq,
    V: 'a,
    S: 'a + BuildHasher,
{
    key: K,
    hashkey: HashKey,
    slot: VacantSlot<&'a mut Table<K, V>>,
    state: &'a S,
}

impl<'a, K, V, S> VacantEntry<'a, K, V, S>
where
    K: 'a + Hash + Eq,
    V: 'a,
    S: BuildHasher,
{
    /// Gets a reference to the key that would be used when inserting a value
    /// through the `VacantEntry`.
    ///
    /// # Examples
    ///
    /// ```
    /// use cuckoo::{CuckooHashMap, Entry};
    ///
    /// let mut map: CuckooHashMap<&str, u32> = CuckooHashMap::new();
    /// assert_eq!(map.entry("poneyland").key(), &"poneyland");
    /// ```
    pub fn key(&self) -> &K {
        &self.key
    }

    /// Take ownership of the key.
    ///
    /// # Examples
    ///
    /// ```
    /// use cuckoo::{CuckooHashMap, Entry};
    ///
    /// let mut map: CuckooHashMap<&str, u32> = CuckooHashMap::new();
    ///
    /// if let Entry::Vacant(v) = map.entry("poneyland") {
    ///     v.into_key();
    /// }
    /// ```
    pub fn into_key(self) -> K {
        self.key
    }

    /// Sets the value of the entry with the VacantEntry's key,
    /// and returns a mutable reference to it.
    ///
    /// # Examples
    ///
    /// ```
    /// use cuckoo::{CuckooHashMap, Entry};
    ///
    /// let mut map: CuckooHashMap<&str, u32> = CuckooHashMap::new();
    ///
    /// if let Entry::Vacant(o) = map.entry("poneyland") {
    ///     o.insert(37);
    /// }
    /// assert_eq!(map["poneyland"], 37);
    /// ```
    pub fn insert(self, value: V) -> &'a mut V {
        let mut slot = self.slot;
        slot.insert(self.state, &self.hashkey, self.key, value);
        slot.into_val_mut()
    }
}

impl<'a, K, Q: ?Sized, V, S> Index<&'a Q> for CuckooHashMap<K, V, S>
where
    K: Eq + Hash + Borrow<Q>,
    Q: Eq + Hash,
    S: BuildHasher,
{
    type Output = V;

    fn index(&self, key: &Q) -> &V {
        self.get(key).expect("no entry found for key")
    }
}

impl<K, V, S> PartialEq for CuckooHashMap<K, V, S>
where
    K: Eq + Hash,
    V: PartialEq,
    S: BuildHasher,
{
    fn eq(&self, other: &CuckooHashMap<K, V, S>) -> bool {
        if self.len() != other.len() {
            return false;
        }

        self.iter().all(|(k, v)| other.get(k).map_or(false, |v2| *v == *v2))
    }
}

impl<K, V, S> Eq for CuckooHashMap<K, V, S>
where
    K: Eq + Hash,
    V: PartialEq,
    S: BuildHasher,
{
}

#[cfg(test)]
mod internal_tests {
    use super::*;

    #[test]
    fn table_basics() {
        let state = RandomState::new();
        let mut table = Table::new(INITIAL_SIZE);
        let hashkey = HashKey::from_hash(hash(&7832, &state));

        {
            let mut slot = find_slot(&mut table, &hashkey, |k| *k == 4);
            assert!(slot.is_vacant());
            slot.insert(&state, &hashkey, 4, "hello".to_string());
        }

        let slot = find_slot(&mut table, &hashkey, |k| *k == 4);
        assert!(slot.is_match());
    }

    #[test]
    fn hashkey_invertable() {
        let table_size = 128;
        let hashkey = HashKey::from_hash(74839);
        let index = hashkey.index(table_size);
        let alt_index = hashkey.alt_index(table_size);

        let hashkey = HashKey::with_partial(alt_index, hashkey.partial);
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
