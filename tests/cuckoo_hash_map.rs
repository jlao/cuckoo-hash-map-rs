extern crate cuckoo;

use std::collections::HashSet;
use std::hash::Hash;
use cuckoo::CuckooHashMap;

#[test]
fn get() {
    let mut map = CuckooHashMap::new();

    assert_eq!(None, map.get(&1));

    map.insert(1, "hello".to_string());
    assert_eq!(Some(&"hello".to_string()), map.get(&1));
}

#[test]
fn insert() {
    let mut map = CuckooHashMap::new();

    let prev = map.insert(1, "hello".to_string());
    assert_eq!(None, prev);
    assert_eq!(Some(&"hello".to_string()), map.get(&1));

    let prev = map.insert(1, "bonjour".to_string());
    assert_eq!(
        Some("hello".to_string()),
        prev,
        "insert() did not return previous value when replacing"
    );

    assert_eq!(
        Some(&"bonjour".to_string()),
        map.get(&1),
        "value did not get replaced"
    );
}

#[test]
fn remove() {
    let mut map = CuckooHashMap::new();
    assert_eq!(map.remove(&1), None);

    map.insert(1, "hello".to_string());
    assert_eq!(map.remove(&1), Some("hello".to_string()));
    assert_eq!(map.remove(&1), None);
}

#[test]
fn remove_entry() {
    let mut map = CuckooHashMap::new();
    assert_eq!(map.remove_entry(&1), None);

    map.insert(1, "hello".to_string());
    assert_eq!(map.remove_entry(&1), Some((1, "hello".to_string())));
    assert_eq!(map.remove(&1), None);
}

#[test]
fn len() {
    let mut map = CuckooHashMap::new();
    assert_eq!(map.len(), 0);

    for i in 1..10 {
        map.insert(i, i.to_string());
        assert_eq!(map.len(), i);
    }

    let mut expected_len = map.len();
    for i in 1..5 {
        map.remove(&i);
        expected_len -= 1;
        assert_eq!(map.len(), expected_len);
    }

    map.insert(42, 42.to_string());
    expected_len += 1;
    assert_eq!(map.len(), expected_len);
}

#[test]
fn is_empty() {
    let mut map = CuckooHashMap::new();
    assert!(map.is_empty());

    map.insert(1, 1.to_string());
    assert!(!map.is_empty());

    map.remove(&1);
    assert!(map.is_empty());
}

#[test]
fn iter() {
    let mut map = CuckooHashMap::new();
    let mut expected = HashSet::new();

    for i in 0..5 {
        map.insert(i, i.to_string());
        expected.insert((i, i.to_string()));

        let actual: HashSet<(usize, String)> = map.iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        assert_eq!(actual, expected);
    }
}

#[test]
fn keys() {
    let mut map = CuckooHashMap::new();
    let mut expected = HashSet::new();

    for i in 0..5 {
        map.insert(i, i.to_string());
        expected.insert(i);

        let actual: HashSet<usize> = map.keys().cloned().collect();
        assert_eq!(actual, expected);
    }
}

#[test]
fn values() {
    let mut map = CuckooHashMap::new();
    let mut expected = HashSet::new();

    for i in 0..5 {
        map.insert(i, i.to_string());
        expected.insert(i.to_string());

        let actual: HashSet<_> = map.values().cloned().collect();
        assert_eq!(actual, expected);
    }
}

#[test]
fn iter_mut() {
    let mut map = CuckooHashMap::new();
    let mut expected = HashSet::new();

    for i in 0..5 {
        map.insert(i.to_string(), i);
        expected.insert((i.to_string(), i * 2));
    }

    for (key, val) in map.iter_mut() {
        assert_eq!(key, &val.to_string());
        *val *= 2;
    }

    let actual: HashSet<_> = map.iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();

    assert_eq!(actual, expected);
}

#[test]
fn values_mut() {
    let mut map = CuckooHashMap::new();
    let mut expected = HashSet::new();

    for i in 0..5 {
        map.insert(i.to_string(), i);
        expected.insert(i * 2);
    }

    for val in map.values_mut() {
        *val *= 2;
    }

    let actual: HashSet<_> = map.values().cloned().collect();
    assert_eq!(actual, expected);
}

#[test]
fn drain() {
    let mut map = CuckooHashMap::new();
    let mut expected = HashSet::new();

    for i in 0..5 {
        map.insert(i, i.to_string());
        expected.insert((i, i.to_string()));
    }

    let actual: HashSet<(usize, String)> = map.drain().collect();
    assert_eq!(actual, expected);
    assert_eq!(map.len(), 0);
    assert!(map.is_empty());

    assert_eq!(map.iter().next(), None);
    assert_eq!(map.keys().next(), None);
    assert_eq!(map.values().next(), None);
    assert_eq!(map.iter_mut().next(), None);

    for i in 0..5 {
        map.insert(i, i.to_string());
        expected.insert((i, i.to_string()));
    }

    {
        assert!(map.drain().take(1).next().is_some());
    }

    assert_map_is_empty(&mut map);
}

#[test]
fn clear() {
    let mut map = CuckooHashMap::new();
    map.insert(1, 2);
    map.insert(2, 3);
    map.insert(3, 4);
    map.clear();

    assert_map_is_empty(&mut map);
}

#[test]
fn retain() {
    let mut map = CuckooHashMap::new();
    for i in 0..10 {
        map.insert(i, i);
    }

    let expected: HashSet<_> = map.iter()
        .filter_map(|(&k, &v)|
            if k % 2 == 0 {
                Some((k, v))
            } else {
                None
            })
        .collect();

    map.retain(|k, _| k % 2 == 0);
    assert_eq!(map.len(), 5);

    let actual: HashSet<_> = map.iter()
        .map(|(&k, &v)| (k, v))
        .collect();

    assert_eq!(actual, expected);
}

#[test]
fn extend() {
    let mut map = CuckooHashMap::new();
    let extender: HashSet<_> = (0..10).map(|i| (i, i.to_string())).collect();
    let expected = extender.clone();

    map.extend(extender);

    let actual: HashSet<_> = map.iter()
        .map(|(&k, v)| (k, v.clone()))
        .collect();

    assert_eq!(actual, expected);
}

#[test]
fn from_iterator() {
    let map: CuckooHashMap<_, _> = (0..10).map(|i| (i, i * 2)).collect();

    assert_eq!(map.len(), 10);
    for i in 0..10 {
        assert_eq!(map.get(&i), Some(&(i * 2)));
    }
}

#[test]
fn equals() {
    let map1: CuckooHashMap<_, _> = (0..10).map(|i| (i, i * 2)).collect();
    let map2: CuckooHashMap<_, _> = (0..10).map(|i| (i, i * 2)).collect();
    assert_eq!(map1, map2);

    // Different values
    let map2: CuckooHashMap<_, _> = (0..10).map(|i| (i, i * 3)).collect();
    assert_ne!(map1, map2);

    // Different number of items
    let map2: CuckooHashMap<_, _> = (0..5).map(|i| (i, i * 2)).collect();
    assert_ne!(map1, map2);

    // Different keys and values
    let map2: CuckooHashMap<_, _> = (0..10).map(|i| (i * 2, i * 3)).collect();
    assert_ne!(map1, map2);
}

#[test]
fn into_iter() {
    let map: CuckooHashMap<i32, String> = (0..10).map(|i| (i, i.to_string())).collect();
    let expected: HashSet<(i32, String)> = (0..10).map(|i| (i, i.to_string())).collect();
    let actual: HashSet<(i32, String)> = map.into_iter().collect();
    assert_eq!(actual, expected);
}

fn assert_map_is_empty<K: Eq + Hash, V>(map: &mut CuckooHashMap<K, V>) {
    assert_eq!(map.len(), 0);
    assert!(map.is_empty());
    assert!(map.iter().next().is_none(), "iter() is not empty");
    assert!(map.keys().next().is_none(), "keys() is not empty");
    assert!(map.values().next().is_none(), "values() is not empty");
    assert!(map.values_mut().next().is_none(), "values() is not empty");
    assert!(map.iter_mut().next().is_none(), "iter_mut() is not empty");
}

#[test]
fn insert_more_than_capacity() {
    const NUM_ITEMS: usize = 128 * 4 + 1;

    let mut map = CuckooHashMap::new();

    for i in 0..NUM_ITEMS {
        assert_eq!(map.insert(i, i.to_string()), None);
    }

    for i in 0..NUM_ITEMS {
        assert_eq!(map.get(&i), Some(&i.to_string()));
    }

    assert_eq!(map.len(), NUM_ITEMS);
}