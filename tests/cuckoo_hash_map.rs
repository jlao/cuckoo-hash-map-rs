extern crate cuckoo;

use std::collections::HashSet;
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
    let mut map: CuckooHashMap<usize, String> = CuckooHashMap::new();
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
    let mut map: CuckooHashMap<usize, String> = CuckooHashMap::new();
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
    let mut map: CuckooHashMap<usize, String> = CuckooHashMap::new();
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
    let mut map: CuckooHashMap<usize, String> = CuckooHashMap::new();
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
        map.drain().take(1);
    }

    assert_eq!(map.len(), 0);
    assert!(map.is_empty());
    assert_eq!(map.iter().next(), None, "iter() is not empty");
    assert_eq!(map.keys().next(), None, "keys() is not empty");
    assert_eq!(map.values().next(), None, "values() is not empty");
    assert_eq!(map.iter_mut().next(), None, "iter_mut() is not empty");
}

#[test]
#[ignore]
fn insert_more_than_capacity() {
    const NUM_ITEMS: u32 = 128 * 4 + 1;

    let mut map = CuckooHashMap::new();

    for i in 0..NUM_ITEMS {
        assert_eq!(map.insert(i, i.to_string()), None);
    }

    for i in 0..NUM_ITEMS {
        assert_eq!(map.get(&i), Some(&i.to_string()));
    }
}