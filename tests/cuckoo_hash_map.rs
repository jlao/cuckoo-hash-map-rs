extern crate cuckoo;

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