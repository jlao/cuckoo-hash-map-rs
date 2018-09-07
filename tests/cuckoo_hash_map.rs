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

    let prev = map.insert(1, "bonjour".to_string());
    assert_eq!(
        Some("hello".to_string()),
        prev,
        "insert() did not return previous value when replacing");

    assert_eq!(
        Some(&"bonjour".to_string()),
        map.get(&1),
        "value did not get replaced");
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