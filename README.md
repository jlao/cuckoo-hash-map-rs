This is a Cuckoo hash map implemented in Rust. All the implemented methods are
API compatible with HashMap with the exception of anything that is capacity
related. For instance, `capacity()` returns how many open slots there are but
that does not guarantee that many items can be inserted without resizing.

I created this as a way to learn Rust. It is not optimized and is almost
certainly slower than HashMap in everyway. Don't use this!

I used partial-key hashing as described in the MemC3 paper: http://www.cs.cmu.edu/~dga/papers/memc3-nsdi2013.pdf

And implemented in libcuckoo: https://github.com/efficient/libcuckoo