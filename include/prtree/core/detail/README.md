# PRTree Core Implementation Details

This directory is reserved for modularizing the PRTree core implementation.

## Planned Structure

The current `prtree.h` (1617 lines) should be split into:

### 1. `types.h` - Common Types and Utilities
- Line 59-103: Type definitions, concepts, utility templates
- `IndexType`, `SignedIndexType` concepts
- `vec`, `svec`, `deque`, `queue` type aliases
- Utility functions: `as_pyarray()`, `list_list_to_arrays()`
- Constants: `REBUILD_THRE`
- Macros: `likely()`, `unlikely()`
- Compression functions

### 2. `bounding_box.h` - Bounding Box Class
- Line 130-251: `BB<D>` class
- Geometric operations on axis-aligned bounding boxes
- Intersection, union, containment tests
- Serialization support

### 3. `data_type.h` - Data Storage
- Line 252-277: `DataType<T, D>` class
- Storage for indices and coordinates
- Refinement data for precision

### 4. `pseudo_tree.h` - Pseudo PRTree
- Line 278-491: Pseudo PRTree implementation
- `Leaf<T, B, D>` - Leaf node
- `PseudoPRTreeNode<T, B, D>` - Internal node
- `PseudoPRTree<T, B, D>` - Pseudo tree structure
- Used during construction phase

### 5. `nodes.h` - PRTree Nodes
- Line 492-640: PRTree node implementations
- `PRTreeLeaf<T, B, D>` - Leaf node
- `PRTreeNode<T, B, D>` - Internal node
- `PRTreeElement<T, B, D>` - Tree element wrapper

### 6. `prtree_impl.h` - PRTree Implementation
- Line 642-end: Main `PRTree<T, B, D>` class
- Construction, query, insert, erase operations
- Serialization and persistence
- Dynamic updates and rebuilding

## Migration Strategy

1. **Phase 1** (Current): Document structure, create directory
2. **Phase 2**: Extract common types and utilities to `types.h`
3. **Phase 3**: Extract `BB` class to `bounding_box.h`
4. **Phase 4**: Extract data types to `data_type.h`
5. **Phase 5**: Extract pseudo tree to `pseudo_tree.h`
6. **Phase 6**: Extract nodes to `nodes.h`
7. **Phase 7**: Main PRTree remains in `prtree.h`, includes all detail headers

## Benefits of Modularization

1. **Faster Compilation**: Changes to one component don't require recompiling everything
2. **Better Organization**: Easier to locate and understand specific functionality
3. **Easier Maintenance**: Smaller, focused files are easier to review and modify
4. **Testing**: Can unit test individual components in isolation (future C++ tests)

## Dependencies Between Modules

```
prtree.h
  ├── types.h (no dependencies)
  ├── bounding_box.h (depends on: types.h)
  ├── data_type.h (depends on: types.h, bounding_box.h)
  ├── pseudo_tree.h (depends on: types.h, bounding_box.h, data_type.h)
  ├── nodes.h (depends on: types.h, bounding_box.h, data_type.h)
  └── prtree_impl.h (depends on: all above)
```

## Current Status

- ✅ Directory structure created
- ✅ Documentation written
- ⏳ Pending: Actual file splitting (future PR)

## Contributing

If you want to help with modularization:

1. Choose a module to extract (start with `types.h`)
2. Create the new header file with proper include guards
3. Move the relevant code from `prtree.h`
4. Update includes in `prtree.h`
5. Verify that all tests pass
6. Create a PR with the changes

For questions, see [ARCHITECTURE.md](../../../ARCHITECTURE.md).
