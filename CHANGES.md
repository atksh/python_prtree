# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.1] - 2025-11-08

### Added
- Modernized project structure for better OSS contribution experience (#52)
  - Added comprehensive CONTRIBUTING.md with development guidelines
  - Improved development tooling configuration
  - Enhanced documentation for contributors

### Fixed
- Fixed critical Windows crash with `std::mutex` (#51)
  - Changed to `std::unique_ptr<std::recursive_mutex>` for thread-safe, pybind11-compatible implementation
  - Resolved fatal crashes and deadlocks on Windows platform
- Optimized CI workflows for better reliability and performance (#51)

### Improved
- Expanded test coverage with comprehensive test suite (#48)
  - Now includes 674 unit tests covering edge cases and various scenarios
  - Improved test organization and structure
- Added `query_intersections()` method for efficient AABB pair detection (#47)
  - Enables finding all pairs of intersecting bounding boxes efficiently

### Changed
- Upgraded dependency versions in CI workflows (#50)
- Migrated to C++20 standard with concepts for type safety
- Enhanced error messages with context while maintaining backward compatibility
  - Example: `"Given index is not found. (Index: 999, tree size: 2)"`

### Performance
- Construction: 9-11M ops/sec (single-threaded)
- Memory: 23 bytes/element
- All 674 unit tests pass

## [0.7.0] - 2024-XX-XX

### Added
- Intersection bug fix
- Python 3.13 support

### Improvements
- Exception Safety: noexcept + RAII (no memory leaks)
- Thread Safety: Recursive mutex protects all mutable operations
