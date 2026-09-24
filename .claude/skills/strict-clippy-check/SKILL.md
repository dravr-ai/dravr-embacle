---
name: strict-clippy-check
description: Enforces zero-tolerance code quality policy using Clippy with strict lints, all warnings treated as errors
user-invocable: true
---

# Strict Clippy Check Skill

## Purpose
Enforces embacle's zero-tolerance code quality policy using Clippy with strict lints.

## Usage
Run this skill before every commit and after refactoring.

## Commands

### Standard Strict Check
```bash
CARGO_BUILD_WARNINGS=deny cargo clippy --all-targets
```

### Fix Auto-Fixable Issues
```bash
CARGO_BUILD_WARNINGS=deny cargo clippy --fix --all-targets --allow-dirty
```

## Linting Configuration

embacle uses `Cargo.toml` lints configuration:

```toml
[lints.clippy]
all = { level = "deny", priority = -1 }
pedantic = { level = "deny", priority = -1 }
nursery = { level = "deny", priority = -1 }
```

## Common Issues & Fixes

### Issue: `unwrap()` detected
```rust
// ❌ Bad
let value = some_option.unwrap();

// ✅ Good
let value = some_option.ok_or(RunnerError::internal("missing value"))?;
```

### Issue: Cast warnings
```rust
// ❌ Might truncate
let small = large_value as u8;

// ✅ Safe conversion with allow attribute
#[allow(clippy::cast_possible_truncation)]
let small = large_value as u8; // validated above
```

## Allowed Exceptions
- `#[allow(clippy::cast_*)]` for validated numeric casts
- `#[allow(clippy::struct_excessive_bools)]` when bools represent capabilities
- `unwrap()` in test code and static data

## Success Criteria
- ✅ Zero Clippy warnings
- ✅ All error handling uses `Result<T, RunnerError>`
- ✅ No `unwrap()` in production code (src/)
- ✅ No `anyhow::anyhow!()` anywhere
- ✅ Public APIs documented

## Related Files
- `Cargo.toml` - Lint configuration
- `.github/workflows/ci.yml` - CI clippy job
