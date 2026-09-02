# Git Policy

## Branching Strategy

| Branch | Purpose | Rules |
|--------|---------|-------|
| `main` | Production | NO direct commits. PRs only |
| `develop` | Integration | Merge feature branches here |
| `feature/*` | New features | Branch from develop |

## Commit Convention

```
type(scope): description
```

Types: feat, fix, docs, test, refactor, security, deps
