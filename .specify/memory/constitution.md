<!--
Sync Impact Report
- Version change: 1.1.0 → 1.1.1
- Modified principles: none
- Added sections: none
- Removed sections: none
- Updated sections: Environment Configuration Standards (development environment database)
- Templates requiring updates:
  - .specify/templates/plan-template.md: ✅ updated (version reference updated to 1.1.1)
  - .specify/templates/spec-template.md: ✅ aligned (no changes required)
  - .specify/templates/tasks-template.md: ✅ aligned (no changes required)
  - .specify/templates/commands/*: ⚠ not present in repository
- Follow-up TODOs:
  - TODO(RATIFICATION_DATE): Original adoption date unknown; set when known
-->

# Autodoc v2 Constitution

## Core Principles

### I. Code Quality and Readability (NON-NEGOTIABLE)
The codebase MUST be simple, explicit, and maintainable.

- Names MUST be meaningful and self-descriptive; avoid abbreviations.
- Lint and format checks MUST pass with zero warnings in CI.
- Public interfaces MUST be documented with concise docstrings and examples.
- Error handling MUST be explicit; no silent catch-and-ignore. Log at source.
- Complexity MUST be controlled: functions ≤ 40 lines or justified; cyclomatic
  complexity targets enforced by linter rules where available.
- Dead code and unused dependencies MUST be removed in the same PR.
- Security-sensitive logic (auth, permissions, data access) MUST have code
  owners and mandatory review.

Rationale: High readability and simple designs reduce defects and speed up
changes across thousands of repositories.

### II. Testing Discipline and Standards
Tests define behavior and guard against regressions. TDD is the default.

- Tests MUST be written first; they MUST fail before implementation (Red →
  Green → Refactor).
- Contract tests MUST exist for all external integrations (GitHub, Bitbucket,
  vector stores, LLM providers). Schemas are versioned.
- Required coverage: ≥ 85% lines and ≥ 85% branches per package; critical
  core modules identified in the plan MUST reach ≥ 95% line coverage.
- Flaky tests are NOT allowed. Any flaky test MUST be fixed or reverted in the
  same day; quarantining is prohibited.
- Integration tests MUST cover primary chat flows, repository sync, and
  permission checks. E2E smoke MUST run on every PR.
- Tests MUST be deterministic: network and time are mocked unless the test is
  an explicit contract or E2E test.

Rationale: Deterministic, comprehensive tests are essential for safety in a
fast-moving, research-heavy product.

### III. UX Consistency and Accessibility
Users interact through a conversational UI and documentation views that MUST be
consistent, accessible, and predictable.

- A single design system MUST be used for all surfaces. Components are reused
  and themable (light/dark, high-contrast) with no one-off styles.
- Accessibility MUST meet WCAG 2.2 AA. Keyboard-first navigation and screen
  reader labels are mandatory. Color contrast budgets enforced.
- Conversational UX MUST support: streaming tokens, partial-cancel, retry,
  copy-with-citation, and “show sources” affordances.
- Empty, loading, and error states MUST be designed and implemented.
- Content MUST include provenance (repo, path, commit SHA) when answers
  reference code or docs.

Rationale: Consistency and accessibility ensure trust and speed for developers
across diverse environments.

### IV. Performance and Reliability
The system MUST be fast, observable, and resilient.

- UI interaction targets: p50 ≤ 500 ms, p95 ≤ 1500 ms. First token for chat
  responses ≤ 1500 ms at p95; streaming MUST begin before 1500 ms.
- Backend SLO: < 1% 5xx over 30 days; error budget policy applies.
- Timeouts, retries with jitter, and circuit breakers MUST be used for all
  remote calls. Idempotency keys for retried writes.
- Caching MUST be applied where safe (per-repo metadata, embeddings, rendered
  docs). Stale-while-revalidate patterns preferred.
- Observability MUST include: distributed tracing, RED/USE metrics, structured
  logs with correlation IDs. Dashboards and alerts are required for SLOs.
- Performance regressions >10% in latency or cost MUST block merges unless an
  approved exception is documented.

Rationale: Responsiveness and reliability are core to conversational discovery
and trust in answers.

### V. Data Freshness and Provenance Integrity
Documentation MUST be current and traceable to source-of-truth commits.

- Freshness target: 95% of default-branch updates reflected within 10 minutes;
  hard cap 30 minutes. Manual reindex available per repo.
- Every answer that cites repository content MUST include source links with
  commit SHA and last-updated timestamp.
- If an answer cannot be supported by sources, the UI MUST clearly indicate
  uncertainty and invite follow-up.
- Privacy and permissions MUST be enforced end-to-end. Private repos are never
  surfaced to unauthorized users; least-privilege OAuth scopes used.
- Indexing and retrieval MUST be version-aware (branch, tag, commit). Users can
  switch the referenced revision.

Rationale: Fresh, attributable answers prevent hallucinations and strengthen
developer confidence.

## Non-Functional Requirements & Benchmarks

- Accessibility: WCAG 2.2 AA compliance verified in CI (axe-core or
  equivalent). Contrast and keyboard tests included.
- Browser Support: Evergreen Chrome, Firefox, Safari, Edge (last 2 versions).
- Latency Budgets: As defined in Principle IV; p95 budgets MUST be met before
  release. Budgets recorded per feature plan.
- Cost Budgets: LLM and embedding spend per request MUST be tracked; changes
  >15% require review.
- Security: Threat model for integrations; OAuth scopes minimized; tokens kept
  out of logs; PII handling documented.
- Data Retention: Conversation logs redacted of secrets by default; opt-in
  retention with explicit user consent.

## Technology Stack Requirements

The following technology stack is MANDATORY for all components:

- **Backend Framework**: FastAPI MUST be used for all REST API endpoints and web
  services. OpenAPI schema generation and automatic validation are required.
- **AI/LLM Framework**: LangChain MUST be used for LLM integrations, prompt
  management, and chain orchestration. LangGraph MUST be used for complex
  multi-step AI workflows and agent-based processing.
- **Data Validation**: Pydantic MUST be used for all data models, request/response
  schemas, and configuration management. Type hints are mandatory.
- **Async Processing**: All I/O operations MUST use Python's async/await patterns.
  Blocking operations in async contexts are prohibited.

Rationale: Standardized stack ensures consistency, leverages strong typing and
validation, and provides robust AI/LLM integration capabilities.

## Environment Configuration Standards

Development and production environments MUST follow these configurations:

### Development Environment
- **Repository Storage**: Local filesystem MUST be used for cloning and storing
  repositories. Path: `./data/repos/` with organization/project structure.
- **Primary Database**: Local MongoDB MUST be used for all persistent data and
  vector storage. Connection string: `mongodb://localhost:27017/autodoc_dev`.
- **Configuration**: Environment variables via `.env` file; sensitive values
  MUST NOT be committed to version control.

### Production Environment
- **Repository Storage**: AWS S3 MUST be used for cloning and storing repositories.
  Bucket structure: `{bucket}/repos/{org}/{project}/{branch}/`.
- **Primary Database**: MongoDB MUST be used for all persistent data (metadata,
  analysis results, user data) and vector storage. Connection pooling and replica sets required.
- **Configuration**: AWS Systems Manager Parameter Store or equivalent for
  configuration management. Secrets rotation policies required.

### Cross-Environment Requirements
- **Migrations**: Database schema changes MUST be versioned and backward-compatible.
- **Monitoring**: Both environments MUST have health checks, metrics collection,
  and alerting configured.
- **Backup**: Production data MUST be backed up daily with tested recovery procedures.

## Development Workflow & Quality Gates

- Branches follow: `<ticket>-<short-description>`.
- PRs MUST include: tests, updated docs (user-facing or ops), and a short
  performance note (expected impact vs. baseline, even if none).
- CI Gates (all required): lint/format, unit, integration, E2E smoke, coverage
  thresholds, a11y checks, contract tests, basic benchmark suite.
- Two approvals required; code owners for security- or privacy-sensitive areas.
- Feature flags for risky changes; safe rollout and quick rollback plans.
- Release notes and changelog entries required for user-visible changes.

## Governance

- This Constitution supersedes conflicting practices. Violations MUST be
  documented with a temporary exception, owner, and expiry date.
- Amendments occur via PR to this file with an impact analysis (principles
  affected, risks, migration plan). Major changes require a migration plan.
- Versioning follows SemVer for this document:
  - MAJOR: Backward-incompatible redefinitions or removals of principles.
  - MINOR: New principle/section or material expansion of guidance.
  - PATCH: Clarifications and wording that do not change meaning.
- Compliance reviews:
  - Per-PR: Constitution Check in plans and CI gates MUST pass.
  - Quarterly: Audit selected features for adherence to principles and SLOs.

**Version**: 1.1.1 | **Ratified**: TODO(RATIFICATION_DATE) | **Last Amended**: 2025-09-22