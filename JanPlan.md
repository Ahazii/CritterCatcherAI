JanPlan - Spec Alignment Remediation

Date: 2026-01-20
Project: CritterCatcherAI

Goal
Align code, API, and deployment behavior with the documented specification in
TECHNICAL_SPECIFICATION.md and README.md.

Scope
- API endpoints, response shapes, and status semantics
- Configuration hierarchy and environment overrides
- Processing pipeline behaviors (download, face recognition, tracking)
- SSE/real-time update claims
- Docker build/runtime documentation and images
- Logging behavior and documentation
- Test documentation accuracy

Assumptions
- The current spec is authoritative and code should conform to it
- Backward compatibility should be preserved where possible
- Changes must remain Docker-first and container-safe

Plan
1) API contract alignment
   - Add alias routes to match spec:
     - POST /api/config/save -> current /api/config handler
     - POST /api/ring/authenticate -> current /api/ring/auth handler
     - POST /api/review/confirm -> map to category-specific confirm
     - POST /api/review/reject -> map to category-specific reject
   - Normalize response shapes for /api/status, /api/config,
     /api/yolo-categories, /api/review/* to match spec examples.
   - Decide on deprecation approach for newer endpoints and document it.

2) Configuration hierarchy compliance
   - Ensure all runtime config reads use /config/config.yaml.
   - Validate environment variable overrides are documented and consistent.

3) Download semantics
   - Respect ring.download_hours unless explicitly "download all" is requested.
   - Honor ring.download_limit when set.
   - Ensure behavior is reflected in docs and UI.

4) Face recognition trigger
   - Run when "person" detected and face recognition enabled.
   - Remove or document any extra gating (e.g., profile requirement).

5) Tracking behavior
   - Restore YOLO tracking with persistent IDs if feasible.
   - Add automatic fallback to detection-only when tracking fails.
   - Update spec if tracking cannot be restored reliably.

6) SSE / real-time updates
   - Implement a working SSE endpoint for logs or status updates,
     or revise documentation to describe polling-only behavior.

7) Versioning
   - Make /api/status and /api/version report the same version.
   - Ensure version is sourced from version.txt or build metadata.

8) Docker and deployment docs
   - Document CUDA base image and GPU dependency, or provide a CPU image.
   - Document build args (PUID/PGID) and image tags.
   - Ensure docker-compose examples match actual files.

9) Logging behavior
   - Default to stdout/stderr; make file logging optional.
   - Document log locations and rotation policy.

10) Testing documentation
   - Update docs to reflect existing tests and how to run them.
   - Add container-based test guidance where feasible.

Deliverables
- Code updates in src/ to match API and behavior
- Updated TECHNICAL_SPECIFICATION.md and README.md
- Optional additional docs for deployment variants (CPU vs GPU)

Validation
- API verification via curl or browser
- Manual processing flow in Docker container
- Review folder and sorted folder outputs match spec
- Tracked videos play in browser
