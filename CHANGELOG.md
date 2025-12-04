# Changelog

All notable changes to CritterCatcherAI will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [v1.0.0] - 2025-12-04
### Stable Release - Baseline Before Multi-Instance Refactor

This release marks the stable baseline of CritterCatcherAI before beginning major architectural changes for multi-category and multi-instance detection support.

#### Added
- YOLO object detection with COCO classes for video classification
- Single-category video routing (highest confidence category)
- Face recognition with training interface for person identification
- CLIP Stage 2 refinement with configurable auto-approval thresholds
- Application logging to persistent file (/config/crittercatcher.log)
- Processing summary logs showing statistics at end of each run
- Status widget for real-time task tracking in web UI
- Ring camera integration for automated video download and processing
- Review workflow with confirm/reject functionality
- Animal profile management with YOLO + CLIP hybrid detection
- Tracked video generation with bounding boxes
- Docker deployment on Unraid with volume persistence

#### Fixed
- **Bug 1**: YOLO categories now persist correctly after container restart
- **Bug 2**: Face training error handling improved (no more misleading errors)
- **Bug 6**: Application logs accessible via Logs tab in web UI
- **Enhancement 11**: Processing summary logs show detailed statistics

#### Known Issues
These issues are documented and will be addressed in upcoming releases:
- **Bug 3 & 7**: Task blocking - Face extraction blocks other operations
- **Bug 4**: Single-category routing only (multi-instance support coming in v1.2.0)
- **Bug 5**: No dark mode or single-page application navigation
- **Bug 8**: Person video confirmation UI doesn't clear selected videos
- **Bug 12**: Invalid YOLO categories (e.g., "rabbit") appear in profile creation

#### Platform
- Docker container running on Unraid
- Python 3.11 with FastAPI backend
- YOLO v8 for object detection
- CLIP ViT for image classification
- Face recognition library for person identification

#### Deployment
```bash
# Pull this specific version
docker pull ghcr.io/ahazii/crittercatcherai:v1.0.0

# Or tag current image
docker tag crittercatcherai:latest crittercatcherai:v1.0.0
```

#### Next Release
**v1.1.0** - Phase 0: Foundation & Testing Infrastructure
- Download management (cleanup, return-to-downloads)
- Task concurrency fix (parallel operations)
- Expected: ~2 days of development

---

## [Unreleased]
### Phase 0: Foundation & Testing Infrastructure (In Development)
Coming in v1.1.0

### Planned for v1.2.0+
- Multi-category video routing (video appears in multiple review tabs)
- Multi-instance detection (identify 3 different cars in same video)
- Instance-level CLIP analysis (no cross-contamination)
- Universal auto-approval (any category, not just person)
- Dark mode + SPA navigation
- Color-coded instance visualization

[v1.0.0]: https://github.com/Ahazii/CritterCatcherAI/releases/tag/v1.0.0
