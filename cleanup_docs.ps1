# CritterCatcherAI Documentation Cleanup Script
# Run this after reviewing PROJECT_STATUS.md

Write-Host "=" * 80
Write-Host "CritterCatcherAI Documentation Cleanup"
Write-Host "This will archive outdated documentation files"
Write-Host "=" * 80
Write-Host ""

# Create archive directory
$archiveDir = ".\docs_archive_$(Get-Date -Format 'yyyy-MM-dd')"
Write-Host "Creating archive directory: $archiveDir"
New-Item -ItemType Directory -Path $archiveDir -Force | Out-Null
Write-Host "✓ Archive directory created" -ForegroundColor Green
Write-Host ""

# Files to archive (completed work)
$completedWork = @(
    "COMPLETED_WORK_2025-12-03.md",
    "CONFIG_PERSISTENCE_FIX.md",
    "DEPLOYMENT_SUMMARY.md",
    "RING_AUTH_FIX_DEPLOYMENT.md",
    "WIDGET_FIX_SUMMARY.md",
    "YOLO_CATEGORIES_FIX.md",
    "BUG_FIX_LOG_AND_DOWNLOADS.md",
    "ASYNC_ISSUES_ANALYSIS.md"
)

# Files to archive (superseded status)
$superseded = @(
    "PLAN_STATUS_UPDATE.md",
    "Bugs.txt"
)

Write-Host "Archiving completed work documentation..." -ForegroundColor Cyan
foreach ($file in $completedWork) {
    if (Test-Path $file) {
        Move-Item $file $archiveDir
        Write-Host "  ✓ Moved $file" -ForegroundColor Green
    } else {
        Write-Host "  ⚠ Not found: $file" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "Archiving superseded status files..." -ForegroundColor Cyan
foreach ($file in $superseded) {
    if (Test-Path $file) {
        Move-Item $file $archiveDir
        Write-Host "  ✓ Moved $file" -ForegroundColor Green
    } else {
        Write-Host "  ⚠ Not found: $file" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "=" * 80
Write-Host "Summary" -ForegroundColor Cyan
Write-Host "=" * 80
Write-Host "✓ Archived files moved to: $archiveDir" -ForegroundColor Green
Write-Host "✓ Kept: README.md, USER_GUIDE.md, TECHNICAL_SPECIFICATION.md, CHANGELOG.md" -ForegroundColor Green
Write-Host "✓ Created: PROJECT_STATUS.md (single source of truth)" -ForegroundColor Green
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "  1. Review PROJECT_STATUS.md"
Write-Host "  2. Verify archived files in $archiveDir"
Write-Host "  3. Commit changes to git"
Write-Host "  4. Delete archive folder (or keep for reference)"
Write-Host ""
Write-Host "Plans (@plans in Warp):" -ForegroundColor Yellow
Write-Host "  - Keep plan 3cc142b8 (Next Phase Implementation) as active"
Write-Host "  - Keep plan 0cec23e2 (Optimized Implementation) for details"
Write-Host "  - Archive others (they're in git history)"
Write-Host ""
Write-Host "Done! 🎉" -ForegroundColor Green
