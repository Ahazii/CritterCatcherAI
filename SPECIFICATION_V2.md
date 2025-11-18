# **CritterCatcherAI - Simplified Specification v2.0**

---

## **1. SYSTEM OVERVIEW**

**Purpose**: Automatically record and organize specific target animals from Ring camera footage using two-stage AI identification.

**Core Process**:
1. User defines target animals (hedgehogs, birds, etc.)
2. YOLO Stage 1 filters videos by broad categories
3. CLIP/ViT Stage 2 accurately identifies target animal
4. High-confidence matches auto-sorted, others sent for review
5. User confirmations improve model accuracy

---

## **2. ARCHITECTURE**

### **2.1 Two-Stage Identification**

**Stage 1: YOLO Detection**
- Detects broad animal categories (cat, dog, bird, etc.)
- User selects which YOLO categories might contain their target animal
- Extracts frames from videos containing selected categories

**Stage 2: CLIP/ViT Classification**
- Text-based image classifier using CLIP or Vision Transformer
- Input: Frame + text description of target animal
- Output: Confidence score (0-1) that frame contains target animal
- **Recommendation**: Use OpenAI CLIP (pre-trained, no retraining required initially)
  - Can be fine-tuned later with user confirmations when needed
  - Works with arbitrary text descriptions
  - Production-ready, reliable accuracy

### **2.2 Text Description Strategy (BOTH Approaches)**

**Default Approach: Simple Animal Name**
- Auto-generated from animal profile name
- Example: User creates "hedgehog" → Default description: "a hedgehog"
- Used immediately when profile created
- Works well with CLIP out-of-the-box

**Custom Approach: User-Defined Description**
- User can edit/customize the description anytime
- Example: "a small spiky animal with four legs and brown fur"
- Useful for improving accuracy if model performance is poor
- Can be changed without retraining

---

## **3. USER INTERFACE - ANIMAL PROFILES**

### **3.1 Create Animal Profile**

User provides:

1. **Animal Name** (text field)
   - Example: "hedgehog", "golden eagle"
   - Required field
   
2. **YOLO Category Selection** (checkbox list with descriptions)
   - Display: Curated subset of YOLO COCO categories
   - Format: Checkbox + Category name + Description
   - Example:
     ```
     ☑ cat - Small carnivorous mammal
     ☐ dog - Canine domestic animal
     ☑ sheep - Woolly livestock
     ☑ cow - Large bovine animal
     ☐ bird - Feathered flying animal
     ```
   - User selects one or more categories
   - Minimum 1 category required

3. **Text Description for CLIP/ViT** (text area with tooltip)
   - Default: Auto-populated with animal name (e.g., "a hedgehog")
   - User can customize: "a small hedgehog with brown spikes"
   - Examples provided in tooltip:
     - "a small spiky animal"
     - "a bird with red wings"
     - "a four-legged mammal"
   - This field is BOTH auto-generated AND user-editable
   - Can be changed anytime without retraining

4. **Auto-Approval Confidence Threshold** (slider + number input)
   - Range: 0.5 - 0.99
   - Default: 0.80
   - Meaning: Frames scoring ≥ threshold auto-sorted to `/data/sorted/`
   - Below threshold → `/data/review/`
   - User can toggle: "Require manual confirmation for all matches"

5. **Model Confidence Tracking** (read-only display)
   - Current model accuracy (% of user confirmations that were correct)
   - Example: "Current Model: 78% accuracy (14/18 confirmations correct)"
   - Recommendation message: "Model is 78% confident. Recommend retraining after 50 confirmations."
   - User can manually trigger retraining

### **3.2 Edit Animal Profile**

User can modify:
- Animal name (not recommended after creation)
- YOLO category selection
- Text description for CLIP/ViT
- Confidence threshold
- Auto-approval toggle
- Manual retraining trigger

---

## **4. PROCESSING WORKFLOW**

### **4.1 Processing Trigger**

Reuse existing scheduler system:
- **Timer**: User-configurable interval (minutes/hours)
- **"Process Now" Button**: Force immediate processing
- Respects existing scheduler enable/disable config
- Applies to all enabled animal profiles

### **4.2 Processing Pipeline**

**Per processing run**:

1. **Download Ring Videos**
   - Fetch recent videos from Ring camera
   - Check which animal profiles are enabled

2. **For Each Enabled Animal Profile**:
   a. Extract frames from videos
   b. Run YOLO detection
   c. Filter frames to selected YOLO categories only
   d. For matching frames, run CLIP/ViT with text description (default or custom)
   e. Score each frame (confidence 0-1)
   f. Sort by confidence:
      - ≥ Threshold AND Auto-Approval Enabled → Move to `/data/sorted/<animal_name>/`
      - < Threshold OR Manual Confirmation Required → Move to `/data/review/<animal_name>/`

3. **Log Results**
   - Frames sorted count per animal
   - Frames pending review count per animal
   - Model accuracy for each animal
   - Processing timestamp

---

## **5. DIRECTORY STRUCTURE**

```
/data/
├── downloads/                      # Ring camera downloads
├── sorted/
│   ├── hedgehog/                  # Auto-approved videos
│   │   ├── video_001.mp4
│   │   ├── video_002.mp4
│   │   └── ...
│   └── golden_eagle/
│       └── ...
├── review/
│   ├── hedgehog/                  # Pending user review
│   │   ├── frame_001.jpg
│   │   ├── frame_001.jpg.json     # Metadata (confidence, timestamp, CLIP score)
│   │   ├── frame_002.jpg
│   │   ├── frame_002.jpg.json
│   │   └── ...
│   └── golden_eagle/
│       └── ...
├── training/
│   ├── hedgehog/                  # Training data for retraining
│   │   ├── confirmed/             # User confirmed as correct
│   │   │   ├── frame_001.jpg
│   │   │   └── ...
│   │   └── rejected/              # User confirmed as incorrect (for negative examples)
│   │       └── ...
│   └── golden_eagle/
│       └── ...
└── models/
    ├── hedgehog/
    │   ├── clip_vit_finetuned.pt  # Fine-tuned model (if retraining done)
    │   └── config.json             # Model metadata
    └── golden_eagle/
        └── ...
```

---

## **6. REVIEW & FEEDBACK UI**

### **6.1 Review Tab Layout**

**Tab Navigation**:
- Tabbed interface showing: `hedgehog (12 pending)`, `golden_eagle (5 pending)`, etc.
- Each tab shows pending images for that animal

**Per Animal Review Screen**:

1. **Image Gallery** (grid layout)
   - Display pending images from `/data/review/<animal>/`
   - Each image shows:
     - Thumbnail (clickable)
     - Confidence score from CLIP/ViT (e.g., "92% confidence")
     - Timestamp (from metadata)
   - Lazy loading for performance

2. **Image Selection Controls**
   - **Single Click**: Select one image (add border/highlight)
   - **Ctrl+Click**: Toggle selection (add/remove to multi-select)
   - **Drag Selection**: Click and drag to select multiple images in grid
   - **"Select All" Button**: Select all images in current folder
   - **"Deselect All" Button**: Deselect all images
   - **Display Counter**: "X of Y selected" (e.g., "5 of 12 selected")

3. **Bulk Actions** (appear when images selected)
   - **✓ Confirm Selected** Button
     - Moves images to `/data/training/<animal>/confirmed/`
     - Updates model confidence tracking (adds to numerator)
     - Removes from review
     - Refreshes gallery
     - Shows: "5 images confirmed. Model accuracy: 80% (16/20)"
   
   - **✗ Reject Selected** Button
     - Deletes images permanently
     - Updates model confidence tracking (adds to denominator)
     - Removes from review
     - Refreshes gallery
     - Shows: "3 images rejected. Model accuracy: 77% (17/22)"

4. **Sorting/Filtering** (optional)
   - Sort by: Confidence (high→low), Date (newest first)
   - Filter by: Confidence range slider (e.g., show only 70-90% confidence)

---

## **7. MODEL MANAGEMENT**

### **7.1 Model Confidence Tracking**

- Calculate from user confirmations:
  - Model Accuracy % = Correct confirmations / Total confirmations
  - Example: User confirmed 15 images correct, 5 incorrect → 75% accuracy (15/20)
  - Displayed as: "Current Model: 75% accuracy (15/20 confirmations)"

### **7.2 Retraining Recommendation System**

**User-Defined Threshold + Automatic Recommendation**:

1. **User Sets Target Accuracy** (settings)
   - Default: 85%
   - User adjustable per animal
   - Example: "Retrain when accuracy drops below 80%"

2. **Automatic Recommendation** (based on count)
   - Default: "Consider retraining after 50 confirmations"
   - User can adjust: 30, 50, 100, etc.
   - Displayed in model tracking: "Recommend retraining after 50 confirmations. Current: 15 confirmations."

3. **Recommendation Message** (appears when triggered)
   - When accuracy < target: "Model accuracy is 72% (below your 85% target). Recommend retraining."
   - When count reached: "Reached 50 confirmations. Recommend retraining to improve accuracy."
   - Display includes "Retrain Now" button

### **7.3 Retraining Process**

**Manual Trigger Only** (user clicks "Retrain Now" or "Retrain Model"):
- Load pre-trained CLIP/ViT
- Fine-tune on `/data/training/<animal>/confirmed/` images
- Use negative examples from `/data/training/<animal>/rejected/` if available
- Save fine-tuned model to `/data/models/<animal>/`
- Display: "Retraining... Processing confirmed examples..."
- After complete: "Model retrained. Accuracy baseline reset. Processing next batch with new model..."
- Apply fine-tuned model to next processing run

### **7.4 Auto-Approval Toggle (Per Animal)**

- Checkbox: "Require manual confirmation for all matches"
- When checked:
  - ALL frames go to `/data/review/<animal>/` (regardless of confidence)
  - Confidence threshold ignored
  - Useful during model training phase to validate accuracy
- When unchecked:
  - Use confidence threshold normally
  - High confidence auto-sorted

---

## **8. CONFIGURATION & SETTINGS**

### **8.1 Animal Profile Management**

**List View**:
- Show all profiles: Name, YOLO categories count, pending reviews count, model accuracy
- Example:
  ```
  hedgehog        | Categories: 3 (cat, dog, sheep) | Pending: 12 | Accuracy: 80%
  golden_eagle    | Categories: 2 (bird, eagle)     | Pending: 5  | Accuracy: 92%
  ```

**Per Profile Actions**:
- Enable/Disable toggle (on/off)
- Edit button (modify settings)
- View pending button (go to review tab)
- Delete button (with confirmation)

**Profile Edit Dialog**:
- Edit: Animal name, YOLO categories, text description, confidence threshold
- View: Current accuracy, confirmation count, recommendation
- Buttons: Save, Cancel, Delete (with confirmation)

### **8.2 Scheduler Settings** (Existing)

- Reuse existing interval timer configuration
- "Process Now" button applies to all enabled animals
- Timer settings: Minutes/Hours interval
- Enable/Disable scheduler globally

---

## **9. DATA FLOW DIAGRAM**

```
Ring Camera Videos
       ↓
   [For Each Enabled Animal Profile]
       ↓
   YOLO Stage 1
   (Detect selected categories)
       ↓
   Extract Frames
       ↓
   CLIP/ViT Stage 2
   (Use default or custom text description)
   (Identify target animal)
       ↓
   Split by Confidence
   ├─→ High Confidence + Auto-Approved Enabled
   │   → /data/sorted/<animal>/
   │
   └─→ Low Confidence OR Manual Confirmation Required
       → /data/review/<animal>/
       ↓
   User Reviews Individual Images
   (Single click, Ctrl+click, Drag, Select All, Deselect All)
       ↓
   Bulk Confirm or Reject
   ├─→ Confirm → /data/training/<animal>/confirmed/
   │            Update model accuracy (correct)
   │
   └─→ Reject → Delete + /data/training/<animal>/rejected/
               Update model accuracy (incorrect)
       ↓
   Model Accuracy Recalculated
   (Correct confirmations / Total confirmations)
       ↓
   [If Recommendation Triggered] 
   User Reviews Retraining Recommendation
       ├─→ Trigger Manual Retraining
       │   (Fine-tune CLIP/ViT with confirmed + rejected data)
       │   Save fine-tuned model
       │
       └─→ Dismiss Recommendation
           Continue with current model
           ↓
       [Next Processing Run Uses Updated Model]
```

---

## **10. KEY FEATURES**

✓ Two-stage identification (YOLO → CLIP/ViT)  
✓ Per-animal profiles with separate models and folders  
✓ Ring camera integration (timer-based + on-demand)  
✓ **Both default and custom text descriptions for CLIP/ViT**  
✓ Individual image confirmation with multi-select  
✓ Multi-select via: Click, Ctrl+Click, Drag, Select All, Deselect All  
✓ User-defined confidence thresholds per animal  
✓ Per-animal auto-approval toggle with manual confirmation option  
✓ Model accuracy tracking with retraining recommendations  
✓ User-defined accuracy targets + confirmation count recommendations  
✓ Simplified UI (removed taxonomy/discovery mode)  
✓ Optional model fine-tuning with user confirmations  
✓ Negative training examples for improved model accuracy  

---

## **11. TECHNOLOGY STACK**

- **Stage 1 YOLO**: Existing YOLO detection (keep as-is)
- **Stage 2 AI**: OpenAI CLIP (pre-trained) or Vision Transformer
  - Installation: `pip install open-clip-torch`
  - Fine-tuning: Transfer learning with user confirmations + negative examples
- **Backend**: FastAPI (existing, add new endpoints)
- **Frontend**: HTML/JavaScript (existing, add new UI)
- **Storage**: File system (`/data/`)
- **Scheduler**: Existing scheduler system (reuse)

---

## **12. NEW BACKEND ENDPOINTS (To Implement)**

- `POST /api/animal-profiles` - Create new animal profile
- `GET /api/animal-profiles` - List all profiles
- `GET /api/animal-profiles/<id>` - Get profile details
- `PUT /api/animal-profiles/<id>` - Update profile
- `DELETE /api/animal-profiles/<id>` - Delete profile
- `POST /api/animal-profiles/<id>/enable` - Enable profile
- `POST /api/animal-profiles/<id>/disable` - Disable profile
- `GET /api/animal-profiles/<id>/pending-reviews` - Get pending images
- `POST /api/animal-profiles/<id>/confirm-images` - Bulk confirm
- `POST /api/animal-profiles/<id>/reject-images` - Bulk reject
- `POST /api/animal-profiles/<id>/retrain` - Trigger retraining
- `GET /api/animal-profiles/<id>/model-stats` - Get model accuracy/recommendations

---

## **13. NEW FRONTEND PAGES/SECTIONS**

- **Animal Profiles Tab**: Create, Edit, Delete profiles
- **Processing Control**: Timer settings + "Process Now" button
- **Review Tab**: Individual image review per animal, multi-select, confirm/reject

---

## **14. MIGRATION PLAN**

This is a complete simplification from v1.0:
- Remove: Discovery Mode, Taxonomy tree complexity, detected objects UI
- Keep: Ring integration, Scheduler system, Video sorting
- Add: Animal profiles, CLIP/ViT integration, Review UI with multi-select

---

**READY FOR IMPLEMENTATION**
