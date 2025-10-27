"""
Face recognition module for identifying specific people.
Uses face_recognition library for encoding and matching faces.
"""
import logging
from pathlib import Path
from typing import List, Dict, Set, Tuple
import pickle
import cv2
import numpy as np
import face_recognition
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class FaceRecognizer:
    """Face recognition for identifying people in videos."""
    
    def __init__(self, encodings_path: str = "/data/faces/encodings.pkl", tolerance: float = 0.6, num_frames: int = 10, model: str = 'hog', unknown_faces_path: str = "/data/faces/unknown"):
        """
        Initialize the face recognizer.
        
        Args:
            encodings_path: Path to saved face encodings database
            tolerance: Face matching tolerance (lower is more strict)
            num_frames: Number of frames to extract and analyze from each video
            model: Face detection model to use ('hog' or 'cnn')
            unknown_faces_path: Path to save unknown face images
        """
        self.encodings_path = Path(encodings_path)
        self.tolerance = tolerance
        self.num_frames = num_frames
        self.model = model
        self.unknown_faces_path = Path(unknown_faces_path)
        self.unknown_faces_path.mkdir(parents=True, exist_ok=True)
        self.known_faces = {}  # Dict mapping name -> list of encodings
        
        # Load existing encodings if available
        if self.encodings_path.exists():
            self.load_encodings()
        else:
            logger.info("No existing face encodings found")
    
    def load_encodings(self):
        """Load face encodings from disk."""
        try:
            with open(self.encodings_path, 'rb') as f:
                self.known_faces = pickle.load(f)
            logger.info(f"Loaded face encodings for {len(self.known_faces)} people")
        except Exception as e:
            logger.error(f"Failed to load face encodings: {e}", exc_info=True)
            self.known_faces = {}
    
    def save_encodings(self):
        """Save face encodings to disk."""
        try:
            self.encodings_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.encodings_path, 'wb') as f:
                pickle.dump(self.known_faces, f)
            logger.info(f"Saved face encodings for {len(self.known_faces)} people")
        except Exception as e:
            logger.error(f"Failed to save face encodings: {e}", exc_info=True)
    
    def add_person(self, name: str, image_paths: List[Path]):
        """
        Add a person to the face recognition database.
        
        Args:
            name: Name of the person
            image_paths: List of image paths containing the person's face
        """
        logger.info(f"Adding face encodings for {name} from {len(image_paths)} images")
        
        encodings = []
        for image_path in image_paths:
            try:
                # Load image
                image = face_recognition.load_image_file(str(image_path))
                
                # Find faces and encode
                face_locations = face_recognition.face_locations(image, model=self.model)
                face_encodings = face_recognition.face_encodings(image, face_locations)
                
                if face_encodings:
                    encodings.extend(face_encodings)
                    logger.debug(f"Found {len(face_encodings)} face(s) in {image_path.name}")
                else:
                    logger.warning(f"No faces found in {image_path.name}")
                    
            except Exception as e:
                logger.error(f"Failed to process {image_path.name}: {e}")
        
        if encodings:
            self.known_faces[name] = encodings
            self.save_encodings()
            logger.info(f"Added {len(encodings)} face encoding(s) for {name}")
        else:
            logger.warning(f"No valid face encodings found for {name}")
    
    def extract_frames(self, video_path: Path, num_frames: int = 10) -> List[np.ndarray]:
        """
        Extract frames from video for face recognition.
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract
            
        Returns:
            List of frame images as numpy arrays
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB for face_recognition
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
        
        cap.release()
        logger.debug(f"Extracted {len(frames)} frames from {video_path.name}")
        return frames
    
    def recognize_faces_in_frame(self, frame: np.ndarray, video_name: str = None, frame_idx: int = None, save_unknown: bool = True) -> Tuple[Set[str], int]:
        """
        Recognize faces in a single frame.
        
        Args:
            frame: Image frame as numpy array (RGB)
            video_name: Name of the video being processed
            frame_idx: Frame index in the video
            save_unknown: Whether to save unknown faces
            
        Returns:
            Tuple of (Set of recognized person names, number of unknown faces saved)
        """
        # Find faces
        face_locations = face_recognition.face_locations(frame, model=self.model)
        
        if not face_locations:
            return set(), 0
        
        # Encode faces
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        
        recognized_names = set()
        unknown_count = 0
        
        # Compare with known faces
        for idx, (face_encoding, face_location) in enumerate(zip(face_encodings, face_locations)):
            matched = False
            
            for name, known_encodings in self.known_faces.items():
                matches = face_recognition.compare_faces(
                    known_encodings, 
                    face_encoding, 
                    tolerance=self.tolerance
                )
                
                if True in matches:
                    recognized_names.add(name)
                    matched = True
                    break  # Found a match, move to next face
            
            # Save unknown face
            if not matched and save_unknown:
                unknown_count += self._save_unknown_face(
                    frame, face_location, face_encoding, 
                    video_name, frame_idx, idx
                )
        
        return recognized_names, unknown_count
    
    def recognize_faces_in_video(self, video_path: Path, num_frames: int = None) -> Set[str]:
        """
        Recognize all faces across the video.
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to analyze (defaults to instance setting)
            
        Returns:
            Set of all recognized person names
        """
        if num_frames is None:
            num_frames = self.num_frames
        
        if not self.known_faces:
            logger.info(f"No trained faces yet - all detected faces will be saved as unknown")
        
        logger.info(f"Analyzing faces in video: {video_path.name} ({num_frames} frames, {self.model} model)")
        
        # Extract frames
        frames = self.extract_frames(video_path, num_frames=num_frames)
        
        if not frames:
            logger.warning(f"No frames extracted from {video_path.name}")
            return set()
        
        # Recognize faces in each frame
        all_recognized = set()
        total_unknown_saved = 0
        
        for i, frame in enumerate(frames):
            recognized, unknown_count = self.recognize_faces_in_frame(
                frame, 
                video_name=video_path.name, 
                frame_idx=i,
                save_unknown=True
            )
            if recognized:
                logger.debug(f"Frame {i+1}/{len(frames)}: Recognized {recognized}")
                all_recognized.update(recognized)
            total_unknown_saved += unknown_count
        
        if all_recognized:
            logger.info(f"Recognized people in {video_path.name}: {all_recognized}")
        else:
            logger.info(f"No known faces recognized in {video_path.name}")
        
        if total_unknown_saved > 0:
            logger.info(f"Saved {total_unknown_saved} unknown face(s) from {video_path.name}")
        
        return all_recognized
    
    def get_primary_person(self, recognized_people: Set[str]) -> str:
        """
        Get the primary person from recognized set.
        
        Args:
            recognized_people: Set of recognized person names
            
        Returns:
            Primary person name or None if empty
        """
        if not recognized_people:
            return None
        
        # For now, just return the first person alphabetically
        # Could be enhanced to use frequency or priority
        return sorted(recognized_people)[0]
    
    def _save_unknown_face(self, frame: np.ndarray, face_location: tuple, face_encoding: np.ndarray, 
                          video_name: str, frame_idx: int, face_idx: int) -> int:
        """
        Save an unknown face image and metadata.
        
        Args:
            frame: Full frame image
            face_location: Tuple of (top, right, bottom, left)
            face_encoding: Face encoding vector
            video_name: Name of source video
            frame_idx: Frame index
            face_idx: Face index in frame
            
        Returns:
            1 if saved, 0 if skipped
        """
        try:
            top, right, bottom, left = face_location
            
            # Add padding around face
            padding = 20
            top = max(0, top - padding)
            left = max(0, left - padding)
            bottom = min(frame.shape[0], bottom + padding)
            right = min(frame.shape[1], right + padding)
            
            # Crop face
            face_image = frame[top:bottom, left:right]
            
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_base = Path(video_name).stem if video_name else "unknown"
            filename = f"{timestamp}_{video_base}_f{frame_idx}_face{face_idx}.jpg"
            
            # Save image
            image_path = self.unknown_faces_path / filename
            face_bgr = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(image_path), face_bgr)
            
            # Save metadata
            metadata = {
                "filename": filename,
                "video_name": video_name,
                "frame_idx": frame_idx,
                "face_idx": face_idx,
                "timestamp": timestamp,
                "encoding": face_encoding.tolist(),
                "location": face_location
            }
            
            metadata_path = self.unknown_faces_path / f"{filename}.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
            
            logger.debug(f"Saved unknown face: {filename}")
            return 1
            
        except Exception as e:
            logger.error(f"Failed to save unknown face: {e}")
            return 0
