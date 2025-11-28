"""
Download Tracker Module

Tracks downloaded Ring videos to prevent duplicate downloads and maintain history.
"""

import sqlite3
import os
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DownloadTracker:
    """Manages download history using SQLite database."""
    
    def __init__(self, db_path="/data/download_history.db"):
        """
        Initialize download tracker.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Create database and tables if they don't exist."""
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS downloaded_videos (
                event_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                camera_name TEXT,
                download_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                file_path TEXT,
                status TEXT DEFAULT 'downloaded'
            )
        """)
        
        conn.commit()
        conn.close()
        
        logger.info(f"Download tracker initialized at {self.db_path}")
    
    def is_downloaded(self, event_id):
        """
        Check if a video has been downloaded.
        
        Args:
            event_id: Ring event ID
            
        Returns:
            bool: True if already downloaded
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT 1 FROM downloaded_videos WHERE event_id = ?",
            (event_id,)
        )
        
        result = cursor.fetchone() is not None
        conn.close()
        
        return result
    
    def mark_downloaded(self, event_id, filename, camera_name=None, file_path=None, status="downloaded"):
        """
        Mark a video as downloaded.
        
        Args:
            event_id: Ring event ID
            filename: Downloaded filename
            camera_name: Name of camera
            file_path: Full path to downloaded file
            status: Download status (default: 'downloaded')
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO downloaded_videos 
                (event_id, filename, camera_name, download_timestamp, file_path, status)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                event_id,
                filename,
                camera_name,
                datetime.utcnow().isoformat(),
                file_path,
                status
            ))
            
            conn.commit()
            logger.debug(f"Marked video as downloaded: {event_id} -> {filename}")
            
        except sqlite3.Error as e:
            logger.error(f"Failed to mark video as downloaded: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def update_status(self, event_id, new_status, new_file_path=None):
        """
        Update the status of a downloaded video.
        
        Args:
            event_id: Ring event ID
            new_status: New status (e.g., 'processed', 'reviewed')
            new_file_path: Updated file path (optional)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            if new_file_path:
                cursor.execute(
                    "UPDATE downloaded_videos SET status = ?, file_path = ? WHERE event_id = ?",
                    (new_status, new_file_path, event_id)
                )
            else:
                cursor.execute(
                    "UPDATE downloaded_videos SET status = ? WHERE event_id = ?",
                    (new_status, event_id)
                )
            
            conn.commit()
            logger.debug(f"Updated video status: {event_id} -> {new_status}")
            
        except sqlite3.Error as e:
            logger.error(f"Failed to update video status: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def get_unprocessed_videos(self):
        """
        Get list of downloaded but unprocessed videos.
        
        Returns:
            list: List of dicts with video info
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT event_id, filename, camera_name, file_path, download_timestamp
            FROM downloaded_videos
            WHERE status = 'downloaded'
            ORDER BY download_timestamp DESC
        """)
        
        videos = []
        for row in cursor.fetchall():
            videos.append({
                'event_id': row[0],
                'filename': row[1],
                'camera_name': row[2],
                'file_path': row[3],
                'download_timestamp': row[4]
            })
        
        conn.close()
        return videos
    
    def cleanup_deleted_entries(self, check_paths=None):
        """
        Remove database entries for videos that no longer exist on disk.
        
        Args:
            check_paths: List of directories to check (default: ['/data/downloads/', '/data/review/'])
        """
        if check_paths is None:
            check_paths = ['/data/downloads/', '/data/review/']
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT event_id, file_path FROM downloaded_videos WHERE file_path IS NOT NULL")
        entries = cursor.fetchall()
        
        deleted_count = 0
        for event_id, file_path in entries:
            if file_path and not os.path.exists(file_path):
                # Check if file exists in any of the check paths
                filename = os.path.basename(file_path)
                found = False
                
                for check_path in check_paths:
                    if os.path.exists(check_path):
                        for root, _, files in os.walk(check_path):
                            if filename in files:
                                # Update path
                                new_path = os.path.join(root, filename)
                                cursor.execute(
                                    "UPDATE downloaded_videos SET file_path = ? WHERE event_id = ?",
                                    (new_path, event_id)
                                )
                                found = True
                                break
                        if found:
                            break
                
                if not found:
                    # File truly deleted, remove entry
                    cursor.execute("DELETE FROM downloaded_videos WHERE event_id = ?", (event_id,))
                    deleted_count += 1
                    logger.debug(f"Removed deleted video entry: {event_id}")
        
        conn.commit()
        conn.close()
        
        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} deleted video entries from database")
        
        return deleted_count
    
    def get_stats(self):
        """
        Get download statistics.
        
        Returns:
            dict: Statistics (total_downloads, by_status, by_camera)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total downloads
        cursor.execute("SELECT COUNT(*) FROM downloaded_videos")
        total = cursor.fetchone()[0]
        
        # By status
        cursor.execute("SELECT status, COUNT(*) FROM downloaded_videos GROUP BY status")
        by_status = {row[0]: row[1] for row in cursor.fetchall()}
        
        # By camera
        cursor.execute("SELECT camera_name, COUNT(*) FROM downloaded_videos GROUP BY camera_name")
        by_camera = {row[0]: row[1] for row in cursor.fetchall()}
        
        conn.close()
        
        return {
            'total_downloads': total,
            'by_status': by_status,
            'by_camera': by_camera
        }
