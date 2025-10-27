"""
Ring video downloader module.
Downloads videos from Ring doorbells using the ring-doorbell library.
"""
import os
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional
import json

from ring_doorbell import Ring, Auth
from ring_doorbell.exceptions import Requires2FAError
from oauthlib.oauth2 import MissingTokenError


logger = logging.getLogger(__name__)


class RingDownloader:
    """Handles downloading videos from Ring devices."""
    
    def __init__(self, download_path: str, token_file: str = "/data/tokens/ring_token.json"):
        """
        Initialize the Ring downloader.
        
        Args:
            download_path: Path where videos will be downloaded
            token_file: Path to store/load refresh token
        """
        self.download_path = Path(download_path)
        self.download_path.mkdir(parents=True, exist_ok=True)
        self.token_file = Path(token_file)
        self.ring = None
        self.auth = None
        
        logger.info(f"Initialized RingDownloader with download path: {download_path}")
    
    def authenticate(self, username: Optional[str] = None, password: Optional[str] = None) -> bool:
        """
        Authenticate with Ring service using refresh token or credentials.
        
        Args:
            username: Ring account username (email)
            password: Ring account password
            
        Returns:
            bool: True if authentication successful
        """
        try:
            # Try to load existing token
            if self.token_file.exists():
                logger.info("Loading existing Ring refresh token")
                with open(self.token_file, 'r') as f:
                    token_data = json.load(f)
                    self.auth = Auth("CritterCatcherAI/1.0", token_data)
                    self.ring = Ring(self.auth)
                    self.ring.update_data()
                    logger.info("Successfully authenticated with existing token")
                    return True
            
            # If no token exists, authenticate with credentials
            if username and password:
                logger.info(f"Authenticating with Ring account: {username}")
                self.auth = Auth("CritterCatcherAI/1.0")
                
                try:
                    # Try to authenticate - this may require 2FA
                    # The ring-doorbell library uses oauthlib which supports 2FA callback
                    self.auth.fetch_token(username, password)
                    
                    # Save token for future use
                    # Access token from internal OAuth session
                    token_data = self.auth._oauth.token
                    self.token_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(self.token_file, 'w') as f:
                        json.dump(token_data, f)
                    logger.info("Saved refresh token for future use")
                    
                    self.ring = Ring(self.auth)
                    self.ring.update_data()
                    logger.info("Successfully authenticated with credentials")
                    return True
                    
                except Exception as auth_error:
                    # Don't log verbose 2FA instructions - web GUI handles this
                    logger.error(f"Authentication failed: {auth_error}")
                    raise
            
            logger.error("No valid authentication method available")
            return False
            
        except Requires2FAError:
            # Let 2FA error bubble up to webapp for handling
            logger.info("2FA required - passing to web interface")
            raise
        except MissingTokenError:
            logger.error("Token expired or invalid, re-authentication required")
            if self.token_file.exists():
                self.token_file.unlink()
            return False
        except Exception as e:
            logger.error(f"Authentication failed: {e}", exc_info=True)
            return False
    
    def authenticate_with_2fa(self, username: str, password: str, code_2fa: str) -> bool:
        """
        Authenticate with Ring using 2FA code.
        
        Args:
            username: Ring account username (email)
            password: Ring account password
            code_2fa: 2FA verification code
            
        Returns:
            bool: True if authentication successful
        """
        try:
            logger.info(f"Authenticating with Ring and 2FA code: {username}")
            self.auth = Auth("CritterCatcherAI/1.0")
            
            # Fetch token with 2FA code
            self.auth.fetch_token(username, password, code_2fa)
            
            # Save token for future use
            # Access token from internal OAuth session
            token_data = self.auth._oauth.token
            self.token_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.token_file, 'w') as f:
                json.dump(token_data, f)
            logger.info("Saved refresh token for future use")
            
            self.ring = Ring(self.auth)
            self.ring.update_data()
            logger.info("Successfully authenticated with 2FA")
            return True
            
        except Exception as e:
            logger.error(f"2FA authentication failed: {e}", exc_info=True)
            return False
    
    def get_devices(self) -> List:
        """
        Get list of Ring devices.
        
        Returns:
            List of Ring devices
        """
        if not self.ring:
            logger.error("Not authenticated with Ring")
            return []
        
        try:
            # Refresh device data from Ring API
            logger.debug("Refreshing Ring device data")
            self.ring.update_data()
            
            devices = []
            
            # The ring-doorbell library uses video_devices() or devices() method
            logger.debug("Getting devices from Ring API")
            
            # Try video_devices property first (newer API)
            if hasattr(self.ring, 'video_devices') and callable(self.ring.video_devices):
                logger.debug("Using video_devices() method")
                devices = self.ring.video_devices()
                logger.debug(f"video_devices() returned: {type(devices)}")
            elif hasattr(self.ring, 'devices') and callable(self.ring.devices):
                logger.debug("Using devices() method")
                devices = self.ring.devices()
                logger.debug(f"devices() returned: {type(devices)}")
            elif hasattr(self.ring, 'get_device_list'):
                logger.debug("Using get_device_list() method")
                devices = self.ring.get_device_list()
                logger.debug(f"get_device_list() returned: {type(devices)}")
            else:
                logger.error("No device accessor method found!")
                logger.debug(f"Ring object attributes: {[attr for attr in dir(self.ring) if not attr.startswith('_')]}")
            
            # Convert to list if needed
            if not isinstance(devices, list):
                try:
                    devices = list(devices)
                    logger.debug(f"Converted devices to list: {len(devices)} items")
                except:
                    logger.error(f"Could not convert devices to list, type: {type(devices)}")
                    devices = []
            
            logger.info(f"Found {len(devices)} Ring devices total")
            
            return devices
        except Exception as e:
            logger.error(f"Error getting Ring devices: {e}", exc_info=True)
            return []
    
    def download_recent_videos(self, hours: int = 24, limit: Optional[int] = None, device_filter: Optional[List[str]] = None) -> List[Path]:
        """
        Download videos from the last N hours.
        
        Args:
            hours: Number of hours to look back
            limit: Maximum number of videos to download
            device_filter: Optional list of device names to filter (None = all devices)
            
        Returns:
            List of downloaded video file paths
        """
        if not self.ring:
            logger.error("Not authenticated with Ring")
            return []
        
        downloaded_files = []
        devices = self.get_devices()
        
        # Filter devices if specified
        if device_filter:
            devices = [d for d in devices if d.name in device_filter]
            logger.info(f"Filtering to {len(devices)} devices: {device_filter}")
        
        for device in devices:
            logger.info(f"Checking videos for device: {device.name}")
            
            try:
                # Get video history
                # Don't filter by kind - get all events (motion, ding, etc.)
                history = device.history(limit=limit or 100)
                logger.info(f"Found {len(history)} events in history for {device.name}")
                
                # Filter by time
                cutoff_time = datetime.now() - timedelta(hours=hours)
                logger.info(f"Filtering for videos after {cutoff_time}")
                
                for event in history:
                    # Parse the ISO format timestamp from Ring API
                    # Example: "2025-10-25T18:23:19.683Z"
                    created_at_str = event.get('created_at')
                    if isinstance(created_at_str, str):
                        # Parse ISO format string
                        event_time = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
                        # Convert to local time (naive datetime for comparison)
                        event_time = event_time.replace(tzinfo=None)
                    else:
                        # Fallback: assume it's a timestamp
                        event_time = datetime.fromtimestamp(created_at_str)
                    
                    logger.debug(f"Event {event['id']} created at {event_time}")
                    
                    if event_time < cutoff_time:
                        logger.debug(f"Skipping event {event['id']} - too old")
                        continue
                    
                    # Download video
                    video_id = event['id']
                    video_url = device.recording_url(video_id)
                    
                    if not video_url:
                        logger.warning(f"No video URL for event {video_id}")
                        continue
                    
                    # Create filename with timestamp and device name
                    timestamp_str = event_time.strftime("%Y%m%d_%H%M%S")
                    filename = f"{device.name}_{timestamp_str}_{video_id}.mp4"
                    filepath = self.download_path / filename
                    
                    # Skip if already downloaded
                    if filepath.exists():
                        logger.debug(f"Video already exists: {filename}")
                        downloaded_files.append(filepath)
                        continue
                    
                    # Download the video
                    logger.info(f"Downloading video: {filename}")
                    device.recording_download(video_id, filename=str(filepath))
                    downloaded_files.append(filepath)
                    logger.info(f"Successfully downloaded: {filename}")
                    
                    if limit and len(downloaded_files) >= limit:
                        logger.info(f"Reached download limit of {limit}")
                        break
                        
            except Exception as e:
                logger.error(f"Error downloading videos from {device.name}: {e}", exc_info=True)
        
        logger.info(f"Downloaded {len(downloaded_files)} videos")
        return downloaded_files
    
    def download_all_videos(self, hours: Optional[int] = None, skip_existing: bool = True) -> List[Path]:
        """
        Download all available videos from Ring, optionally filtered by time.
        This ignores the last_processed.json tracking.
        
        Args:
            hours: Number of hours to look back (None = all available)
            skip_existing: If True, skip videos that already exist on disk
            
        Returns:
            List of downloaded video file paths
        """
        if not self.ring:
            logger.error("Not authenticated with Ring")
            return []
        
        downloaded_files = []
        skipped_files = []
        devices = self.get_devices()
        
        # Calculate cutoff time if hours specified
        cutoff_time = None
        if hours is not None:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            logger.info(f"Download All: Filtering for videos after {cutoff_time}")
        else:
            logger.info("Download All: No time filter - downloading ALL available videos")
        
        for device in devices:
            logger.info(f"Download All: Processing device: {device.name}")
            
            try:
                # Get video history (100 is the API limit per request)
                # Don't filter by kind - get all events (motion, ding, etc.)
                history = device.history(limit=100)
                logger.info(f"Download All: Found {len(history)} events for {device.name}")
                
                for event in history:
                    # Parse the ISO format timestamp from Ring API
                    created_at_str = event.get('created_at')
                    if isinstance(created_at_str, str):
                        event_time = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
                        event_time = event_time.replace(tzinfo=None)
                    else:
                        event_time = datetime.fromtimestamp(created_at_str)
                    
                    # Check time filter if specified
                    if cutoff_time and event_time < cutoff_time:
                        logger.debug(f"Skipping event {event['id']} - before cutoff time")
                        continue
                    
                    # Get video info
                    video_id = event['id']
                    video_url = device.recording_url(video_id)
                    
                    if not video_url:
                        logger.warning(f"No video URL for event {video_id}")
                        continue
                    
                    # Create filename
                    timestamp_str = event_time.strftime("%Y%m%d_%H%M%S")
                    filename = f"{device.name}_{timestamp_str}_{video_id}.mp4"
                    filepath = self.download_path / filename
                    
                    # Check if already exists
                    if filepath.exists():
                        if skip_existing:
                            logger.debug(f"Skipping existing video: {filename}")
                            skipped_files.append(filepath)
                            continue
                        else:
                            logger.info(f"Re-downloading existing video: {filename}")
                    
                    # Download the video
                    logger.info(f"Downloading: {filename}")
                    try:
                        device.recording_download(video_id, filename=str(filepath))
                        downloaded_files.append(filepath)
                        logger.info(f"✓ Downloaded: {filename}")
                    except Exception as dl_error:
                        logger.error(f"Failed to download {filename}: {dl_error}")
                        
            except Exception as e:
                logger.error(f"Error processing {device.name}: {e}", exc_info=True)
        
        logger.info(f"Download All Complete: {len(downloaded_files)} new videos, {len(skipped_files)} already existed")
        return downloaded_files
