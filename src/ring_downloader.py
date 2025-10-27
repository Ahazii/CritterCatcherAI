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
            # Access devices - the ring-doorbell library uses direct properties
            logger.debug("Getting Ring devices from properties")
            if hasattr(self.ring, 'doorbots'):
                devices.extend(self.ring.doorbots)
                logger.debug(f"Found {len(self.ring.doorbots)} doorbots")
            if hasattr(self.ring, 'stickup_cams'):
                devices.extend(self.ring.stickup_cams)
                logger.debug(f"Found {len(self.ring.stickup_cams)} stickup_cams")
            if hasattr(self.ring, 'other'):
                devices.extend(self.ring.other)
                logger.debug(f"Found {len(self.ring.other)} other devices")
            
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
                history = device.history(limit=limit or 100, kind='ding')
                
                # Filter by time
                cutoff_time = datetime.now() - timedelta(hours=hours)
                
                for event in history:
                    event_time = datetime.fromtimestamp(event['created_at'])
                    
                    if event_time < cutoff_time:
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
                    
                    if limit and len(downloaded_files) >= limit:
                        logger.info(f"Reached download limit of {limit}")
                        break
                        
            except Exception as e:
                logger.error(f"Error downloading videos from {device.name}: {e}", exc_info=True)
        
        logger.info(f"Downloaded {len(downloaded_files)} videos")
        return downloaded_files
