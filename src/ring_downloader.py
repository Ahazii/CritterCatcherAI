"""
Ring video downloader module.
Downloads videos from Ring doorbells using the ring-doorbell library.
"""
import os
import logging
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
import json
from functools import wraps

from ring_doorbell import Ring, Auth
from ring_doorbell.exceptions import Requires2FAError
from oauthlib.oauth2 import MissingTokenError

from datetime_utils import parse_timestamp, format_timestamp
from download_tracker import DownloadTracker


logger = logging.getLogger(__name__)

# Rate limiting constants
DELAY_BETWEEN_DOWNLOADS = 2.0  # Seconds between each video download
MAX_RETRIES = 3  # Maximum number of retry attempts for rate-limited requests
INITIAL_BACKOFF = 5.0  # Initial backoff delay in seconds for 429 errors
MAX_BACKOFF = 60.0  # Maximum backoff delay in seconds
BACKOFF_MULTIPLIER = 2.0  # Exponential backoff multiplier


def retry_on_rate_limit(max_retries=MAX_RETRIES, initial_backoff=INITIAL_BACKOFF):
    """
    Decorator that implements exponential backoff retry logic for rate-limited requests.
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_backoff: Initial delay in seconds before first retry
    
    Returns:
        Decorated function with retry logic
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            backoff_delay = initial_backoff
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_str = str(e).lower()
                    
                    # Check if this is a rate limit error (HTTP 429)
                    is_rate_limit = (
                        '429' in error_str or 
                        'too many requests' in error_str or
                        'rate limit' in error_str
                    )
                    
                    if is_rate_limit and attempt < max_retries:
                        logger.warning(
                            f"Rate limit hit (attempt {attempt + 1}/{max_retries}). "
                            f"Retrying in {backoff_delay:.1f}s..."
                        )
                        time.sleep(backoff_delay)
                        backoff_delay = min(backoff_delay * BACKOFF_MULTIPLIER, MAX_BACKOFF)
                    else:
                        # Not a rate limit error or out of retries
                        raise
            
            return None
        return wrapper
    return decorator


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
        self.download_tracker = DownloadTracker()
        
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
                    # Access token from Auth object (ring-doorbell library uses _token)
                    token_data = self.auth._token
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
            # Access token from Auth object (ring-doorbell library uses _token)
            token_data = self.auth._token
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
                    # Parse the timestamp from Ring API (handles various formats)
                    created_at = event.get('created_at')
                    
                    try:
                        event_time = parse_timestamp(created_at)
                    except ValueError as e:
                        logger.error(f"Failed to parse timestamp for event {event.get('id')}: {e}")
                        continue
                    
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
                    timestamp_str = format_timestamp(event_time)
                    filename = f"{device.name}_{timestamp_str}_{video_id}.mp4"
                    filepath = self.download_path / filename
                    
                    # Skip if already downloaded
                    if filepath.exists():
                        logger.debug(f"Video already exists: {filename}")
                        downloaded_files.append(filepath)
                        continue
                    
                    # Download the video with rate limiting
                    logger.info(f"Downloading video: {filename}")
                    success, error_msg = self._download_video_with_retry(device, video_id, filepath)
                    
                    if success:
                        downloaded_files.append(filepath)
                        logger.info(f"✓ Downloaded: {filename}")
                        
                        # Add delay between downloads to respect rate limits
                        if DELAY_BETWEEN_DOWNLOADS > 0:
                            time.sleep(DELAY_BETWEEN_DOWNLOADS)
                    else:
                        logger.error(f"Failed to download {filename}: {error_msg}")
                    
                    if limit and len(downloaded_files) >= limit:
                        logger.info(f"Reached download limit of {limit}")
                        break
                        
            except Exception as e:
                logger.error(f"Error downloading videos from {device.name}: {e}", exc_info=True)
        
        logger.info(f"Downloaded {len(downloaded_files)} videos")
        return downloaded_files
    
    def _download_video_with_retry(self, device, video_id: int, filepath: Path) -> Tuple[bool, str]:
        """
        Download a single video with retry logic for rate limiting.
        
        Args:
            device: Ring device object
            video_id: Video ID to download
            filepath: Path where video should be saved
        
        Returns:
            Tuple of (success: bool, error_message: str)
        """
        backoff_delay = INITIAL_BACKOFF
        
        for attempt in range(MAX_RETRIES + 1):
            try:
                device.recording_download(video_id, filename=str(filepath))
                return True, ""
            
            except Exception as e:
                error_str = str(e).lower()
                
                # Check if this is a 404 error (video not available)
                if '404' in error_str or 'not found' in error_str:
                    logger.warning(f"Video {video_id} not available (404) - motion event without recording")
                    return False, "404_NOT_FOUND"
                
                # Check if this is a rate limit error (HTTP 429)
                is_rate_limit = (
                    '429' in error_str or 
                    'too many requests' in error_str or
                    'rate limit' in error_str
                )
                
                if is_rate_limit and attempt < MAX_RETRIES:
                    logger.warning(
                        f"Rate limit hit on video {video_id} "
                        f"(attempt {attempt + 1}/{MAX_RETRIES}). "
                        f"Waiting {backoff_delay:.1f}s before retry..."
                    )
                    time.sleep(backoff_delay)
                    backoff_delay = min(backoff_delay * BACKOFF_MULTIPLIER, MAX_BACKOFF)
                else:
                    # Not a rate limit error or out of retries
                    return False, str(e)
        
        return False, "Max retries exceeded"
    
    def download_all_videos(
        self,
        hours: Optional[int] = None,
        skip_existing: bool = True,
        download_limit: Optional[int] = None
    ) -> dict:
        """
        Download all available videos from Ring, optionally filtered by time.
        Uses download tracker to prevent duplicate downloads.
        
        Args:
            hours: Number of hours to look back (None = all available)
            skip_existing: If True, skip videos that already exist on disk or in database
            download_limit: Maximum number of new videos to download (None = unlimited)
            
        Returns:
            dict: Statistics with keys: new_downloads, already_downloaded, unavailable, failed
        """
        if not self.ring:
            logger.error("Not authenticated with Ring")
            return {'new_downloads': 0, 'already_downloaded': 0, 'unavailable': 0, 'failed': 0}

        if download_limit is not None and download_limit <= 0:
            logger.info("Download limit set to 0 - skipping downloads")
            return {'new_downloads': 0, 'already_downloaded': 0, 'unavailable': 0, 'failed': 0}
        
        downloaded_files = []
        already_downloaded_count = 0
        unavailable_count = 0
        failed_files = []
        rate_limited_count = 0
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
                    # Parse the timestamp from Ring API (handles various formats)
                    created_at = event.get('created_at')
                    
                    try:
                        event_time = parse_timestamp(created_at)
                    except ValueError as e:
                        logger.error(f"Failed to parse timestamp for event {event.get('id')}: {e}")
                        continue
                    
                    # Check time filter if specified
                    if cutoff_time and event_time < cutoff_time:
                        logger.debug(f"Skipping event {event['id']} - before cutoff time")
                        continue
                    
                    # Get video info
                    video_id = event['id']
                    event_id_str = str(video_id)
                    
                    # Check if already downloaded in database
                    if skip_existing and self.download_tracker.is_downloaded(event_id_str):
                        logger.debug(f"Event {video_id} already in database - skipping")
                        already_downloaded_count += 1
                        continue
                    
                    video_url = device.recording_url(video_id)
                    
                    if not video_url:
                        logger.warning(f"No video URL for event {video_id}")
                        continue
                    
                    # Create filename
                    timestamp_str = format_timestamp(event_time)
                    filename = f"{device.name}_{timestamp_str}_{video_id}.mp4"
                    filepath = self.download_path / filename
                    
                    # Check if already exists on filesystem
                    if filepath.exists():
                        if skip_existing:
                            logger.debug(f"Skipping existing video on disk: {filename}")
                            # Track in database if not already there
                            if not self.download_tracker.is_downloaded(event_id_str):
                                self.download_tracker.mark_downloaded(
                                    event_id_str, filename, device.name, str(filepath)
                                )
                            already_downloaded_count += 1
                            continue
                        else:
                            logger.info(f"Re-downloading existing video: {filename}")
                    
                    # Also check review folders
                    review_base = Path("/data/review")
                    found_in_review = False
                    if review_base.exists():
                        for root, _, files in os.walk(review_base):
                            if filename in files:
                                logger.debug(f"Video already in review: {filename}")
                                # Track in database if not already there
                                if not self.download_tracker.is_downloaded(event_id_str):
                                    review_path = os.path.join(root, filename)
                                    self.download_tracker.mark_downloaded(
                                        event_id_str, filename, device.name, review_path, "processed"
                                    )
                                already_downloaded_count += 1
                                found_in_review = True
                                break
                        if found_in_review:
                            continue
                    
                    # Download the video with rate limiting
                    logger.info(f"Downloading: {filename}")
                    success, error_msg = self._download_video_with_retry(device, video_id, filepath)
                    
                    if success:
                        downloaded_files.append(filepath)
                        # Track in database
                        self.download_tracker.mark_downloaded(
                            event_id_str, filename, device.name, str(filepath)
                        )
                        logger.info(f"✓ Downloaded: {filename}")
                        
                        # Add delay between downloads to respect rate limits
                        if DELAY_BETWEEN_DOWNLOADS > 0:
                            time.sleep(DELAY_BETWEEN_DOWNLOADS)

                        # Stop if download limit reached
                        if download_limit is not None and len(downloaded_files) >= download_limit:
                            logger.info(f"Download limit reached ({download_limit}). Stopping further downloads.")
                            break
                    else:
                        # Handle 404 (unavailable) separately
                        if error_msg == "404_NOT_FOUND":
                            unavailable_count += 1
                            # Don't log as error, just info
                            logger.info(f"Video {video_id} unavailable (motion event without recording)")
                        else:
                            failed_files.append(filename)
                            # Check if it was a rate limit error
                            if '429' in error_msg or 'too many requests' in error_msg.lower():
                                rate_limited_count += 1
                                logger.error(f"Rate limited downloading {filename}. Consider waiting before trying again.")
                            else:
                                logger.error(f"Failed to download {filename}: {error_msg}")
                        
                # Stop after this device if limit reached
                if download_limit is not None and len(downloaded_files) >= download_limit:
                    break

            except Exception as e:
                logger.error(f"Error processing {device.name}: {e}", exc_info=True)

            # Stop after this device if limit reached
            if download_limit is not None and len(downloaded_files) >= download_limit:
                break
        
        # Provide detailed summary
        logger.info(f"="*80)
        logger.info(f"Download All Complete Summary:")
        logger.info(f"  ✓ New downloads: {len(downloaded_files)}")
        logger.info(f"  ⊘ Already downloaded: {already_downloaded_count}")
        logger.info(f"  ⓘ Unavailable (404): {unavailable_count}")
        logger.info(f"  ✗ Failed downloads: {len(failed_files)}")
        if rate_limited_count > 0:
            logger.warning(f"  ⚠ Rate limited: {rate_limited_count} videos")
            logger.warning(f"  Ring API has rate limits. Consider waiting 10-15 minutes before retrying.")
        logger.info(f"="*80)
        
        return {
            'new_downloads': len(downloaded_files),
            'already_downloaded': already_downloaded_count,
            'unavailable': unavailable_count,
            'failed': len(failed_files)
        }
