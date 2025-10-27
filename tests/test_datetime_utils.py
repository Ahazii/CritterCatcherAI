"""
Unit tests for datetime_utils module.
"""
import sys
from pathlib import Path
from datetime import datetime, timezone
import pytest

# Add src to path so we can import modules
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from datetime_utils import parse_timestamp, format_timestamp


class TestParseTimestamp:
    """Tests for parse_timestamp() function."""
    
    def test_parse_iso_string_with_z(self):
        """Test parsing ISO format string with Z suffix."""
        result = parse_timestamp("2025-10-25T18:23:19.683Z")
        
        assert isinstance(result, datetime)
        assert result.year == 2025
        assert result.month == 10
        assert result.day == 25
        assert result.hour == 18
        assert result.minute == 23
        assert result.second == 19
        assert result.microsecond == 683000
        assert result.tzinfo is None  # Should be naive
    
    def test_parse_iso_string_without_z(self):
        """Test parsing ISO format string without Z suffix."""
        result = parse_timestamp("2025-10-25T18:23:19.683+00:00")
        
        assert isinstance(result, datetime)
        assert result.year == 2025
        assert result.tzinfo is None  # Should be naive
    
    def test_parse_iso_string_no_microseconds(self):
        """Test parsing ISO format string without microseconds."""
        result = parse_timestamp("2025-10-25T18:23:19Z")
        
        assert isinstance(result, datetime)
        assert result.microsecond == 0
        assert result.tzinfo is None
    
    def test_parse_datetime_object_naive(self):
        """Test parsing naive datetime object."""
        dt = datetime(2025, 10, 25, 18, 23, 19)
        result = parse_timestamp(dt)
        
        assert result == dt
        assert result.tzinfo is None
    
    def test_parse_datetime_object_with_timezone(self):
        """Test parsing datetime object with timezone."""
        dt = datetime(2025, 10, 25, 18, 23, 19, tzinfo=timezone.utc)
        result = parse_timestamp(dt)
        
        assert isinstance(result, datetime)
        assert result.tzinfo is None  # Should strip timezone
        assert result.year == dt.year
        assert result.hour == dt.hour
    
    def test_parse_unix_timestamp_int(self):
        """Test parsing integer Unix timestamp."""
        # October 26, 2025 12:23:58 UTC
        timestamp = 1761512638
        result = parse_timestamp(timestamp)
        
        assert isinstance(result, datetime)
        assert result.year == 2025
        assert result.month == 10
        assert result.day == 26
    
    def test_parse_unix_timestamp_float(self):
        """Test parsing float Unix timestamp with milliseconds."""
        timestamp = 1761512638.095
        result = parse_timestamp(timestamp)
        
        assert isinstance(result, datetime)
        assert result.microsecond > 0  # Should preserve fractional seconds
    
    def test_parse_invalid_string(self):
        """Test parsing invalid string raises ValueError."""
        with pytest.raises(ValueError, match="Invalid ISO format string"):
            parse_timestamp("not a valid timestamp")
    
    def test_parse_invalid_type(self):
        """Test parsing unsupported type raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported timestamp type"):
            parse_timestamp([2025, 10, 25])  # List is not supported
    
    def test_parse_invalid_unix_timestamp(self):
        """Test parsing invalid Unix timestamp raises ValueError."""
        with pytest.raises(ValueError, match="Invalid Unix timestamp"):
            # Year 3000 timestamp (out of range for some systems)
            parse_timestamp(32503680000)


class TestFormatTimestamp:
    """Tests for format_timestamp() function."""
    
    def test_format_default(self):
        """Test formatting with default format string."""
        dt = datetime(2025, 10, 25, 18, 23, 19)
        result = format_timestamp(dt)
        
        assert result == "20251025_182319"
    
    def test_format_custom_format(self):
        """Test formatting with custom format string."""
        dt = datetime(2025, 10, 25, 18, 23, 19)
        result = format_timestamp(dt, "%Y-%m-%d %H:%M:%S")
        
        assert result == "2025-10-25 18:23:19"
    
    def test_format_iso_format(self):
        """Test formatting to ISO format."""
        dt = datetime(2025, 10, 25, 18, 23, 19)
        result = format_timestamp(dt, "%Y-%m-%dT%H:%M:%S")
        
        assert result == "2025-10-25T18:23:19"
    
    def test_format_date_only(self):
        """Test formatting date only."""
        dt = datetime(2025, 10, 25, 18, 23, 19)
        result = format_timestamp(dt, "%Y-%m-%d")
        
        assert result == "2025-10-25"
    
    def test_format_invalid_datetime(self):
        """Test formatting non-datetime raises ValueError."""
        with pytest.raises(ValueError, match="Expected datetime object"):
            format_timestamp("not a datetime")
    
    def test_format_invalid_format_string(self):
        """Test formatting with non-string format raises TypeError."""
        dt = datetime(2025, 10, 25, 18, 23, 19)
        with pytest.raises(TypeError, match="format_string must be str"):
            format_timestamp(dt, 123)


class TestIntegration:
    """Integration tests combining both functions."""
    
    def test_parse_and_format_iso_string(self):
        """Test parsing ISO string and formatting back."""
        original = "2025-10-25T18:23:19.683Z"
        dt = parse_timestamp(original)
        formatted = format_timestamp(dt, "%Y-%m-%dT%H:%M:%S")
        
        assert formatted == "2025-10-25T18:23:19"
    
    def test_parse_and_format_unix_timestamp(self):
        """Test parsing Unix timestamp and formatting."""
        timestamp = 1761512638
        dt = parse_timestamp(timestamp)
        formatted = format_timestamp(dt)
        
        assert len(formatted) == 15  # YYYYmmdd_HHMMSS format
        assert "_" in formatted
    
    def test_round_trip_datetime(self):
        """Test datetime object survives parse and format."""
        original = datetime(2025, 10, 25, 18, 23, 19)
        parsed = parse_timestamp(original)
        formatted = format_timestamp(parsed)
        
        assert parsed == original
        assert formatted == "20251025_182319"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
