"""Comprehensive tests for utils module."""

import re
import secrets
import time

import pytest

from memory_system.utils.cache import SmartCache
from memory_system.utils.security import (
    EnhancedPIIFilter,
    PasswordManager,
    PIIPatterns,
    SecureTokenManager,
    SecurityError,
)


class TestSmartCache:
    """Test SmartCache functionality."""

    @pytest.fixture
    def cache(self):
        """Create SmartCache instance."""
        return SmartCache(max_size=100, ttl=300)

    def test_cache_initialization(self, cache):
        """Test cache initialization."""
        assert cache.max_size == 100
        assert cache.ttl == 300
        assert cache._data == {}

    def test_cache_get_put(self, cache):
        """Test basic get/put operations."""
        key = "test_key"
        value = "test_value"

        # Should return None for non-existent key
        assert cache.get(key) is None

        # Put and get
        cache.put(key, value)
        assert cache.get(key) == value

    def test_cache_overwrite(self, cache):
        """Test overwriting existing key."""
        key = "test_key"
        value1 = "value1"
        value2 = "value2"

        cache.put(key, value1)
        assert cache.get(key) == value1

        cache.put(key, value2)
        assert cache.get(key) == value2

    def test_cache_different_types(self, cache):
        """Test caching different data types."""
        test_data = {
            "string": "test",
            "integer": 42,
            "float": 3.14,
            "list": [1, 2, 3],
            "dict": {"key": "value"},
            "none": None,
        }

        for key, value in test_data.items():
            cache.put(key, value)
            assert cache.get(key) == value

    def test_cache_clear(self, cache):
        """Test cache clear operation."""
        cache.put("key1", "value1")
        cache.put("key2", "value2")

        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"

        cache.clear()

        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_cache_stats(self, cache):
        """Test cache statistics."""
        stats = cache.get_stats()
        assert isinstance(stats, dict)
        assert "size" in stats
        assert "max_size" in stats
        assert "hit_rate" in stats
        assert stats["size"] == 0
        assert stats["max_size"] == 100
        assert stats["hit_rate"] == 0.0

        # Add some data
        cache.put("key1", "value1")
        cache.put("key2", "value2")

        stats = cache.get_stats()
        assert stats["size"] == 2


class TestPIIPatterns:
    """Test PII regex patterns."""

    def test_email_pattern(self):
        """Test email pattern matching."""
        pattern = PIIPatterns.EMAIL

        # Valid emails
        valid_emails = [
            "test@example.com",
            "user.name@domain.co.uk",
            "first.last+tag@example.org",
            "user123@test-domain.com",
        ]

        for email in valid_emails:
            assert pattern.search(email) is not None

        # Invalid emails
        invalid_emails = [
            "not-an-email",
            "@domain.com",
            "user@",
            "user@domain",
        ]

        for email in invalid_emails:
            assert pattern.search(email) is None

    def test_credit_card_pattern(self):
        """Test credit card pattern matching."""
        pattern = PIIPatterns.CREDIT_CARD

        # Valid credit cards
        valid_cards = [
            "1234 5678 9012 3456",
            "1234-5678-9012-3456",
            "1234567890123456",
        ]

        for card in valid_cards:
            assert pattern.search(card) is not None

        # Invalid credit cards
        invalid_cards = [
            "1234 5678 9012",  # Too short
            "1234 5678 9012 3456 7890",  # Too long
            "abcd efgh ijkl mnop",  # Non-numeric
        ]

        for card in invalid_cards:
            assert pattern.search(card) is None

    def test_ssn_pattern(self):
        """Test SSN pattern matching."""
        pattern = PIIPatterns.SSN

        # Valid SSNs
        valid_ssns = [
            "123-45-6789",
            "987-65-4321",
        ]

        for ssn in valid_ssns:
            assert pattern.search(ssn) is not None

        # Invalid SSNs
        invalid_ssns = [
            "123456789",  # No dashes
            "123-45-678",  # Too short
            "123-45-67890",  # Too long
            "abc-de-fghi",  # Non-numeric
        ]

        for ssn in invalid_ssns:
            assert pattern.search(ssn) is None

    def test_phone_pattern(self):
        """Test phone pattern matching."""
        pattern = PIIPatterns.PHONE

        # Valid phone numbers
        valid_phones = [
            "123-456-7890",
            "(123) 456-7890",
            "123.456.7890",
            "123 456 7890",
            "+1 123-456-7890",
            "1234567890",
        ]

        for phone in valid_phones:
            assert pattern.search(phone) is not None

        # Invalid phone numbers
        invalid_phones = [
            "123-456-789",  # Too short
            "123-456-78901",  # Too long
            "abc-def-ghij",  # Non-numeric
        ]

        for phone in invalid_phones:
            assert pattern.search(phone) is None

    def test_ip_address_pattern(self):
        """Test IP address pattern matching."""
        pattern = PIIPatterns.IP_ADDRESS

        # Valid IP addresses
        valid_ips = [
            "192.168.1.1",
            "10.0.0.1",
            "127.0.0.1",
            "255.255.255.255",
        ]

        for ip in valid_ips:
            assert pattern.search(ip) is not None

        # Invalid IP addresses
        invalid_ips = [
            "192.168.1",  # Too short
            "192.168.1.1.1",  # Too long
            "256.256.256.256",  # Out of range (but pattern might still match)
            "abc.def.ghi.jkl",  # Non-numeric
        ]

        for ip in invalid_ips:
            # Note: The regex might not catch all invalid IPs (like 256.256.256.256)
            # but it should catch clearly invalid ones
            assert pattern.search(ip) is None


class TestEnhancedPIIFilter:
    """Test EnhancedPIIFilter functionality."""

    @pytest.fixture
    def pii_filter(self):
        """Create EnhancedPIIFilter instance."""
        return EnhancedPIIFilter()

    def test_filter_initialization(self, pii_filter):
        """Test filter initialization."""
        assert isinstance(pii_filter.patterns, dict)
        assert len(pii_filter.patterns) > 0
        assert isinstance(pii_filter.stats, dict)
        assert all(count == 0 for count in pii_filter.stats.values())

    def test_detect_pii(self, pii_filter):
        """Test PII detection."""
        text = "Contact John at john@example.com or call 123-456-7890"
        detections = pii_filter.detect(text)

        assert isinstance(detections, dict)
        assert "email" in detections
        assert "phone" in detections
        assert detections["email"] == ["john@example.com"]
        assert detections["phone"] == ["123-456-7890"]

    def test_detect_no_pii(self, pii_filter):
        """Test detection with no PII."""
        text = "This is a normal text without any sensitive information."
        detections = pii_filter.detect(text)

        assert isinstance(detections, dict)
        assert len(detections) == 0

    def test_redact_pii(self, pii_filter):
        """Test PII redaction."""
        text = "Contact John at john@example.com or call 123-456-7890"
        redacted, found_pii, pii_types = pii_filter.redact(text)

        assert found_pii is True
        assert isinstance(pii_types, list)
        assert "email" in pii_types
        assert "phone" in pii_types
        assert "john@example.com" not in redacted
        assert "123-456-7890" not in redacted
        assert "[EMAIL_REDACTED]" in redacted
        assert "[PHONE_REDACTED]" in redacted

    def test_redact_no_pii(self, pii_filter):
        """Test redaction with no PII."""
        text = "This is a normal text without any sensitive information."
        redacted, found_pii, pii_types = pii_filter.redact(text)

        assert found_pii is False
        assert pii_types == []
        assert redacted == text

    def test_partial_redact_pii(self, pii_filter):
        """Test partial PII redaction."""
        text = "Contact John at john@example.com"
        redacted, found_pii, pii_types = pii_filter.partial_redact(text, preserve_chars=2)

        assert found_pii is True
        assert "email" in pii_types
        assert "john@example.com" not in redacted
        # Should preserve some characters
        assert "jo" in redacted and "om" in redacted

    def test_filter_stats(self, pii_filter):
        """Test filter statistics."""
        text = "Contact john@example.com and jane@test.org"
        pii_filter.detect(text)

        stats = pii_filter.get_stats()
        assert stats["email"] == 2

        # Reset stats
        pii_filter.reset_stats()
        stats = pii_filter.get_stats()
        assert all(count == 0 for count in stats.values())

    def test_custom_patterns(self):
        """Test custom PII patterns."""
        custom_patterns = {"custom_id": re.compile(r"ID-\d{6}")}

        pii_filter = EnhancedPIIFilter(custom_patterns=custom_patterns)

        text = "Your ID is ID-123456"
        detections = pii_filter.detect(text)

        assert "custom_id" in detections
        assert detections["custom_id"] == ["ID-123456"]


class TestSecureTokenManager:
    """Test SecureTokenManager functionality."""

    @pytest.fixture
    def token_manager(self):
        """Create SecureTokenManager instance."""
        secret = secrets.token_urlsafe(32)
        return SecureTokenManager(secret)

    def test_token_manager_initialization(self, token_manager):
        """Test token manager initialization."""
        assert token_manager.algorithm == "HS256"
        assert token_manager.issuer == "unified-memory-system"
        assert len(token_manager.secret_key) >= 32

    def test_token_manager_invalid_secret(self):
        """Test token manager with invalid secret."""
        with pytest.raises(SecurityError) as exc_info:
            SecureTokenManager("short")
        assert "at least 32 characters" in str(exc_info.value)

    def test_token_manager_invalid_algorithm(self):
        """Test token manager with invalid algorithm."""
        secret = secrets.token_urlsafe(32)
        with pytest.raises(SecurityError) as exc_info:
            SecureTokenManager(secret, algorithm="INVALID")
        assert "not allowed" in str(exc_info.value)

    def test_generate_token(self, token_manager):
        """Test token generation."""
        user_id = "test_user"
        token = token_manager.generate_token(user_id)

        assert isinstance(token, str)
        assert len(token) > 0

        # Verify token can be decoded
        payload = token_manager.verify_token(token)
        assert payload is not None
        assert payload["sub"] == user_id

    def test_generate_token_with_custom_params(self, token_manager):
        """Test token generation with custom parameters."""
        user_id = "test_user"
        expires_in = 7200
        scopes = ["read", "write"]
        audience = "test_api"

        token = token_manager.generate_token(
            user_id, expires_in=expires_in, scopes=scopes, audience=audience
        )

        payload = token_manager.verify_token(token, audience=audience)
        assert payload["sub"] == user_id
        assert payload["scopes"] == scopes
        assert payload["aud"] == audience

    def test_generate_token_invalid_user_id(self, token_manager):
        """Test token generation with invalid user ID."""
        with pytest.raises(SecurityError) as exc_info:
            token_manager.generate_token("")
        assert "Invalid user_id" in str(exc_info.value)

        with pytest.raises(SecurityError) as exc_info:
            token_manager.generate_token("x" * 101)
        assert "Invalid user_id" in str(exc_info.value)

    def test_generate_token_invalid_expiration(self, token_manager):
        """Test token generation with invalid expiration."""
        with pytest.raises(SecurityError) as exc_info:
            token_manager.generate_token("test_user", expires_in=0)
        assert "Invalid expiration time" in str(exc_info.value)

        with pytest.raises(SecurityError) as exc_info:
            token_manager.generate_token("test_user", expires_in=100000)
        assert "Invalid expiration time" in str(exc_info.value)

    def test_verify_token(self, token_manager):
        """Test token verification."""
        user_id = "test_user"
        token = token_manager.generate_token(user_id)

        payload = token_manager.verify_token(token)
        assert payload is not None
        assert payload["sub"] == user_id
        assert payload["iss"] == "unified-memory-system"

    def test_verify_invalid_token(self, token_manager):
        """Test verification of invalid token."""
        with pytest.raises(SecurityError) as exc_info:
            token_manager.verify_token("invalid_token")
        assert "Invalid token" in str(exc_info.value)

    def test_verify_expired_token(self, token_manager):
        """Test verification of expired token."""
        user_id = "test_user"
        token = token_manager.generate_token(user_id, expires_in=1)

        # Wait for token to expire
        time.sleep(2)

        with pytest.raises(SecurityError) as exc_info:
            token_manager.verify_token(token)
        assert "expired" in str(exc_info.value).lower()

    def test_revoke_token(self, token_manager):
        """Test token revocation."""
        user_id = "test_user"
        token = token_manager.generate_token(user_id)

        # Verify token works
        payload = token_manager.verify_token(token)
        assert payload is not None

        # Revoke token
        result = token_manager.revoke_token(token)
        assert result is True

        # Verify token is now invalid
        with pytest.raises(SecurityError) as exc_info:
            token_manager.verify_token(token)
        assert "revoked" in str(exc_info.value).lower()

    def test_generate_refresh_token(self, token_manager):
        """Test refresh token generation."""
        user_id = "test_user"
        refresh_token = token_manager.generate_refresh_token(user_id)

        assert isinstance(refresh_token, str)
        assert len(refresh_token) > 0

        # Verify refresh token
        payload = token_manager.verify_token(refresh_token, audience="refresh")
        assert payload["sub"] == user_id
        assert payload["token_type"] == "refresh"

    def test_token_manager_stats(self, token_manager):
        """Test token manager statistics."""
        stats = token_manager.get_stats()
        assert isinstance(stats, dict)
        assert "algorithm" in stats
        assert "issuer" in stats
        assert "revoked_tokens_count" in stats
        assert stats["algorithm"] == "HS256"
        assert stats["issuer"] == "unified-memory-system"


class TestPasswordManager:
    """Test PasswordManager functionality."""

    def test_hash_password(self):
        """Test password hashing."""
        password = "test_password_123"
        hashed, salt = PasswordManager.hash_password(password)

        assert isinstance(hashed, str)
        assert isinstance(salt, bytes)
        assert len(hashed) > 0
        assert len(salt) > 0
        assert hashed != password

    def test_hash_password_with_salt(self):
        """Test password hashing with provided salt."""
        password = "test_password_123"
        salt = b"test_salt_16_bytes"
        hashed, returned_salt = PasswordManager.hash_password(password, salt)

        assert returned_salt == salt
        assert isinstance(hashed, str)

    def test_hash_password_short_password(self):
        """Test hashing short password."""
        with pytest.raises(SecurityError) as exc_info:
            PasswordManager.hash_password("short")
        assert "at least 8 characters" in str(exc_info.value)

    def test_verify_password(self):
        """Test password verification."""
        password = "test_password_123"
        hashed, salt = PasswordManager.hash_password(password)

        # Correct password
        assert PasswordManager.verify_password(password, hashed, salt) is True

        # Wrong password
        assert PasswordManager.verify_password("wrong_password", hashed, salt) is False

    def test_verify_password_with_exception(self):
        """Test password verification with exception."""
        # Should return False on any exception
        result = PasswordManager.verify_password("password", "invalid_hash", b"salt")
        assert result is False

    def test_generate_secure_password(self):
        """Test secure password generation."""
        password = PasswordManager.generate_secure_password()

        assert isinstance(password, str)
        assert len(password) == 16  # Default length

        # Check character requirements
        assert any(c.islower() for c in password)
        assert any(c.isupper() for c in password)
        assert any(c.isdigit() for c in password)
        assert any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)

    def test_generate_secure_password_custom_length(self):
        """Test secure password generation with custom length."""
        password = PasswordManager.generate_secure_password(length=20)
        assert len(password) == 20

    def test_generate_secure_password_no_symbols(self):
        """Test secure password generation without symbols."""
        password = PasswordManager.generate_secure_password(include_symbols=False)

        assert isinstance(password, str)
        assert len(password) == 16
        assert not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)

    def test_generate_secure_password_invalid_length(self):
        """Test secure password generation with invalid length."""
        with pytest.raises(SecurityError) as exc_info:
            PasswordManager.generate_secure_password(length=5)
        assert "between 8 and 128" in str(exc_info.value)

        with pytest.raises(SecurityError) as exc_info:
            PasswordManager.generate_secure_password(length=200)
        assert "between 8 and 128" in str(exc_info.value)


class TestEncryptionManager:
    """Test EncryptionManager functionality."""

    @pytest.fixture
    def encryption_manager(self):
        """Create EncryptionManager instance."""
