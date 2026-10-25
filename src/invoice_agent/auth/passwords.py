"""Password hashing and random tokens, using only the standard library.

Passwords are hashed with scrypt (memory-hard, so guessing is slow even on GPUs). Tokens
(sessions, invites) are random; only their SHA-256 is stored, so a leaked database can't
be used to log in.
"""

import base64
import hashlib
import secrets

MIN_PASSWORD_LENGTH = 10

# scrypt cost: 2**14 iterations x 8 blocks = 16 MB per hash, ~50 ms. OWASP's minimum.
_N, _R, _P = 2**14, 8, 1


def _b64(data: bytes) -> str:
    """Encode bytes as unpadded URL-safe base64.

    Args:
        data: The bytes to encode.

    Returns:
        The base64 text.
    """
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _unb64(text: str) -> bytes:
    """Decode unpadded URL-safe base64 (the inverse of `_b64`).

    Args:
        text: The base64 text.

    Returns:
        The decoded bytes.
    """
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def hash_password(password: str) -> str:
    """Hash a password with a fresh random salt.

    Args:
        password: The plain-text password.

    Returns:
        A self-describing string 'scrypt$N$r$p$salt$hash' to store in the database.
    """
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=_N, r=_R, p=_P, dklen=32)
    return f"scrypt${_N}${_R}${_P}${_b64(salt)}${_b64(digest)}"


def verify_password(password: str, stored: str) -> bool:
    """Check a password against a stored hash.

    Args:
        password: The plain-text password someone typed.
        stored: A string produced by `hash_password`.

    Returns:
        True if the password matches. Compared in constant time; a malformed stored
        value simply doesn't match.
    """
    try:
        scheme, n, r, p, salt, digest = stored.split("$")
        if scheme != "scrypt":
            return False
        expected = _unb64(digest)
        actual = hashlib.scrypt(
            password.encode(), salt=_unb64(salt), n=int(n), r=int(r), p=int(p), dklen=len(expected)
        )
    except ValueError:
        return False
    return secrets.compare_digest(actual, expected)


def new_token() -> str:
    """Create a random token for a session cookie or an invite link.

    Returns:
        43 URL-safe characters (256 bits of randomness).
    """
    return secrets.token_urlsafe(32)


def token_hash(token: str) -> str:
    """Hash a token for storage (tokens are random, so a fast hash is enough).

    Args:
        token: A token from `new_token`.

    Returns:
        Its SHA-256 as hex.
    """
    return hashlib.sha256(token.encode()).hexdigest()


# Verified against when an email isn't found, so a failed login takes the same time
# whether or not the account exists.
DUMMY_HASH = hash_password(secrets.token_urlsafe(16))
