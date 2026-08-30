# security.py
import os
import time
import token
import requests
from typing import Dict, Any, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError

from dotenv import load_dotenv
from observability import observe

load_dotenv()

## Task 3 - Complete the security.py file to handle JWT authentication and authorization.

AUTH0_DOMAIN = os.getenv("AUTH0_DOMAIN")
API_AUDIENCE = os.getenv("AUTH0_AUDIENCE")
ALGORITHMS = [os.getenv("ALGORITHMS", "RS256")]

if not AUTH0_DOMAIN or not API_AUDIENCE:
    raise RuntimeError("AUTH0_DOMAIN and API_AUDIENCE must be set in environment")

JWKS_URL = f"https://{AUTH0_DOMAIN}/.well-known/jwks.json"

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")  # tokenUrl not used but required by class

# Simple JWKS cache
_jwks_cache: Dict[str, Any] = {"keys": [], "fetched_at": 0}
JWKS_CACHE_SECONDS = 60 * 60  # 1 hour

@observe()
def _fetch_jwks(force: bool = False) -> Dict[str, Any]:
    """Fetch JWKS from Auth0 with simple caching."""
    now = time.time()
    if force or not _jwks_cache["keys"] or (now - _jwks_cache["fetched_at"] > JWKS_CACHE_SECONDS):
        resp = requests.get(JWKS_URL, timeout=5)
        resp.raise_for_status()
        _jwks_cache["keys"] = resp.json().get("keys", [])
        _jwks_cache["fetched_at"] = now
    return _jwks_cache


def _get_rsa_key(unverified_header: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    jwks = _fetch_jwks()
    for key in jwks.get("keys", []):
        if key.get("kid") == unverified_header.get("kid"):
            return {
                "kty": key.get("kty"),
                "kid": key.get("kid"),
                "use": key.get("use"),
                "n": key.get("n"),
                "e": key.get("e"),
            }
    return None


def validate_token(token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
    """
    FastAPI dependency to validate a JWT from Auth0.
    Returns the decoded payload (claims) if validation succeeded.
    Raises HTTPException(401) on failure.
    """
    try:
        unverified_header = jwt.get_unverified_header(token)
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token header.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    rsa_key = _get_rsa_key(unverified_header)
    if not rsa_key:
        # refresh JWKS once and retry (in case key rotated)
        _fetch_jwks(force=True)
        rsa_key = _get_rsa_key(unverified_header)
        if not rsa_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unable to find appropriate key",
                headers={"WWW-Authenticate": "Bearer"},
            )

    try:
        payload = jwt.decode(
            token,
            rsa_key,
            algorithms=ALGORITHMS,
            audience=API_AUDIENCE,
            issuer=f"https://{AUTH0_DOMAIN}/",
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.JWTClaimsError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect claims. Check audience and issuer.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unable to parse authentication token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload["_raw_token"] = token
    return payload

@observe()
def get_user_email_from_auth0(token: str) -> Optional[str]:
    """
    Fetch user email from Auth0's userinfo endpoint using the access token.
    """
    try:
        userinfo_url = f"https://{AUTH0_DOMAIN}/userinfo"
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(userinfo_url, headers=headers, timeout=5)
        response.raise_for_status()
        userinfo = response.json()
        return userinfo.get("email")
    except Exception:
        return None
