"""
auth/auth.py — JWT authentication with bcrypt password hashing.
"""

import os
import datetime
import bcrypt
import jwt
from db.database import create_user, get_user_by_email

JWT_SECRET  = os.getenv("JWT_SECRET", "dev-secret-change-in-prod-32chars!!")
JWT_ALGO    = "HS256"
JWT_EXPIRES = 7  # days


# ── Password ───────────────────────────────────────────────────────────────────
def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())


# ── JWT ────────────────────────────────────────────────────────────────────────
def create_token(user_id: int, email: str) -> str:
    payload = {
        "sub":   str(user_id),
        "email": email,
        "exp":   datetime.datetime.utcnow() + datetime.timedelta(days=JWT_EXPIRES),
        "iat":   datetime.datetime.utcnow(),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)


def decode_token(token: str) -> dict | None:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
        payload["sub"] = int(payload["sub"])  # restore int user_id
        return payload
    except Exception:
        return None


# ── Register / Login ───────────────────────────────────────────────────────────
def register(name: str, email: str, password: str) -> tuple[bool, str]:
    if get_user_by_email(email):
        return False, "An account with this email already exists."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    try:
        create_user(email, hash_password(password), name)
        return True, "Account created successfully."
    except Exception as e:
        return False, str(e)


def login(email: str, password: str) -> tuple[str | None, str]:
    user = get_user_by_email(email)
    if not user:
        return None, "No account found with that email."
    if not verify_password(password, user["password_hash"]):
        return None, "Incorrect password."
    token = create_token(user["id"], user["email"])
    return token, "Login successful."
