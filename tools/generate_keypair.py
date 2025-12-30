#!/usr/bin/env python3
"""
Generate Ed25519 Keypair for MemOpt License System

SECURITY WARNING:
- Run this script ONCE
- Store the private key in a secure password manager
- NEVER commit the private key to git
- NEVER share the private key

Usage:
    python tools/generate_keypair.py
"""

import sys

try:
    from nacl.signing import SigningKey
    import binascii
except ImportError:
    print("ERROR: PyNaCl not installed")
    print("Install with: pip install pynacl")
    sys.exit(1)


def generate_keypair():
    """Generate new Ed25519 keypair for license signing"""

    print("=" * 80)
    print("MemOpt License Keypair Generator")
    print("=" * 80)
    print()
    print("Generating new Ed25519 keypair...")
    print()

    # Generate keypair
    private_key = SigningKey.generate()
    public_key = private_key.verify_key

    private_key_hex = binascii.hexlify(bytes(private_key)).decode()
    public_key_hex = binascii.hexlify(bytes(public_key)).decode()

    print("=" * 80)
    print("PRIVATE KEY (KEEP SECRET - NEVER SHARE)")
    print("=" * 80)
    print()
    print("Store this in a secure password manager:")
    print()
    print(private_key_hex)
    print()
    print("Use this key in tools/sign_license.py")
    print()

    print("=" * 80)
    print("PUBLIC KEY (EMBED IN CODE)")
    print("=" * 80)
    print()
    print("Update memopt/license.py line ~30:")
    print()
    print(f'PUBLIC_KEY_HEX = "{public_key_hex}"')
    print()

    print("=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print()
    print("1. Copy the PRIVATE KEY to your password manager")
    print("2. Update PUBLIC_KEY_HEX in memopt/license.py")
    print("3. Update PRIVATE_KEY_HEX in tools/sign_license.py")
    print("4. Test by running: python tools/sign_license.py --test")
    print()
    print("SECURITY REMINDER:")
    print("- Private key = your master signing key (keep secret!)")
    print("- Public key = embedded in customer code (safe to share)")
    print()


if __name__ == "__main__":
    generate_keypair()
