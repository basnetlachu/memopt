#!/usr/bin/env python3
"""
MemOpt License Signing Tool

Sign customer licenses with Ed25519 cryptographic signatures.

SECURITY WARNING:
- Keep this file PRIVATE
- NEVER commit PRIVATE_KEY_HEX to git
- Store private key in password manager only

Usage:
    # Test mode (creates demo license)
    python tools/sign_license.py --test

    # Create license for customer
    python tools/sign_license.py \
        --customer-id acme-corp \
        --tier enterprise \
        --expires 2026-12-30 \
        --max-gpus 8 \
        --features vllm,quantization,profiling
"""

import sys
import json
import base64
import argparse
from datetime import datetime, timezone

try:
    from nacl.signing import SigningKey
except ImportError:
    print("ERROR: PyNaCl not installed")
    print("Install with: pip install pynacl")
    sys.exit(1)


# ============================================================================
# CONFIGURATION - UPDATE AFTER RUNNING generate_keypair.py
# ============================================================================

# Open tools/sign_license.py and change line 44 back to:
PRIVATE_KEY_HEX = "REPLACE_WITH_YOUR_PRIVATE_KEY_FROM_generate_keypair_py"


# ============================================================================


def sign_license(
    customer_id: str,
    tier: str,
    expires_at: str,
    max_gpus: int,
    features: list,
    output_file: str = None
) -> dict:
    """
    Sign a license for a customer.

    Args:
        customer_id: Unique customer identifier
        tier: License tier (enterprise or revenue_share)
        expires_at: Expiration date (YYYY-MM-DD)
        max_gpus: Maximum number of GPUs allowed
        features: List of enabled features
        output_file: Output filename (default: license_{customer_id}.json)

    Returns:
        Signed license dictionary
    """

    # Validate private key is set
    if PRIVATE_KEY_HEX == "REPLACE_WITH_YOUR_PRIVATE_KEY_FROM_generate_keypair_py":
        print("ERROR: Private key not configured!")
        print()
        print("Steps to fix:")
        print("1. Run: python tools/generate_keypair.py")
        print("2. Copy the PRIVATE KEY from output")
        print("3. Update PRIVATE_KEY_HEX in this file (tools/sign_license.py)")
        print()
        sys.exit(1)

    # Create license data
    issued_at = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    expires_at_iso = f"{expires_at}T23:59:59Z"

    license_data = {
        "customer_id": customer_id,
        "tier": tier,
        "issued_at": issued_at,
        "expires_at": expires_at_iso,
        "max_gpus": max_gpus,
        "features": features,
        "telemetry_required": tier == "revenue_share"
    }

    # Sign the license
    try:
        signing_key = SigningKey(bytes.fromhex(PRIVATE_KEY_HEX))
    except Exception as e:
        print(f"ERROR: Invalid private key: {e}")
        print("Make sure you copied the full hex string from generate_keypair.py")
        sys.exit(1)

    # Create signature
    payload = json.dumps(license_data, sort_keys=True).encode()
    signature = signing_key.sign(payload).signature
    license_data["signature"] = base64.b64encode(signature).decode()

    # Save to file
    if output_file is None:
        output_file = f"license_{customer_id}.json"

    with open(output_file, 'w') as f:
        json.dump(license_data, f, indent=2)

    print("=" * 80)
    print("License Created Successfully")
    print("=" * 80)
    print()
    print(f"Customer ID: {customer_id}")
    print(f"Tier: {tier}")
    print(f"Expires: {expires_at_iso}")
    print(f"Max GPUs: {max_gpus}")
    print(f"Features: {', '.join(features)}")
    print(f"Telemetry: {'Required' if license_data['telemetry_required'] else 'Optional'}")
    print()
    print(f"License saved to: {output_file}")
    print()
    print("=" * 80)
    print("Next Steps")
    print("=" * 80)
    print()
    print(f"1. Securely send {output_file} to customer")
    print("2. Provide installation instructions:")
    print()
    print("   pip install memopt")
    print(f"   sudo cp {output_file} /etc/memopt/license.json")
    print("   export MEMOPT_ENABLED=1")
    print("   # Restart vLLM")
    print()

    return license_data


def test_license():
    """Create a test license for validation"""
    print("Creating test license...")
    print()

    return sign_license(
        customer_id="test-customer",
        tier="enterprise",
        expires_at="2026-12-30",
        max_gpus=4,
        features=["vllm", "quantization", "profiling"],
        output_file="license_test.json"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Sign MemOpt customer licenses",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test mode
  python tools/sign_license.py --test

  # Enterprise customer
  python tools/sign_license.py \\
      --customer-id acme-corp \\
      --tier enterprise \\
      --expires 2026-12-30 \\
      --max-gpus 8 \\
      --features vllm,quantization,profiling

  # Revenue share customer
  python tools/sign_license.py \\
      --customer-id startup-inc \\
      --tier revenue_share \\
      --expires 2026-12-30 \\
      --max-gpus 999 \\
      --features vllm,quantization,profiling
        """
    )

    parser.add_argument(
        '--test',
        action='store_true',
        help='Create a test license for validation'
    )

    parser.add_argument(
        '--customer-id',
        type=str,
        help='Customer identifier (e.g., acme-corp)'
    )

    parser.add_argument(
        '--tier',
        type=str,
        choices=['enterprise', 'revenue_share'],
        help='License tier'
    )

    parser.add_argument(
        '--expires',
        type=str,
        help='Expiration date (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--max-gpus',
        type=int,
        help='Maximum number of GPUs'
    )

    parser.add_argument(
        '--features',
        type=str,
        help='Comma-separated list of features (e.g., vllm,quantization,profiling)'
    )

    parser.add_argument(
        '--output',
        type=str,
        help='Output filename (default: license_{customer_id}.json)'
    )

    args = parser.parse_args()

    if args.test:
        test_license()
        return

    # Validate required arguments
    if not all([args.customer_id, args.tier, args.expires, args.max_gpus, args.features]):
        parser.print_help()
        print()
        print("ERROR: All arguments required (or use --test)")
        sys.exit(1)

    # Parse features
    features = [f.strip() for f in args.features.split(',')]

    # Sign license
    sign_license(
        customer_id=args.customer_id,
        tier=args.tier,
        expires_at=args.expires,
        max_gpus=args.max_gpus,
        features=features,
        output_file=args.output
    )


if __name__ == "__main__":
    main()
