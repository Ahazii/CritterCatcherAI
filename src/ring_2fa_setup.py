#!/usr/bin/env python3
"""
Ring 2FA Authentication Helper
Run this script interactively to set up 2FA authentication with Ring.
"""
import os
import sys
import json
from pathlib import Path
from ring_doorbell import Auth, Ring

def main():
    print("=" * 80)
    print("CritterCatcherAI - Ring 2FA Setup")
    print("=" * 80)
    print()
    
    # Get credentials from environment or prompt
    username = os.environ.get('RING_USERNAME')
    password = os.environ.get('RING_PASSWORD')
    
    if not username:
        username = input("Enter your Ring email address: ").strip()
    
    if not password:
        import getpass
        password = getpass.getpass("Enter your Ring password: ")
    
    print()
    print(f"Authenticating as: {username}")
    print("If you have 2FA enabled, you'll be prompted to enter the code...")
    print()
    
    token_file = Path("/data/tokens/ring_token.json")
    
    # Attempt authentication
    try:
        auth = Auth("CritterCatcherAI/1.0")
        
        # Try to fetch token - if 2FA is needed, it will raise an exception
        try:
            auth.fetch_token(username, password)
        except Exception as e:
            # If 2FA is required, ask for code
            if "2fa" in str(e).lower() or "tsv_state" in str(e).lower():
                code = input("\nEnter 2FA code (check SMS/email/app): ")
                auth.fetch_token(username, password, code)
            else:
                raise
        
        # Save token
        token_file.parent.mkdir(parents=True, exist_ok=True)
        with open(token_file, 'w') as f:
            json.dump(auth.token, f)
        
        # Verify by creating Ring instance
        ring = Ring(auth)
        ring.update_data()
        
        print()
        print("=" * 80)
        print("SUCCESS! Authentication completed.")
        print("=" * 80)
        print()
        print(f"Your Ring token has been saved to: {token_file}")
        print("You can now start the CritterCatcherAI container normally.")
        print()
        sys.exit(0)
        
    except KeyboardInterrupt:
        print()
        print("Authentication cancelled.")
        sys.exit(1)
    except Exception as e:
        print()
        print(f"Error during authentication: {e}")
        print()
        print("Please ensure:")
        print("  1. Your Ring username and password are correct")
        print("  2. You entered the correct 2FA code")
        print("  3. You have internet connectivity")
        print("  4. Your Ring account is active")
        sys.exit(1)

if __name__ == "__main__":
    main()
