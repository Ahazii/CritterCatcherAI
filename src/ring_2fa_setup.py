#!/usr/bin/env python3
"""
Ring 2FA Authentication Helper
Run this script interactively to set up 2FA authentication with Ring.
"""
import os
import sys
from ring_downloader import RingDownloader

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
    
    # Initialize downloader
    rd = RingDownloader(
        download_path="/tmp",
        token_file="/data/ring_token.json"
    )
    
    # Attempt authentication
    try:
        if rd.authenticate(username, password):
            print()
            print("=" * 80)
            print("SUCCESS! Authentication completed.")
            print("=" * 80)
            print()
            print("Your Ring token has been saved to: /data/ring_token.json")
            print("You can now start the CritterCatcherAI container normally.")
            print()
            sys.exit(0)
        else:
            print()
            print("Authentication failed. Please check your credentials.")
            sys.exit(1)
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
        print("  2. You have internet connectivity")
        print("  3. Your Ring account is active")
        sys.exit(1)

if __name__ == "__main__":
    main()
