#!/usr/bin/env python3
"""Retrain Claire Banner face profile with cleaned training data."""
import sys
sys.path.insert(0, '/app/src')

from webapp import retrain_face_profile
import asyncio

async def main():
    await retrain_face_profile("Claire Banner")
    print("✓ Claire Banner face profile retrained")

if __name__ == "__main__":
    asyncio.run(main())
