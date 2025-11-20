#!/bin/bash
# CritterCatcherAI Deployment Verification Script
# Run this on your Unraid server: bash verify_deployment.sh

echo "================================"
echo "CritterCatcherAI Deployment Check"
echo "================================"
echo ""

# 1. Check if container is running
echo "1. Checking if container is running..."
if docker ps | grep -q crittercatcher; then
    echo "   ✅ Container is RUNNING"
    docker ps | grep crittercatcher
else
    echo "   ❌ Container is NOT running"
    exit 1
fi
echo ""

# 2. Check container logs for errors
echo "2. Checking recent logs (last 20 lines)..."
docker logs crittercatcher-ai --tail=20 2>&1 | head -20
echo ""

# 3. Check data directory
echo "3. Checking data directory structure..."
if [ -d "/mnt/user/data/CritterCatcher" ]; then
    echo "   ✅ Data directory exists"
    ls -la /mnt/user/data/CritterCatcher/ | head -10
else
    echo "   ❌ Data directory NOT found at /mnt/user/data/CritterCatcher/"
fi
echo ""

# 4. Test API endpoint
echo "4. Testing API endpoint..."
curl -s http://localhost:8089/api/animal-profiles | head -c 200
echo ""
echo ""

# 5. Test web UI
echo "5. Testing web UI accessibility..."
curl -s -o /dev/null -w "HTTP Status: %{http_code}\n" http://localhost:8089/
echo ""

echo "================================"
echo "Verification Complete"
echo "================================"
