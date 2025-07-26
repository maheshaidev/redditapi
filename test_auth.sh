#!/bin/bash

# Test script for Reddit authentication

echo "Testing Reddit video download authentication..."

# Test if cookies file exists
if [ -f "cookies.txt" ]; then
    echo "✓ cookies.txt found"
    # Check if cookies file has content
    if [ -s "cookies.txt" ]; then
        echo "✓ cookies.txt has content"
    else
        echo "✗ cookies.txt is empty"
    fi
else
    echo "✗ cookies.txt not found"
fi

# Test environment variables
if [ -n "$REDDIT_USERNAME" ] && [ -n "$REDDIT_PASSWORD" ]; then
    echo "✓ Reddit username/password environment variables set"
elif [ -n "$REDDIT_USERNAME" ]; then
    echo "✗ Reddit username set but password missing"
elif [ -n "$REDDIT_PASSWORD" ]; then
    echo "✗ Reddit password set but username missing"
else
    echo "✗ No Reddit username/password environment variables"
fi

# Test yt-dlp directly (if available)
if command -v yt-dlp &> /dev/null; then
    echo "Testing yt-dlp with a sample Reddit video..."
    
    # Test URL (replace with actual Reddit video URL)
    TEST_URL="https://www.reddit.com/r/videos/comments/sample/sample_video/"
    
    if [ -f "cookies.txt" ]; then
        echo "Testing with cookies..."
        yt-dlp --cookies cookies.txt --no-download --get-title "$TEST_URL" 2>&1
    elif [ -n "$REDDIT_USERNAME" ] && [ -n "$REDDIT_PASSWORD" ]; then
        echo "Testing with username/password..."
        yt-dlp --username "$REDDIT_USERNAME" --password "$REDDIT_PASSWORD" --no-download --get-title "$TEST_URL" 2>&1
    else
        echo "No authentication method available for testing"
    fi
else
    echo "yt-dlp not found - install it to test authentication"
fi

echo "Authentication test complete."
