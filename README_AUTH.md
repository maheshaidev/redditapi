# Reddit Video Download Authentication Fix

## Problem
Reddit now requires authentication for video downloads through yt-dlp. Without proper authentication, you'll get the error:
```
ERROR: [Reddit] 1m78mo1: Account authentication is required. Use --cookies, --cookies-from-browser, --username and --password, --netrc-cmd, or --netrc (reddit) to provide account credentials.
```

## Solutions

### Option 1: Using Cookies (Recommended)

1. **Extract Reddit cookies** from your browser:
   - Install a browser extension like "Get cookies.txt LOCALLY" 
   - Go to reddit.com and make sure you're logged in
   - Use the extension to export cookies to a file named `cookies.txt`

2. **Place the cookies file** in your project directory

3. **Update Docker Compose** with the cookies file:
   ```yaml
   volumes:
     - ./downloads:/app/downloads
     - ./temp:/app/temp
     - ./cookies.txt:/app/cookies.txt:ro  # Add this line
   ```

### Option 2: Using Username/Password

1. **Set environment variables** in your Docker Compose or deployment:
   ```bash
   export REDDIT_USERNAME=your_reddit_username
   export REDDIT_PASSWORD=your_reddit_password
   ```

2. **Update Docker Compose** environment section:
   ```yaml
   environment:
     - REDDIT_USERNAME=your_reddit_username
     - REDDIT_PASSWORD=your_reddit_password
   ```

### Option 3: Using Browser Cookies (Alternative)

You can also use cookies directly from your browser:

1. **For Chrome**: 
   ```bash
   export REDDIT_COOKIES_FROM_BROWSER=chrome
   ```

2. **For Firefox**:
   ```bash
   export REDDIT_COOKIES_FROM_BROWSER=firefox
   ```

## Implementation Steps

1. **Choose your authentication method** (cookies recommended)
2. **Update your deployment configuration**
3. **Rebuild and redeploy** your application
4. **Test** with a Reddit video URL

## Example Docker Compose with Cookies

```yaml
version: '3.8'

services:
  redviddown-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./downloads:/app/downloads
      - ./temp:/app/temp
      - ./cookies.txt:/app/cookies.txt:ro
    environment:
      - REDDIT_CLIENT_ID=9P4xvnzTHN_SJBmUTtlR3g
      - REDDIT_CLIENT_SECRET=vMKz8Hl0LgN19ju2fOI72ojckp80VA
      - REDDIT_USER_AGENT=redviddown:v1.0.0 (by /u/redviddown)
      - REDDIT_COOKIES_FILE=/app/cookies.txt
    restart: unless-stopped
```

## Security Notes

- **Never commit credentials** to your repository
- **Use environment variables** for sensitive data
- **Restrict file permissions** on cookies.txt (chmod 600)
- **Regularly rotate** your Reddit credentials

## Troubleshooting

1. **Check authentication**: Ensure your cookies/credentials are valid
2. **Verify file paths**: Make sure cookies.txt is accessible in the container
3. **Test manually**: Try yt-dlp command with same credentials locally
4. **Check logs**: Look for authentication-related error messages
