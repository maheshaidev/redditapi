import os
import asyncio
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any
import uuid
import json
import time
import shutil
from urllib.parse import urlparse, parse_qs

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import yt_dlp
import requests
from PIL import Image
import aiofiles

# Initialize FastAPI app
app = FastAPI(title="Redviddown API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Reddit API credentials
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "9P4xvnzTHN_SJBmUTtlR3g")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "vMKz8Hl0LgN19ju2fOI72ojckp80VA")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "redviddown:v1.0.0 (by /u/redviddown)")

# Reddit authentication for yt-dlp
REDDIT_USERNAME = os.getenv("REDDIT_USERNAME")
REDDIT_PASSWORD = os.getenv("REDDIT_PASSWORD")
REDDIT_COOKIES_FILE = os.getenv("REDDIT_COOKIES_FILE", "/app/cookies.txt")

# Directory setup
BASE_DIR = Path(__file__).parent
TEMP_DIR = BASE_DIR / "temp"
DOWNLOADS_DIR = BASE_DIR / "downloads"
TEMP_DIR.mkdir(exist_ok=True)
DOWNLOADS_DIR.mkdir(exist_ok=True)

# Mount static files
app.mount("/downloads", StaticFiles(directory=DOWNLOADS_DIR), name="downloads")

# Pydantic models
class VideoInfoRequest(BaseModel):
    url: str

class DownloadRequest(BaseModel):
    url: str
    quality: str = "720"

class VideoInfo(BaseModel):
    title: str
    author: str
    thumbnail: str
    duration: Optional[str] = None
    qualities: list[str]
    contentType: str = "video"

class DownloadResult(BaseModel):
    title: str
    fileName: str
    fileUrl: str
    quality: str
    hasAudio: bool
    fileSizeFormatted: str

# Reddit API helper
class RedditAPI:
    def __init__(self):
        self.access_token = None
        self.token_expires = 0
    
    async def get_access_token(self):
        """Get Reddit OAuth access token"""
        if self.access_token and time.time() < self.token_expires:
            return self.access_token
        
        auth = requests.auth.HTTPBasicAuth(REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET)
        data = {
            'grant_type': 'client_credentials'
        }
        headers = {'User-Agent': REDDIT_USER_AGENT}
        
        response = requests.post(
            'https://www.reddit.com/api/v1/access_token',
            auth=auth,
            data=data,
            headers=headers
        )
        
        if response.status_code == 200:
            token_data = response.json()
            self.access_token = token_data['access_token']
            self.token_expires = time.time() + token_data['expires_in'] - 60
            return self.access_token
        else:
            raise HTTPException(status_code=500, detail="Failed to get Reddit access token")
    
    async def get_post_info(self, post_id: str, subreddit: str = None):
        """Get Reddit post information using OAuth API"""
        token = await self.get_access_token()
        headers = {
            'Authorization': f'Bearer {token}',
            'User-Agent': REDDIT_USER_AGENT
        }
        
        if subreddit:
            url = f'https://oauth.reddit.com/r/{subreddit}/comments/{post_id}'
        else:
            # Try to get post info directly
            url = f'https://oauth.reddit.com/comments/{post_id}'
        
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=404, detail="Reddit post not found")
    
    async def get_video_urls_from_json(self, reddit_url: str):
        """Get video URLs directly from Reddit JSON endpoint"""
        try:
            # Convert Reddit URL to JSON endpoint
            if reddit_url.endswith('/'):
                json_url = reddit_url[:-1] + '.json'
            else:
                json_url = reddit_url + '.json'
            
            headers = {'User-Agent': REDDIT_USER_AGENT}
            response = requests.get(json_url, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                
                if data and len(data) > 0 and 'data' in data[0]:
                    post_data = data[0]['data']['children'][0]['data']
                    
                    # Extract video information
                    video_info = {
                        'title': post_data.get('title', 'Reddit Video'),
                        'author': post_data.get('author', 'Unknown'),
                        'thumbnail': post_data.get('thumbnail', ''),
                        'is_video': post_data.get('is_video', False),
                        'video_urls': {},
                        'audio_url': None
                    }
                    
                    # Check for video in media
                    media = post_data.get('media')
                    if media and 'reddit_video' in media:
                        reddit_video = media['reddit_video']
                        
                        # Get video URL (usually DASH format without audio)
                        video_url = reddit_video.get('fallback_url')
                        if video_url:
                            video_info['video_urls']['fallback'] = video_url
                            
                            # Audio URL is typically the video URL with DASH_audio appended
                            if 'DASH_' in video_url:
                                audio_url = video_url.replace('DASH_', 'DASH_audio_')
                                video_info['audio_url'] = audio_url
                        
                        # Get HLS URL if available
                        hls_url = reddit_video.get('hls_url')
                        if hls_url:
                            video_info['video_urls']['hls'] = hls_url
                        
                        video_info['duration'] = reddit_video.get('duration', 0)
                        video_info['height'] = reddit_video.get('height', 720)
                        video_info['width'] = reddit_video.get('width', 1280)
                    
                    # Check for secure media (newer Reddit posts)
                    secure_media = post_data.get('secure_media')
                    if secure_media and 'reddit_video' in secure_media:
                        reddit_video = secure_media['reddit_video']
                        
                        video_url = reddit_video.get('fallback_url')
                        if video_url:
                            video_info['video_urls']['secure_fallback'] = video_url
                            
                            if 'DASH_' in video_url:
                                audio_url = video_url.replace('DASH_', 'DASH_audio_')
                                video_info['audio_url'] = audio_url
                    
                    return video_info
            
            return None
            
        except Exception as e:
            print(f"Error getting video URLs from JSON: {e}")
            return None

reddit_api = RedditAPI()

def get_ytdlp_options_with_auth(base_opts: dict = None) -> dict:
    """Get yt-dlp options with Reddit authentication"""
    opts = base_opts.copy() if base_opts else {}
    
    # Add authentication options
    if os.path.exists(REDDIT_COOKIES_FILE):
        opts['cookiefile'] = REDDIT_COOKIES_FILE
        print(f"Using cookies file: {REDDIT_COOKIES_FILE}")
    elif REDDIT_USERNAME and REDDIT_PASSWORD:
        opts['username'] = REDDIT_USERNAME
        opts['password'] = REDDIT_PASSWORD
        print("Using username/password authentication")
    else:
        print("Warning: No Reddit authentication configured. This may cause download failures.")
    
    # Add user agent for better compatibility
    opts['http_headers'] = opts.get('http_headers', {})
    opts['http_headers']['User-Agent'] = REDDIT_USER_AGENT
    
    return opts

def extract_post_info_from_url(reddit_url: str) -> tuple[str, str]:
    """Extract post ID and subreddit from Reddit URL"""
    parsed = urlparse(reddit_url)
    path_parts = parsed.path.strip('/').split('/')
    
    # Handle different Reddit URL formats
    if 'comments' in path_parts:
        comment_index = path_parts.index('comments')
        if comment_index + 1 < len(path_parts):
            post_id = path_parts[comment_index + 1]
            if 'r' in path_parts and path_parts.index('r') + 1 < len(path_parts):
                subreddit = path_parts[path_parts.index('r') + 1]
                return post_id, subreddit
            return post_id, None
    
    # Handle v.redd.it URLs
    if 'v.redd.it' in parsed.netloc:
        post_id = path_parts[0] if path_parts else None
        return post_id, None
    
    raise HTTPException(status_code=400, detail="Invalid Reddit URL format")

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    size = float(size_bytes)
    
    while size >= 1024.0 and i < len(size_names) - 1:
        size /= 1024.0
        i += 1
    
    return f"{size:.1f} {size_names[i]}"

def clean_filename(filename: str) -> str:
    """Clean filename for safe storage"""
    # Remove or replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Limit length
    if len(filename) > 200:
        filename = filename[:200]
    
    return filename

def build_format_selector(formats, max_quality):
    """Build a format selector for yt-dlp based on available formats and requested quality"""
    print(f"Building format selector for max quality: {max_quality}")
    print(f"Available formats: {len(formats)}")
    
    # Convert max_quality to reasonable range if it seems off
    # Handle cases where user selects a standardized quality
    if max_quality >= 2160:
        target_height = 2160
    elif max_quality >= 1440:
        target_height = 1440  
    elif max_quality >= 1080:
        target_height = 1080
    elif max_quality >= 720:
        target_height = 720
    elif max_quality >= 480:
        target_height = 480
    elif max_quality >= 360:
        target_height = 360
    else:
        target_height = 240
    
    print(f"Target height for quality {max_quality}: {target_height}")
    
    # First, try to find formats with both video and audio
    video_audio_formats = []
    video_only_formats = []
    audio_only_formats = []
    
    for fmt in formats:
        height = fmt.get('height')
        vcodec = fmt.get('vcodec', 'none')
        acodec = fmt.get('acodec', 'none')
        
        if height and vcodec != 'none' and acodec != 'none':
            # Format with both video and audio
            if int(height) <= target_height * 1.2:  # Allow 20% tolerance
                video_audio_formats.append(fmt)
        elif height and vcodec != 'none' and acodec == 'none':
            # Video-only format
            if int(height) <= target_height * 1.2:  # Allow 20% tolerance
                video_only_formats.append(fmt)
        elif vcodec == 'none' and acodec != 'none':
            # Audio-only format
            audio_only_formats.append(fmt)
    
    # Sort by quality (closest to target first, then highest)
    video_audio_formats.sort(key=lambda x: (abs(int(x['height']) - target_height), -int(x['height'])))
    video_only_formats.sort(key=lambda x: (abs(int(x['height']) - target_height), -int(x['height'])))
    
    # Build format selector string
    if video_audio_formats:
        # Best case: format with both video and audio
        return f"{video_audio_formats[0]['format_id']}"
    elif video_only_formats and audio_only_formats:
        # Combine best video with best audio
        best_video = video_only_formats[0]['format_id']
        best_audio = audio_only_formats[0]['format_id']
        return f"{best_video}+{best_audio}"
    elif video_only_formats:
        # Video only
        return f"{video_only_formats[0]['format_id']}"
    else:
        # Fallback to generic selectors
        quality_selectors = [
            f"best[height<={target_height}]",
            f"bestvideo[height<={target_height}]+bestaudio",
            "best"
        ]
        return "/".join(quality_selectors)

async def download_from_direct_urls(video_info: dict, temp_dir: Path, quality: str) -> Path:
    """Download video using direct URLs from Reddit JSON"""
    try:
        video_urls = video_info.get('video_urls', {})
        audio_url = video_info.get('audio_url')
        title = clean_filename(video_info.get('title', 'Reddit Video'))
        
        if not video_urls:
            raise Exception("No video URLs found")
        
        # Choose the best video URL
        video_url = None
        if 'secure_fallback' in video_urls:
            video_url = video_urls['secure_fallback']
        elif 'fallback' in video_urls:
            video_url = video_urls['fallback']
        elif 'hls' in video_urls:
            video_url = video_urls['hls']
        
        if not video_url:
            raise Exception("No suitable video URL found")
        
        print(f"Downloading video from: {video_url}")
        
        # Try to find better quality video URL if available
        selected_quality = int(quality)
        if selected_quality and "DASH_" in video_url:
            try:
                # Parse base URL
                base_url = video_url.rsplit('/', 1)[0]
                # Extract query parameters if any
                query_params = ""
                if "?" in video_url:
                    query_params = "?" + video_url.split('?', 1)[1]
                
                # Try different quality versions
                quality_options = [1080, 720, 480, 360, 240]
                # Find closest available quality (start with requested or higher)
                available_qualities = [q for q in quality_options if q <= selected_quality]
                if not available_qualities:
                    available_qualities = quality_options
                
                for test_quality in available_qualities:
                    test_url = f"{base_url}/DASH_{test_quality}.mp4{query_params}"
                    try:
                        head_response = requests.head(test_url, 
                                                    headers={'User-Agent': REDDIT_USER_AGENT},
                                                    timeout=5)
                        if head_response.status_code == 200:
                            video_url = test_url
                            print(f"Found better quality video: {test_quality}p")
                            break
                    except Exception as e:
                        print(f"Failed to check quality {test_quality}p: {e}")
            except Exception as e:
                print(f"Error while trying to find better quality: {e}")
        
        # Download video
        headers = {'User-Agent': REDDIT_USER_AGENT}
        video_response = requests.get(video_url, headers=headers, stream=True)
        
        if video_response.status_code != 200:
            raise Exception(f"Failed to download video: HTTP {video_response.status_code}")
        
        video_file = temp_dir / f"{title}_video.mp4"
        with open(video_file, 'wb') as f:
            for chunk in video_response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Try to download audio if available
        audio_file = None
        if audio_url:
            try:
                print(f"Downloading audio from: {audio_url}")
                audio_response = requests.get(audio_url, headers=headers, stream=True)
                
                if audio_response.status_code == 200:
                    audio_file = temp_dir / f"{title}_audio.mp4"
                    with open(audio_file, 'wb') as f:
                        for chunk in audio_response.iter_content(chunk_size=8192):
                            f.write(chunk)
                else:
                    print(f"Audio download failed: HTTP {audio_response.status_code}")
                    
                    # Try alternative audio URL formats if the original fails
                    if "DASH_" in video_url:
                        # Try different audio URL patterns
                        base_url = video_url.rsplit('/', 1)[0]
                        audio_urls_to_try = [
                            f"{base_url}/DASH_audio.mp4",
                            f"{base_url}/audio",
                            f"{base_url}/DASH_AUDIO_128.mp4",
                            f"{base_url}/DASH_AUDIO_64.mp4"
                        ]
                        
                        for alt_audio_url in audio_urls_to_try:
                            try:
                                print(f"Trying alternative audio URL: {alt_audio_url}")
                                alt_response = requests.get(alt_audio_url, headers=headers, stream=True)
                                if alt_response.status_code == 200:
                                    audio_file = temp_dir / f"{title}_audio.mp4"
                                    with open(audio_file, 'wb') as f:
                                        for chunk in alt_response.iter_content(chunk_size=8192):
                                            f.write(chunk)
                                    if os.path.getsize(audio_file) > 1024:  # At least 1KB
                                        print(f"Successfully downloaded audio from alternative URL")
                                        break
                            except Exception as e:
                                print(f"Alternative audio URL failed: {e}")
            except Exception as e:
                print(f"Audio download error: {e}")
        
        # Merge video and audio if both exist
        if audio_file and audio_file.exists() and video_file.exists():
            try:
                merged_file = temp_dir / f"{title}_merged.mp4"
                
                # Use ffmpeg to merge video and audio
                merge_cmd = [
                    'ffmpeg', '-i', str(video_file), '-i', str(audio_file),
                    '-c:v', 'copy', '-c:a', 'aac', '-strict', 'experimental',
                    str(merged_file), '-y'
                ]
                
                result = subprocess.run(merge_cmd, capture_output=True, text=True)
                
                if result.returncode == 0 and merged_file.exists():
                    print("Successfully merged video and audio")
                    return merged_file
                else:
                    print(f"Merge failed: {result.stderr}")
                    # Try alternate merge command
                    alt_merge_cmd = [
                        'ffmpeg', '-y',
                        '-i', str(video_file),
                        '-i', str(audio_file),
                        '-c', 'copy',
                        str(merged_file)
                    ]
                    alt_result = subprocess.run(alt_merge_cmd, capture_output=True, text=True)
                    if alt_result.returncode == 0 and merged_file.exists():
                        print("Successfully merged video and audio with alternate command")
                        return merged_file
                    else:
                        return video_file
            except Exception as e:
                print(f"Merge error: {e}")
                return video_file
        else:
            return video_file
            
    except Exception as e:
        print(f"Direct download error: {e}")
        raise
    """Download and save thumbnail image"""
    try:
        response = requests.get(thumbnail_url, timeout=10)
        if response.status_code == 200:
            # Save thumbnail
            thumbnail_path = TEMP_DIR / f"thumb_{post_id}.jpg"
            with open(thumbnail_path, 'wb') as f:
                f.write(response.content)
            
            # Optimize thumbnail
            try:
                with Image.open(thumbnail_path) as img:
                    # Resize if too large
                    if img.width > 800 or img.height > 600:
                        img.thumbnail((800, 600), Image.Resampling.LANCZOS)
                    
                    # Convert to RGB if necessary
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Save optimized version
                    img.save(thumbnail_path, 'JPEG', quality=85, optimize=True)
            except Exception as e:
                print(f"Thumbnail optimization failed: {e}")
            
            return str(thumbnail_path)
    except Exception as e:
        print(f"Thumbnail download failed: {e}")
    
    return None

@app.get("/")
async def root():
    return {"message": "Redviddown API is running", "version": "1.0.0"}

@app.get("/test-connection")
async def test_connection():
    """Test API connection endpoint"""
    return {"status": "connected", "message": "Backend API is running"}

@app.post("/debug-formats")
async def debug_formats(request: VideoInfoRequest):
    """Debug endpoint to list all available formats for a Reddit URL"""
    try:
        ydl_opts = get_ytdlp_options_with_auth({
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
        })
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(request.url, download=False)
            
            formats = info.get('formats', [])
            
            debug_info = {
                'title': info.get('title', 'Unknown'),
                'uploader': info.get('uploader', 'Unknown'),
                'duration': info.get('duration'),
                'total_formats': len(formats),
                'formats': []
            }
            
            for fmt in formats:
                debug_info['formats'].append({
                    'format_id': fmt.get('format_id'),
                    'ext': fmt.get('ext'),
                    'width': fmt.get('width'),
                    'height': fmt.get('height'),
                    'fps': fmt.get('fps'),
                    'vcodec': fmt.get('vcodec'),
                    'acodec': fmt.get('acodec'),
                    'filesize': fmt.get('filesize'),
                    'url': fmt.get('url', 'N/A')[:100] + '...' if fmt.get('url') else 'N/A'
                })
            
            return debug_info
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Debug failed: {str(e)}")

@app.post("/video-info")
async def get_video_info(request: VideoInfoRequest):
    """Get video information from Reddit URL"""
    try:
        # Try JSON endpoint first (no authentication needed)
        video_data = await reddit_api.get_video_urls_from_json(request.url)
        
        if video_data and video_data.get('video_urls'):
            # Determine available qualities based on video dimensions
            height = video_data.get('height', 720)
            qualities = []
            
            if height >= 1080:
                qualities = ["1080", "720", "480", "360"]
            elif height >= 720:
                qualities = ["720", "480", "360"]
            else:
                qualities = ["480", "360", "240"]
            
            return VideoInfo(
                title=video_data.get('title', 'Reddit Video'),
                author=video_data.get('author', 'Unknown'),
                thumbnail=video_data.get('thumbnail', ''),
                duration=str(video_data.get('duration', '')),
                qualities=qualities,
                contentType="video"
            )
        
        # Fallback to existing methods if JSON doesn't work
        post_id, subreddit = extract_post_info_from_url(request.url)
        
        # Create yt-dlp options for info extraction only
        ydl_opts = get_ytdlp_options_with_auth({
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
        })
        
        video_info = None
        
        # Try to extract info using yt-dlp first
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(request.url, download=False)
                
                if info:
                    # Get available qualities
                    qualities = []
                    formats = info.get('formats', [])
                    
                    for fmt in formats:
                        height = fmt.get('height')
                        if height and height > 0:
                            # Map actual heights to standard quality labels
                            if height >= 2160:
                                quality_label = "2160"  # 4K
                            elif height >= 1440:
                                quality_label = "1440"  # 2K
                            elif height >= 1080:
                                quality_label = "1080"  # Full HD
                            elif height >= 720:
                                quality_label = "720"   # HD
                            elif height >= 480:
                                quality_label = "480"   # SD
                            elif height >= 360:
                                quality_label = "360"   # Low
                            else:
                                quality_label = "240"   # Very Low
                            
                            if quality_label not in qualities:
                                qualities.append(quality_label)
                    
                    # Sort qualities (highest first, then convert to standard order)
                    qualities = sorted([int(q) for q in qualities if q.isdigit()], reverse=True)
                    qualities = [str(q) for q in qualities]
                    
                    if not qualities:
                        qualities = ["720", "480", "360", "240"]
                    
                    # Determine content type
                    content_type = "video"
                    if info.get('ext') == 'gif' or 'gif' in info.get('title', '').lower():
                        content_type = "gif"
                    
                    video_info = VideoInfo(
                        title=info.get('title', 'Reddit Video'),
                        author=info.get('uploader', 'Unknown'),
                        thumbnail=info.get('thumbnail', ''),
                        duration=str(info.get('duration', '')),
                        qualities=qualities,
                        contentType=content_type
                    )
        
        except Exception as e:
            print(f"yt-dlp extraction failed: {e}")
        
        # Fallback to Reddit API if yt-dlp fails
        if not video_info:
            try:
                reddit_data = await reddit_api.get_post_info(post_id, subreddit)
                
                if reddit_data and len(reddit_data) > 0:
                    post_data = reddit_data[0]['data']['children'][0]['data']
                    
                    title = post_data.get('title', 'Reddit Video')
                    author = post_data.get('author', 'Unknown')
                    thumbnail = post_data.get('thumbnail', '')
                    
                    # Check if it's a video post
                    is_video = post_data.get('is_video', False)
                    media = post_data.get('media', {})
                    
                    content_type = "video"
                    if not is_video and media is None:
                        content_type = "gif"
                    
                    video_info = VideoInfo(
                        title=title,
                        author=author,
                        thumbnail=thumbnail,
                        duration="",
                        qualities=["720", "480", "360", "240"],  # Default qualities
                        contentType=content_type
                    )
            
            except Exception as e:
                print(f"Reddit API fallback failed: {e}")
        
        # Final fallback with basic info
        if not video_info:
            video_info = VideoInfo(
                title="Reddit Video",
                author="Unknown",
                thumbnail="",
                duration="",
                qualities=["720", "480", "360", "240"],
                contentType="video"
            )
        
        return video_info
    
    except Exception as e:
        print(f"Error getting video info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get video information: {str(e)}")

@app.post("/download")
async def download_video(request: DownloadRequest, background_tasks: BackgroundTasks):
    """Download Reddit video with specified quality"""
    try:
        # Generate unique filename
        download_id = str(uuid.uuid4())
        temp_dir = TEMP_DIR / download_id
        temp_dir.mkdir(exist_ok=True)
        
        # Try JSON endpoint first (no authentication needed)
        video_data = await reddit_api.get_video_urls_from_json(request.url)
        
        downloaded_file = None
        video_title = "Reddit Video"
        has_audio = False
        
        if video_data and video_data.get('video_urls'):
            print("Using direct JSON download method")
            try:
                downloaded_file = await download_from_direct_urls(video_data, temp_dir, request.quality)
                video_title = clean_filename(video_data.get('title', 'Reddit Video'))
                
                # Check if the file has audio
                try:
                    probe_cmd = [
                        'ffprobe', '-v', 'quiet', '-select_streams', 'a',
                        '-show_entries', 'stream=codec_name', '-of', 'csv=p=0',
                        str(downloaded_file)
                    ]
                    audio_result = subprocess.run(probe_cmd, capture_output=True, text=True)
                    has_audio = bool(audio_result.stdout.strip())
                except Exception as e:
                    print(f"Audio check failed: {e}")
                    has_audio = video_data.get('audio_url') is not None
            except Exception as e:
                print(f"Direct download failed: {e}")
                downloaded_file = None
        
        # Fallback to yt-dlp if direct download fails
        if not downloaded_file:
            print("Falling back to yt-dlp method")
            
            # Get available formats first
            info_opts = get_ytdlp_options_with_auth({
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False,
                # Skip authentication requirement
                'extractor_args': {
                    'reddit': {
                        'skip_auth': True
                    }
                }
            })
            
            # Try to extract info without downloading
            try:
                with yt_dlp.YoutubeDL(info_opts) as ydl_info:
                    info = ydl_info.extract_info(request.url, download=False)
                    video_title = clean_filename(info.get('title', 'Reddit Video'))
                    
                    # Build format selector based on available formats
                    formats = info.get('formats', [])
                    format_selector = build_format_selector(formats, int(request.quality))
                    
                    # yt-dlp options for download
                    ydl_opts = get_ytdlp_options_with_auth({
                        'format': format_selector,
                        'outtmpl': str(temp_dir / '%(title)s.%(ext)s'),
                        'writesubtitles': False,
                        'writeautomaticsub': False,
                        'quiet': False,
                        'no_warnings': False,
                        # Skip authentication requirement
                        'extractor_args': {
                            'reddit': {
                                'skip_auth': True
                            }
                        }
                    })
                    
                    # Try to download
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        ydl.download([request.url])
                        
                        # Find the downloaded file
                        for file_path in temp_dir.glob('*'):
                            if file_path.is_file() and file_path.suffix.lower() in ['.mp4', '.webm', '.mkv', '.avi', '.mov']:
                                downloaded_file = file_path
                                break
            except Exception as info_error:
                print(f"yt-dlp info extraction failed: {info_error}")
                
                # Try direct download with more aggressive options
                try:
                    # Try to parse video ID from URL
                    parsed_url = urlparse(request.url)
                    path_parts = parsed_url.path.strip('/').split('/')
                    post_id = None
                    
                    if 'comments' in path_parts and len(path_parts) > path_parts.index('comments') + 1:
                        post_id = path_parts[path_parts.index('comments') + 1]
                    elif 'v.redd.it' in parsed_url.netloc and path_parts:
                        post_id = path_parts[0]
                    
                    if post_id:
                        # Try to access Reddit JSON directly
                        json_url = f"https://www.reddit.com/comments/{post_id}/.json"
                        headers = {'User-Agent': REDDIT_USER_AGENT}
                        
                        response = requests.get(json_url, headers=headers)
                        if response.status_code == 200:
                            try:
                                data = response.json()
                                if data and len(data) > 0 and 'data' in data[0]:
                                    post_data = data[0]['data']['children'][0]['data']
                                    
                                    # Extract video URLs
                                    if post_data.get('is_video', False):
                                        media = post_data.get('media', {})
                                        if media and 'reddit_video' in media:
                                            reddit_video = media['reddit_video']
                                            video_url = reddit_video.get('fallback_url')
                                            
                                            if video_url:
                                                # Guess audio URL
                                                audio_url = None
                                                if 'DASH_' in video_url:
                                                    audio_url = video_url.replace('DASH_', 'DASH_audio_')
                                                
                                                # Create temporary structure for download_from_direct_urls
                                                direct_video_info = {
                                                    'title': post_data.get('title', 'Reddit Video'),
                                                    'video_urls': {'fallback': video_url},
                                                    'audio_url': audio_url
                                                }
                                                
                                                downloaded_file = await download_from_direct_urls(direct_video_info, temp_dir, request.quality)
                                                video_title = clean_filename(post_data.get('title', 'Reddit Video'))
                            except Exception as json_parse_error:
                                print(f"JSON parse error: {json_parse_error}")
                except Exception as direct_fallback_error:
                    print(f"Direct fallback error: {direct_fallback_error}")
                
                # As a last resort, try generic yt-dlp options
                if not downloaded_file:
                    try:
                        ydl_opts = get_ytdlp_options_with_auth({
                            'format': 'best[ext=mp4]/best',
                            'outtmpl': str(temp_dir / '%(title)s.%(ext)s'),
                            'quiet': False,
                            'no_warnings': False
                        })
                        
                        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                            ydl.download([request.url])
                            
                            # Find the downloaded file
                            for file_path in temp_dir.glob('*'):
                                if file_path.is_file() and file_path.suffix.lower() in ['.mp4', '.webm', '.mkv', '.avi', '.mov']:
                                    downloaded_file = file_path
                                    break
                    except Exception as final_error:
                        print(f"Final fallback download failed: {final_error}")
        
        if not downloaded_file or not downloaded_file.exists():
            raise HTTPException(status_code=500, detail="Failed to download video")
        
        # Prepare final filename
        final_filename = f"{video_title}_{request.quality}p.mp4"
        final_path = DOWNLOADS_DIR / f"{download_id}_{final_filename}"
        
        # Check audio streams if not already determined
        if not has_audio:
            try:
                # Get video info to check for separate audio
                probe_cmd = [
                    'ffprobe', '-v', 'quiet', '-print_format', 'json', 
                    '-show_streams', str(downloaded_file)
                ]
                
                result = subprocess.run(probe_cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    probe_data = json.loads(result.stdout)
                    streams = probe_data.get('streams', [])
                    
                    has_video = any(s.get('codec_type') == 'video' for s in streams)
                    has_audio = any(s.get('codec_type') == 'audio' for s in streams)
                    
                    if has_video and has_audio:
                        # File already has both video and audio, just copy
                        shutil.copy2(downloaded_file, final_path)
                    else:
                        # Copy anyway, already tried our best for audio
                        shutil.copy2(downloaded_file, final_path)
                else:
                    # Fallback if ffprobe fails
                    shutil.copy2(downloaded_file, final_path)
                    has_audio = True  # Assume it has audio
            except Exception as e:
                print(f"Audio check failed: {e}")
                shutil.copy2(downloaded_file, final_path)
                has_audio = True  # Assume it has audio
        else:
            # Direct download already handled audio, just copy
            shutil.copy2(downloaded_file, final_path)
        
        # Get file size
        file_size = final_path.stat().st_size
        file_size_formatted = format_file_size(file_size)
        
        # Clean up temp directory
        background_tasks.add_task(cleanup_temp_dir, temp_dir)
        
        # Return download result with proper download endpoint
        download_url = f"/download-file/{download_id}"
        
        return DownloadResult(
            title=video_title,
            fileName=final_filename,
            fileUrl=download_url,
            quality=request.quality,
            hasAudio=has_audio,
            fileSizeFormatted=file_size_formatted
        )
    
    except Exception as e:
        print(f"Download error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to download video: {str(e)}")

@app.get("/download-file/{file_id}")
async def download_file(file_id: str):
    """Direct file download endpoint with proper headers"""
    try:
        # Find the file in downloads directory
        file_path = None
        for file in DOWNLOADS_DIR.glob(f"{file_id}_*"):
            if file.is_file():
                file_path = file
                break
        
        if not file_path or not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        # Get file info
        file_size = file_path.stat().st_size
        filename = file_path.name.split('_', 1)[1] if '_' in file_path.name else file_path.name
        
        # Return file with proper download headers
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type='application/octet-stream',
            headers={
                "Content-Disposition": f"attachment; filename=\"{filename}\"",
                "Content-Length": str(file_size),
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Expose-Headers": "Content-Disposition"
            }
        )
    except Exception as e:
        print(f"File download error: {e}")
        raise HTTPException(status_code=500, detail="Failed to download file")

@app.get("/thumbnail-proxy")
async def thumbnail_proxy(url: str):
    """Proxy for Reddit thumbnails to avoid CORS issues"""
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return Response(
                content=response.content,
                media_type=response.headers.get('content-type', 'image/jpeg'),
                headers={
                    "Cache-Control": "public, max-age=3600",
                    "Access-Control-Allow-Origin": "*"
                }
            )
    except Exception as e:
        print(f"Thumbnail proxy error: {e}")
    
    raise HTTPException(status_code=404, detail="Thumbnail not found")

async def cleanup_temp_dir(temp_dir: Path):
    """Clean up temporary directory"""
    try:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
    except Exception as e:
        print(f"Cleanup failed: {e}")

@app.on_event("startup")
async def startup_event():
    """Startup tasks"""
    print("Redviddown API starting up...")
    
    # Test Reddit API connection
    try:
        await reddit_api.get_access_token()
        print("Reddit API connection successful")
    except Exception as e:
        print(f"Reddit API connection failed: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("Redviddown API shutting down...")
    
    # Clean up temp files
    try:
        for temp_file in TEMP_DIR.glob('*'):
            if temp_file.is_file():
                temp_file.unlink()
            elif temp_file.is_dir():
                shutil.rmtree(temp_file)
    except Exception as e:
        print(f"Cleanup failed: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
