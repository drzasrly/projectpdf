
# src/scrape_youtube_comments.py

import csv
import time
from googleapiclient.discovery import build
from langdetect import detect

# ===========================
# Konfigurasi
# ===========================
API_KEY = "AIzaSyCn45DbRGlK94FUGSMInHrhRnS1YZbH5zI"  # ganti dengan API key kamu
SEARCH_KEYWORD = "burger"
MAX_VIDEOS = 5       # jumlah video untuk di-scrape
MAX_COMMENTS = 1000  # total komentar per video
OUTPUT_CSV = "komentar_burger.csv"

youtube = build("youtube", "v3", developerKey=API_KEY)

# ===========================
# Fungsi Ambil Video ID
# ===========================
def get_video_ids(keyword, max_videos=5):
    video_ids = []
    request = youtube.search().list(
        part="id",
        q=keyword,
        type="video",
        maxResults=max_videos
    )
    response = request.execute()
    
    for item in response['items']:
        video_ids.append(item['id']['videoId'])
    return video_ids

# ===========================
# Fungsi Ambil Komentar
# ===========================
def get_comments(video_id, max_results=1000):
    comments = []
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100,  # max per request
        textFormat="plainText"
    )
    while request and len(comments) < max_results:
        response = request.execute()
        for item in response['items']:
            text = item['snippet']['topLevelComment']['snippet']['textDisplay']
            # hanya ambil komentar bahasa Indonesia
            try:
                if detect(text) == "id":
                    comments.append(text)
            except:
                continue
            if len(comments) >= max_results:
                break
        request = youtube.commentThreads().list_next(request, response)
        time.sleep(1)  # untuk menghindari rate limit
    return comments

# ===========================
# Main
# ===========================
all_comments = []

print(f"🔎 Mencari {MAX_VIDEOS} video untuk keyword: '{SEARCH_KEYWORD}'")
video_ids = get_video_ids(SEARCH_KEYWORD, MAX_VIDEOS)
print("Video ID ditemukan:", video_ids)

for vid in video_ids:
    print(f"💬 Mengambil komentar dari video {vid} ...")
    comments = get_comments(vid, MAX_COMMENTS)
    for c in comments:
        all_comments.append({"comment": c})
    print(f"✅ Selesai ambil {len(comments)} komentar dari video {vid}")

# ===========================
# Simpan ke CSV
# ===========================
keys = ["comment"]
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    dict_writer = csv.DictWriter(f, keys)
    dict_writer.writeheader()
    dict_writer.writerows(all_comments)

print(f"📂 Semua komentar berhasil disimpan di '{OUTPUT_CSV}' (total {len(all_comments)} komentar)")