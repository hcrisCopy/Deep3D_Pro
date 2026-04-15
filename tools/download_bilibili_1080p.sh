#!/bin/bash
# Bilibili 1080p 纯视频下载脚本
# 视频BV号: BV19T4y1J7v5

# 配置
BV_ID="${1:-BV19T4y1J7v5}"
CLIP_ID="${2:-1}"
OUTPUT_DIR="/root/autodl-tmp/deep3d/data/test_set/speed_test_video/${CLIP_ID}/left"
COOKIE_FILE="/root/autodl-tmp/deep3d/data/test_set/speed_test_video/${CLIP_ID}/www.bilibili.com_cookies.txt"
VIDEO_URL="https://www.bilibili.com/video/${BV_ID}"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "Bilibili 1080p 纯视频下载"
echo "视频地址: $VIDEO_URL"
echo "输出目录: $OUTPUT_DIR"
echo "Cookie文件: $COOKIE_FILE"
echo "========================================"

# 使用 yt-dlp 下载 1080p 纯视频
# 参数说明:
# -f "bv*[height=1080][ext=mp4]" - 选择1080p视频流，mp4格式
# --no-audio - 不下载音频（纯视频）
# -o - 输出路径和文件名格式
# --cookies-from-browser chrome - 从浏览器获取cookies（用于1080p需要登录的情况）
# 或者使用 --cookies cookies.txt 指定cookies文件

echo "正在下载 1080p 纯视频..."

if [ -f "$COOKIE_FILE" ]; then
    yt-dlp \
        --cookies "$COOKIE_FILE" \
        -f "bv*[height<=1080][ext=mp4]/bv*[height<=1080]" \
        --no-audio \
        -o "${OUTPUT_DIR}/video_1080p.%(ext)s" \
        --merge-output-format mp4 \
        "$VIDEO_URL"
else
    yt-dlp \
        -f "bv*[height<=1080][ext=mp4]/bv*[height<=1080]" \
        --no-audio \
        -o "${OUTPUT_DIR}/video_1080p.%(ext)s" \
        --merge-output-format mp4 \
        "$VIDEO_URL"
fi

echo "========================================"
echo "下载完成!"
echo "文件保存在: $OUTPUT_DIR"
echo "========================================"

# 列出下载的文件
ls -lh "$OUTPUT_DIR"
