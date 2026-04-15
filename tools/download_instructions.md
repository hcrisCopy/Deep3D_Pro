# Bilibili 1080p 纯视频下载操作指南

## 视频信息
- **示例视频BV号**: `BV19T4y1J7v5`
- **示例视频链接**: `https://www.bilibili.com/video/BV19T4y1J7v5`
- **目标路径模板**: `/root/autodl-tmp/deep3d/data/test_set/speed_test_video/<编号>/left`
- **Cookie 路径模板**: `/root/autodl-tmp/deep3d/data/test_set/speed_test_video/<编号>/www.bilibili.com_cookies.txt`

## 前置条件

### 1. 安装 yt-dlp
```bash
# 方法1: 使用 pip
pip install yt-dlp

# 方法2: 直接下载二进制文件
curl -L https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp -o /usr/local/bin/yt-dlp
chmod +x /usr/local/bin/yt-dlp

# 验证安装
yt-dlp --version
```

### 2. 安装依赖（可选但推荐）
```bash
# 用于合并视频和音频（如果需要的话）
apt-get update && apt-get install -y ffmpeg
```

### 3. 登录凭证（获取1080p必需）
Bilibili的1080p视频需要登录后才能访问，有两种方式提供凭证：

**方式A: 从浏览器自动获取 cookies**
```bash
# 在服务器上安装浏览器或使用本地导出
yt-dlp --cookies-from-browser chrome "视频链接"
```

**方式B: 手动导出 cookies.txt（推荐用于服务器）**
1. 在本地浏览器安装 "Get cookies.txt LOCALLY" 扩展
2. 访问 bilibili.com 并登录
3. 点击扩展导出 cookies.txt
4. 将 cookies.txt 上传到服务器的 `/root/` 目录

## 操作步骤

### 步骤1: 准备环境
```bash
# 以编号 1 为例创建目标目录
mkdir -p /root/autodl-tmp/deep3d/data/test_set/speed_test_video/1/left

# 上传 cookies.txt 到对应编号目录
scp cookies.txt root@你的服务器IP:/root/autodl-tmp/deep3d/data/test_set/speed_test_video/1/www.bilibili.com_cookies.txt
```

### 步骤2: 运行下载脚本
```bash
# 给脚本执行权限
chmod +x download_bilibili_1080p.sh

# 运行脚本：参数1是 BV 号，参数2是编号目录
./download_bilibili_1080p.sh BV19T4y1J7v5 1
```

### 步骤3: 直接使用命令下载（如果不使用脚本）

#### 基础命令（720p，无需登录）
```bash
yt-dlp \
    -f "bv*[height<=1080]+ba/b[height<=1080]" \
    --no-audio \
    -o "/root/autodl-tmp/deep3d/data/test_set/speed_test_video/1/left/video_1080p.%(ext)s" \
    "https://www.bilibili.com/video/BV19T4y1J7v5"
```

#### 1080p 需要 cookies
```bash
yt-dlp \
    --cookies /root/autodl-tmp/deep3d/data/test_set/speed_test_video/1/www.bilibili.com_cookies.txt \
    -f "bv*[height<=1080][ext=mp4]" \
    --no-audio \
    -o "/root/autodl-tmp/deep3d/data/test_set/speed_test_video/1/left/video_1080p.%(ext)s" \
    "https://www.bilibili.com/video/BV19T4y1J7v5"
```

## 常用 yt-dlp 参数说明

| 参数 | 说明 |
|------|------|
| `-f "bv*[height=1080]"` | 选择1080p视频流 |
| `--no-audio` | 只下载视频，不下载音频 |
| `-o "路径"` | 指定输出路径和文件名格式 |
| `--cookies cookies.txt` | 使用cookie文件登录 |
| `--list-formats` | 列出所有可用格式 |
| `-F` | 简写的 --list-formats |

## 查看可用格式
```bash
yt-dlp -F "https://www.bilibili.com/video/BV19T4y1J7v5"
```

## 常见问题

### Q1: 提示 "This video requires login"
**解决方案**: 使用 `--cookies` 参数提供登录凭证

### Q2: 下载速度慢
**解决方案**: 添加 `-N 4` 参数启用多线程下载
```bash
yt-dlp -N 4 --cookies cookies.txt "视频链接"
```

### Q3: 需要特定编码格式
**解决方案**: 使用 `--merge-output-format` 指定输出格式
```bash
yt-dlp --merge-output-format mp4 --cookies cookies.txt "视频链接"
```

## 替代工具（可选）

### 使用 you-get
```bash
pip install you-get
you-get --cookies /root/autodl-tmp/deep3d/data/test_set/speed_test_video/1/www.bilibili.com_cookies.txt -o /root/autodl-tmp/deep3d/data/test_set/speed_test_video/1/left "https://www.bilibili.com/video/BV19T4y1J7v5"
```

### 使用 BBDown（专门用于B站）
```bash
# 下载 BBDown
wget https://github.com/nilaoda/BBDown/releases/latest/download/BBDown_linux-x64
chmod +x BBDown_linux-x64

# 使用BBDown下载
./BBDown_linux-x64 --cookie "你的cookie字符串" -hevc -F "1080P 高清" "BV19T4y1J7v5"
```

## 文件说明
- `download_bilibili_1080p.sh` - 自动化下载脚本
- `download_instructions.md` - 本说明文档

## 当前项目中的实际目录约定

测速视频现在统一放在：

```text
/root/autodl-tmp/deep3d/data/test_set/speed_test_video/
├── 1/
│   ├── left/
│   └── www.bilibili.com_cookies.txt
├── 2/
│   ├── left/
│   └── www.bilibili.com_cookies.txt
└── 3/
    ├── left/
    └── www.bilibili.com_cookies.txt
```

如果要下载到 `2/left` 或 `3/left`，只需要把命令中的编号从 `1` 改成对应目录编号即可。
