#!/usr/bin/env bash
# 把原始影片的音軌接回處理後的影片（OpenCV 輸出沒有聲音時的備用方案）。
#
#   scripts/mux_audio.sh out/stage_noaudio.mp4 samples/demo.mp4 out/stage.mp4
#
# 環境變數：
#   FFMPEG=/path/to/ffmpeg   指定 ffmpeg（預設依序找 PATH、imageio-ffmpeg）
#   COPY_VIDEO=1             影片已是 H.264 時直接複製視訊流，不重新編碼
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 <processed_video> <original_with_audio> <output>" >&2
    exit 2
fi
video="$1"; audio="$2"; out="$3"

ffmpeg="${FFMPEG:-}"
if [[ -z "$ffmpeg" ]]; then
    if command -v ffmpeg >/dev/null 2>&1; then
        ffmpeg=ffmpeg
    else
        ffmpeg="$(python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())' 2>/dev/null || true)"
    fi
fi
if [[ -z "$ffmpeg" ]]; then
    echo "ffmpeg not found: install ffmpeg or 'pip install imageio-ffmpeg'" >&2
    exit 1
fi

if [[ "${COPY_VIDEO:-0}" == "1" ]]; then
    vcodec=(-c:v copy)
else
    vcodec=(-c:v libx264 -crf 18 -preset medium -pix_fmt yuv420p)
fi

mkdir -p "$(dirname "$out")"
# -shortest 防止長度不一；1:a:0? 讓原檔沒有音軌時也不會失敗
"$ffmpeg" -hide_banner -loglevel error -y -i "$video" -i "$audio" \
    -map 0:v:0 -map '1:a:0?' "${vcodec[@]}" -c:a aac -b:a 192k -shortest \
    -movflags +faststart "$out"
echo "wrote $out"
