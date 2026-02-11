# -*- coding: utf-8 -*-
"""
管理者调研报告 — 专业商业仪表盘
前端 UI/UX 重构，后端逻辑不变（config / data_processor 保持不变）。
"""

import io
import math
import os
import tempfile
import urllib.request
from collections import Counter
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import jieba
from PIL import Image
from wordcloud import WordCloud

from config import (
    CATEGORY_ORDER,
    COLORS_BARS,
    COLOR_SCHEME,
    BASIC_INFO_COLS,
    BASIC_INFO_DISPLAY,
    OPEN_QUESTION_COLS,
)
from data_processor import (
    clean_and_score,
    compute_dimension_scores,
    get_behavior_avg_by_dimension,
    get_person_behavior_scores,
    get_all_behavior_avgs,
    get_person_total_and_dims,
)

# 中文停用词（词云过滤）
STOPWORDS_CN = {
    "的", "了", "是", "在", "我", "有", "和", "就", "不", "人", "都", "一", "一个",
    "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好",
    "自己", "这", "那", "等", "能", "与", "及", "或", "而", "把", "被", "让", "给",
    "无", "希望", "可以", "能够", "更多", "一些", "什么", "怎么", "如何", "为什么",
}

def _get_chinese_font_path():
    """返回系统可用的中文字体路径，用于词云（兼容 macOS / Windows / Linux 线上环境）"""
    candidates = [
        # macOS
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Supplemental/Songti.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
        # Windows
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        # Linux / Streamlit Cloud 常见路径
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
        "/usr/share/fonts/truetype/fonts-japanese-gothic/fonts-japanese-gothic.ttf",
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    # 通过 matplotlib 字体列表查找任意 CJK 字体（线上环境常带 Noto 等）
    try:
        import matplotlib.font_manager as fm
        for f in fm.fontManager.ttflist:
            path = getattr(f, "fname", None)
            if not path or not os.path.isfile(path):
                continue
            name = (f.name or "").lower()
            if "noto" in name or "cjk" in name or "sans" in name and ("sc" in name or "tc" in name or "jp" in name or "kr" in name):
                return path
    except Exception:
        pass
    # 线上无系统 CJK 字体时：下载 Noto Sans SC 并缓存，保证词云能显示中文
    return _download_chinese_font_cached()


def _download_chinese_font_cached():
    """无系统字体时下载并缓存中文字体，返回本地路径；失败返回 None。"""
    cache_dir = tempfile.gettempdir()
    cache_path = os.path.join(cache_dir, "NotoSansSC-Regular-wordcloud.otf")
    if os.path.isfile(cache_path):
        return cache_path
    url = "https://github.com/google/fonts/raw/main/ofl/notosanssc/NotoSansSC-Regular.otf"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Streamlit-App"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read()
        if len(data) > 1000:
            with open(cache_path, "wb") as f:
                f.write(data)
            return cache_path
    except Exception:
        pass
    return None

def _make_center_ellipse_mask(width: int, height: int, ratio=0.58):
    """生成中央椭圆镂空蒙版：椭圆内=0（不填字），椭圆外=255（填词云）。ratio 大则词云环更窄。"""
    canvas = np.full((height, width), 255, dtype=np.uint8)
    cx, cy = width // 2, height // 2
    rx = int(width * ratio * 0.6)
    ry = int(height * ratio * 0.85)
    y_grid, x_grid = np.ogrid[:height, :width]
    inside = ((x_grid - cx) ** 2 / (rx ** 2 + 1)) + ((y_grid - cy) ** 2 / (ry ** 2 + 1)) <= 1
    canvas[inside] = 0
    return canvas


def _load_wordcloud_mask_and_overlay(app_dir: str, width=900, height=380, character_ratio=0.58):
    """
    加载卡通 PNG 保持比例居中；中央椭圆镂空，词云只在环状区域。卡通用较大比例以更清晰。
    返回 (mask, overlay_img)。
    """
    mask_path = os.path.join(app_dir, "wordcloud_mask.png")
    if not os.path.isfile(mask_path):
        return None, None
    try:
        img = Image.open(mask_path)
        img = img.convert("RGBA")
        w0, h0 = img.size
        short = min(width, height)
        target_short = int(short * character_ratio)
        scale = min(target_short / w0, target_short / h0)
        nw, nh = int(round(w0 * scale)), int(round(h0 * scale))
        if nw < 1 or nh < 1:
            return None, None
        img_scaled = img.resize((nw, nh), Image.Resampling.LANCZOS)
        canvas_mask = _make_center_ellipse_mask(width, height, ratio=character_ratio)
        canvas_overlay = Image.new("RGBA", (width, height), (255, 255, 255, 0))
        x = (width - nw) // 2
        y = (height - nh) // 2
        canvas_overlay.paste(img_scaled, (x, y), img_scaled)
        return canvas_mask, canvas_overlay
    except Exception:
        return None, None


def build_wordcloud_image(text: str, width=900, height=380, mask_dir: str = None):
    """
    根据反馈文本生成词云图：红/橙配色，文字围绕卡通形象（保持比例）。
    返回 (PNG 字节流, 高频词列表)，失败时返回 (None, [])。
    """
    text = (text or "").strip()
    if not text:
        return None, []
    segs = jieba.lcut(text)
    words = [w for w in segs if len(w) >= 2 and w not in STOPWORDS_CN and w.strip()]
    if not words:
        return None, []
    freq = Counter(words)
    top_words = [w for w, _ in freq.most_common(20)]
    font_path = _get_chinese_font_path()
    mask, overlay_img = None, None
    if mask_dir:
        mask, overlay_img = _load_wordcloud_mask_and_overlay(
            mask_dir, width=width, height=height, character_ratio=0.58
        )
    kw = dict(
        width=width,
        height=height,
        background_color="#ffffff",
        max_words=80,
        relative_scaling=0.48,
        prefer_horizontal=0.6,
        max_font_size=72,
        min_font_size=12,
        colormap="Oranges",
        margin=3,
    )
    if font_path:
        kw["font_path"] = font_path
    if mask is not None:
        kw["mask"] = mask
        kw["contour_width"] = 0
        kw["contour_color"] = "white"
    try:
        wc = WordCloud(**kw)
        wc.generate_from_frequencies(freq)
        out = wc.to_image()
        if overlay_img is not None:
            out = out.convert("RGBA")
            out.paste(overlay_img, (0, 0), overlay_img)
        buf = io.BytesIO()
        out.save(buf, format="PNG")
        buf.seek(0)
        return buf, top_words
    except Exception:
        kw.pop("mask", None)
        kw.pop("contour_width", None)
        kw.pop("contour_color", None)
        try:
            wc = WordCloud(**kw)
            wc.generate_from_frequencies(freq)
            buf = io.BytesIO()
            wc.to_image().save(buf, format="PNG")
            buf.seek(0)
            return buf, top_words
        except Exception:
            return None, []

# ---------- 页面配置（必须最先） ----------
st.set_page_config(
    page_title="管理者调研报告",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================== 1. 全局样式美化 ====================
CUSTOM_CSS = """
<style>
  /* 强制浅色 + Ant Design 风：确定性、自然、高效 */
  [data-theme="dark"] .stApp,
  .stApp {
    color: rgba(0, 0, 0, 0.88) !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
    background: #f5f5f5 !important;
    min-height: 100vh;
  }
  [data-theme="dark"] section.main,
  section.main {
    background: #fafafa !important;
  }
  [data-theme="dark"] .main .block-container,
  [data-theme="dark"] .stMarkdown,
  [data-theme="dark"] p,
  [data-theme="dark"] span,
  [data-theme="dark"] label {
    color: #1e293b !important;
  }
  [data-theme="dark"] [data-testid="stSidebar"] {
    background: rgba(248, 250, 252, 0.95) !important;
    color: #1e293b !important;
  }
  [data-theme="dark"] [data-testid="stSidebar"] .stMarkdown,
  [data-theme="dark"] [data-testid="stSidebar"] p,
  [data-theme="dark"] [data-testid="stSidebar"] label { color: #334155 !important; }
  [data-theme="dark"] div[data-testid="stDataFrame"],
  [data-theme="dark"] .stDataFrame {
    background: #ffffff !important;
    color: #1e293b !important;
  }
  [data-theme="dark"] .stDataFrame th,
  [data-theme="dark"] .stDataFrame td,
  [data-theme="dark"] .stDataFrame tbody tr,
  [data-theme="dark"] .stDataFrame tbody tr:nth-child(even) { color: #1e293b !important; background: #ffffff !important; }
  [data-theme="dark"] .stDataFrame tbody tr:nth-child(even) { background: #f8fafc !important; }
  [data-theme="dark"] .stDataFrame thead tr th { background: #f1f5f9 !important; color: #334155 !important; }
  [data-theme="dark"] .kpi-card,
  [data-theme="dark"] .dim-block-card {
    background: rgba(255, 255, 255, 0.95) !important;
    color: #1e293b !important;
  }
  [data-theme="dark"] .stTabs [data-baseweb="tab-list"] { background: rgba(255, 255, 255, 0.9) !important; }
  [data-theme="dark"] .stSelectbox label,
  [data-theme="dark"] div[data-testid="stSelectbox"] { color: #1e293b !important; }

  /* 隐藏默认元素（保留 header 以显示侧边栏展开按钮，确保上传栏可打开） */
  #MainMenu { visibility: hidden; }
  footer { visibility: hidden; }

  /* Ant Design 风格：8px 栅格、中性色、确定性 */
  .stApp { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif; }
  .main .block-container {
    padding-top: 24px;
    padding-bottom: 32px;
    padding-left: 24px;
    padding-right: 24px;
    max-width: 1400px;
  }

  /* 主标题区：清晰层级 */
  .main-title-wrap {
    margin-bottom: 24px;
    padding-bottom: 16px;
    border-bottom: 1px solid #f0f0f0;
  }
  .main-title-wrap h1 {
    margin-bottom: 4px !important;
    font-size: 24px !important;
    font-weight: 600 !important;
    color: rgba(0, 0, 0, 0.88) !important;
  }
  .main-caption {
    font-size: 14px;
    color: rgba(0, 0, 0, 0.45);
  }

  /* 标题层级（Ant 规范） */
  h1, h2, h3, h4 { font-family: inherit; margin-bottom: 8px; }
  h1 { color: rgba(0,0,0,0.88); font-weight: 600; font-size: 24px; }
  h2 { color: rgba(0,0,0,0.88); font-weight: 600; font-size: 20px; margin-top: 24px; }
  h3, h4 { color: rgba(0,0,0,0.88); font-weight: 600; font-size: 16px; margin-top: 16px; }

  /* 标签页：Ant 线型 + 主色 */
  .stTabs [data-baseweb="tab-list"] {
    gap: 0;
    margin-bottom: 24px;
    padding: 0;
    background: transparent;
    border-bottom: 1px solid #f0f0f0;
  }
  .stTabs [data-baseweb="tab"] {
    padding: 12px 16px;
    border-radius: 0;
    font-weight: 500;
    font-size: 14px;
    background: transparent;
    color: rgba(0, 0, 0, 0.65);
    transition: color 0.2s;
    margin-bottom: -1px;
  }
  .stTabs [data-baseweb="tab"]:hover { color: #1677ff; }
  .stTabs [aria-selected="true"] {
    color: #1677ff !important;
    background: transparent !important;
    border-bottom: 2px solid #1677ff !important;
    box-shadow: none !important;
  }

  /* 表格：白底、细边框、Ant 表头 */
  div[data-testid="stDataFrame"] {
    border-radius: 6px;
    overflow: hidden;
    background: #fff;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    border: 1px solid #f0f0f0;
  }
  .stDataFrame thead tr th {
    background: #fafafa !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    color: rgba(0,0,0,0.88) !important;
    padding: 12px 16px !important;
    border-bottom: 1px solid #f0f0f0 !important;
  }
  .stDataFrame tbody tr:nth-child(even) { background-color: #fafafa !important; }
  .stDataFrame tbody tr:hover { background-color: #f5f5f5 !important; }
  .stDataFrame td, .stDataFrame th {
    padding: 12px 16px !important;
    white-space: normal !important;
    word-break: break-word !important;
    overflow-wrap: break-word !important;
    vertical-align: top !important;
    border-color: #f0f0f0 !important;
    font-size: 14px;
  }

  /* 各维度得分：标题条 + 表格块拼成同一框体；固定高度保证左右框体一致 */
  .dim-score-block-title {
    font-size: 16px;
    font-weight: 600;
    color: rgba(0,0,0,0.88);
    padding: 16px 20px;
    background: #fff;
    border: 1px solid #f0f0f0;
    border-bottom: none;
    border-radius: 6px 6px 0 0;
    margin-bottom: 0 !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    width: 100%;
    box-sizing: border-box;
    min-height: 52px;
    display: flex;
    align-items: center;
  }
  /* 表格块：紧跟标题的兄弟 + 含维度标题的列内偶数位块，固定高度+留白 */
  [data-testid="stVerticalBlock"]:has(.dim-score-block-title) + [data-testid="stVerticalBlock"],
  div:has(> .dim-score-block-title) + div,
  [data-testid="column"]:has(.dim-score-block-title) > div:nth-child(2n) {
    background: #fff !important;
    border: 1px solid #f0f0f0 !important;
    border-top: 1px solid #f0f0f0 !important;
    border-radius: 0 0 6px 6px !important;
    margin-top: 0 !important;
    margin-bottom: 16px !important;
    height: 380px !important;
    min-height: 380px !important;
    padding: 0 20px 16px 20px !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    overflow: hidden !important;
    display: flex !important;
    flex-direction: column !important;
  }
  [data-testid="stVerticalBlock"]:has(.dim-score-block-title) + [data-testid="stVerticalBlock"] div[data-testid="stDataFrame"],
  div:has(> .dim-score-block-title) + div div[data-testid="stDataFrame"],
  [data-testid="column"]:has(.dim-score-block-title) > div:nth-child(2n) div[data-testid="stDataFrame"] {
    margin: 0 !important;
    border: none !important;
    box-shadow: none !important;
    flex: 1 !important;
    min-height: 0 !important;
    overflow: auto !important;
  }
  [data-testid="stVerticalBlock"]:has(.dim-score-block-title) + [data-testid="stVerticalBlock"] .stDataFrame,
  div:has(> .dim-score-block-title) + div .stDataFrame,
  [data-testid="column"]:has(.dim-score-block-title) > div:nth-child(2n) .stDataFrame {
    border: none !important;
  }
  [data-testid="stVerticalBlock"]:has(.dim-score-block-title) + [data-testid="stVerticalBlock"] .stDataFrame th,
  [data-testid="stVerticalBlock"]:has(.dim-score-block-title) + [data-testid="stVerticalBlock"] .stDataFrame td,
  div:has(> .dim-score-block-title) + div .stDataFrame th,
  div:has(> .dim-score-block-title) + div .stDataFrame td,
  [data-testid="column"]:has(.dim-score-block-title) > div:nth-child(2n) .stDataFrame th,
  [data-testid="column"]:has(.dim-score-block-title) > div:nth-child(2n) .stDataFrame td {
    padding: 10px 8px !important;
  }
  /* 左右两列等宽、框体对齐；列内内容顶对齐，页面整洁 */
  .main [data-testid="column"] {
    min-width: 0;
    flex: 1 1 0;
    align-items: flex-start;
  }
  /* 个人报告：姓名/雷达图标题与内容顶对齐，无多余留白 */
  .main [data-testid="column"] .stMarkdown:first-child { margin-top: 0; }
  /* 个人报告三模块横排：等高、卡片样式（仅当一行恰有 3 列时） */
  .report-three-modules-marker { display: none; }
  .main [data-testid="stHorizontalBlock"]:has(> [data-testid="column"]:first-child:nth-last-child(3)),
  .main div:has(> [data-testid="column"]:first-child:nth-last-child(3)) {
    align-items: stretch !important;
  }
  .main [data-testid="stHorizontalBlock"]:has(> [data-testid="column"]:first-child:nth-last-child(3)) > [data-testid="column"],
  .main div:has(> [data-testid="column"]:first-child:nth-last-child(3)) > [data-testid="column"] {
    min-height: 380px !important;
    background: #fff !important;
    border: 1px solid #f0f0f0 !important;
    border-radius: 6px !important;
    padding: 16px !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03) !important;
  }

  /* 维度深度分析：模块名 + 下方图表同一框体（与个人报告维度块一致） */
  .dim-depth-block-title {
    font-size: 16px;
    font-weight: 600;
    color: rgba(0,0,0,0.88);
    padding: 16px 20px;
    background: #fff;
    border: 1px solid #f0f0f0;
    border-bottom: none;
    border-radius: 6px 6px 0 0;
    margin-bottom: 0 !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    width: 100%;
    box-sizing: border-box;
    min-height: 52px;
    display: flex;
    align-items: center;
  }
  [data-testid="stVerticalBlock"]:has(.dim-depth-block-title) + [data-testid="stVerticalBlock"],
  div:has(> .dim-depth-block-title) + div {
    background: #fff !important;
    border: 1px solid #f0f0f0 !important;
    border-top: 1px solid #f0f0f0 !important;
    border-radius: 0 0 6px 6px !important;
    margin-top: 0 !important;
    margin-bottom: 16px !important;
    padding: 12px 20px 16px 20px !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03);
  }
  [data-testid="stVerticalBlock"]:has(.dim-depth-block-title) + [data-testid="stVerticalBlock"] div[data-testid="stPlotlyChart"],
  div:has(> .dim-depth-block-title) + div div[data-testid="stPlotlyChart"] {
    margin: 0 !important;
  }

  /* KPI 卡片：Ant 卡片风格 */
  .kpi-card {
    background: #fff;
    border-radius: 6px;
    padding: 16px 20px;
    text-align: center;
    border: 1px solid #f0f0f0;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    transition: box-shadow 0.2s;
  }
  .kpi-card:hover { box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
  .kpi-card .kpi-value { font-size: 24px; font-weight: 600; color: rgba(0,0,0,0.88); }
  .kpi-label { font-size: 14px; font-weight: 500; margin-top: 8px; color: rgba(0,0,0,0.45); }

  /* 洞察区：Ant 风格左侧色条 + 浅色背景 */
  .insight-box {
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 14px;
    line-height: 1.6;
    border: none;
    border-radius: 6px;
  }
  .insight-box.high {
    border-left: 3px solid #52c41a;
    background: #f6ffed;
  }
  .insight-box.low {
    border-left: 3px solid #faad14;
    background: #fffbe6;
  }
  .insight-box.neutral {
    border-left: 3px solid #1677ff;
    background: #e6f4ff;
  }

  [data-testid="stMetricValue"] { font-size: 24px !important; font-weight: 600 !important; }
  .stCaptionContainer { font-size: 12px !important; color: rgba(0,0,0,0.45) !important; }

  /* 侧边栏：浅灰底、细边 */
  [data-testid="stSidebar"] {
    background: #fff !important;
    border-right: 1px solid #f0f0f0;
  }
  [data-testid="stSidebar"] .stMarkdown { font-weight: 500; }
  [data-testid="stSidebar"] h3 { font-size: 14px !important; color: rgba(0,0,0,0.88) !important; }

  .stSuccess, .stInfo { border-radius: 6px; border: 1px solid #f0f0f0; }

  /* 图表/图片容器：白卡 */
  div[data-testid="stPlotlyChart"],
  div[data-testid="stImage"] {
    border-radius: 6px;
    overflow: hidden;
    background: #fff;
    border: 1px solid #f0f0f0;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03);
  }

  .main .block-container h3, .main .block-container h4 { margin-bottom: 12px; }

  /* 数据可视化页：分区标题与卡片组（Ant 规范） */
  .viz-section { margin-top: 24px; }
  .viz-section:first-of-type { margin-top: 0; }
  .viz-section-title { font-size: 14px; color: rgba(0,0,0,0.45); margin-bottom: 8px; font-weight: 500; }
  .detail-card-wrap { background: #fff; border: 1px solid #f0f0f0; border-radius: 6px; padding: 16px; margin-bottom: 16px; box-shadow: 0 1px 2px rgba(0,0,0,0.03); }
  .disclaimer-highlight { color: #c5221f; font-weight: 600; background: #fce8e6; padding: 2px 6px; border-radius: 4px; }
  .disclaimer-box { background: #fff; border: 1px solid #f0f0f0; border-radius: 8px; padding: 24px 28px; margin: 24px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.06); line-height: 1.75; font-size: 15px; min-height: 380px; }
  .disclaimer-box h4 { margin-top: 0; margin-bottom: 16px; font-size: 18px; }
  .disclaimer-box ul { margin: 12px 0; padding-left: 22px; }
  .disclaimer-box .tip { margin-top: 20px; padding: 14px 16px; background: #fffbe6; border-left: 4px solid #faad14; border-radius: 4px; font-size: 14px; color: rgba(0,0,0,0.85); }
  .disclaimer-design-box { background: #fff; border: 1px solid #f0f0f0; border-radius: 8px; padding: 24px 28px; margin: 24px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.06); line-height: 1.75; font-size: 15px; min-height: 340px; }
  .disclaimer-questions-box.disclaimer-right-wrap { min-height: 720px; }
  .disclaimer-design-box h4 { margin-top: 0; margin-bottom: 12px; font-size: 16px; color: rgba(0,0,0,0.88); }
  .disclaimer-design-box .score-table { margin-top: 12px; border-collapse: collapse; width: 100%; max-width: 320px; font-size: 14px; }
  .disclaimer-design-box .score-table th, .disclaimer-design-box .score-table td { border: 1px solid #f0f0f0; padding: 10px 14px; text-align: left; }
  .disclaimer-design-box .score-table th { background: #fafafa; font-weight: 600; }
  .disclaimer-questions-box { background: #fff; border: 1px solid #f0f0f0; border-radius: 8px; padding: 20px 24px; margin: 0; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
  .disclaimer-questions-box h4 { margin: 0 0 16px 0; font-size: 16px; color: rgba(0,0,0,0.88); }
  .disclaimer-questions-box .questions-table { border-collapse: collapse; width: 100%; font-size: 13px; }
  .disclaimer-questions-box .questions-table th, .disclaimer-questions-box .questions-table td { border: 1px solid #f0f0f0; padding: 8px 10px; text-align: left; vertical-align: top; }
  .disclaimer-questions-box .questions-table th { background: #fafafa; font-weight: 600; }
  .disclaimer-questions-box .questions-table .col-module { width: 90px; white-space: nowrap; min-width: 90px; }
  .disclaimer-questions-box .questions-table .col-behavior { width: 90px; }
  .disclaimer-questions-box .questions-table .col-desc { font-size: 12px; line-height: 1.5; color: rgba(0,0,0,0.75); }
  /* 调研题目设置：模块列色块（与报告 COLOR_SCHEME 一致） */
  .disclaimer-questions-box .mod-role { background: rgba(230, 126, 34, 0.18); }
  .disclaimer-questions-box .mod-coach { background: rgba(243, 156, 18, 0.18); }
  .disclaimer-questions-box .mod-task { background: rgba(52, 152, 219, 0.18); }
  .disclaimer-questions-box .mod-motivate { background: rgba(41, 128, 185, 0.18); }
  .disclaimer-questions-box .mod-comm { background: rgba(26, 188, 156, 0.18); }

  /* 欢迎页左侧：第一块顶对齐、第二块与右侧表格容器下端对齐 */
  .disclaimer-left-wrap {
    min-height: 720px !important;
    flex: 1 1 auto !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: space-between !important;
    box-sizing: border-box !important;
  }
  .disclaimer-left-wrap .disclaimer-box { margin-top: 0 !important; margin-bottom: 0 !important; }
  .disclaimer-left-wrap .disclaimer-design-box { margin-top: auto !important; margin-bottom: 0 !important; }

  /* 欢迎页：左右列等高，左侧内容区撑满以便第二块底对齐 */
  .main [data-testid="stHorizontalBlock"]:has(.disclaimer-left-wrap):has(.disclaimer-questions-box) {
    align-items: stretch !important;
  }
  .main [data-testid="stHorizontalBlock"]:has(.disclaimer-left-wrap):has(.disclaimer-questions-box) > [data-testid="column"] {
    display: flex !important;
    flex-direction: column !important;
    min-height: 720px !important;
  }
  .main [data-testid="stHorizontalBlock"]:has(.disclaimer-left-wrap):has(.disclaimer-questions-box) > [data-testid="column"] > div {
    flex: 1 1 auto !important;
    min-height: 720px !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: flex-start !important;
  }
  /* 左侧列内包裹 disclaimer-left-wrap 的块撑满高度，使第二段能贴底 */
  .main [data-testid="stHorizontalBlock"]:has(.disclaimer-left-wrap) > [data-testid="column"]:first-child > div > div {
    flex: 1 1 auto !important;
    min-height: 0 !important;
    display: flex !important;
    flex-direction: column !important;
  }
  /* 任意直接包裹 .disclaimer-left-wrap 的父级都参与 flex，保证框体下端对齐 */
  .main [data-testid="stHorizontalBlock"]:has(.disclaimer-left-wrap) div:has(> .disclaimer-left-wrap) {
    flex: 1 1 auto !important;
    min-height: 0 !important;
    display: flex !important;
    flex-direction: column !important;
  }
  .disclaimer-questions-box.disclaimer-right-wrap {
    min-height: 720px !important;
    height: 100% !important;
    box-sizing: border-box !important;
  }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------- Plotly 配置 ----------
PLOTLY_CONFIG = {
    "displayModeBar": True,
    "modeBarButtonsToRemove": ["pan2d", "lasso2d", "select2d", "autoScale2d", "hoverClosestCartesian", "hoverCompareCartesian"],
    "displaylogo": False,
}

def apply_chart_style(fig, font_size=12):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=font_size),
        xaxis=dict(tickfont=dict(size=font_size - 1)),
        yaxis=dict(tickfont=dict(size=font_size - 1), showgrid=True, gridcolor="#F0F0F0"),
    )
    try:
        fig.update_yaxes(showgrid=True, gridcolor="#F0F0F0")
    except Exception:
        pass
    return fig

# ==================== 侧边栏 ====================
with st.sidebar:
    st.markdown("### 📁 数据上传")
    default_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "原始底表.xlsx")
    uploaded = st.file_uploader("上传 .xlsx 或 .csv", type=["xlsx", "csv"], key="uploader", label_visibility="collapsed")
    if not uploaded and os.path.isfile(default_path):
        with open(default_path, "rb") as f:
            uploaded = io.BytesIO(f.read())
        uploaded.name = "原始底表.xlsx"
    st.markdown("---")
    st.markdown("### ⚙️ 参数说明")
    st.caption("分值：总是=5，经常=4，有时=3，很少=2，从未=1")

if not uploaded:
    st.markdown("## 📊 管理者调研报告")
    st.markdown("请从 **左侧边栏** 上传问卷底表（.xlsx 或 .csv）后开始分析。")
    st.markdown("---")
    st.markdown("**文件要求**：表头含问卷题目、至少一列「填写人」或「姓名」，选项为五级量表。")
    st.stop()

# ---------- 数据加载（逻辑不变） ----------
@st.cache_data
def load_and_process(uploaded_file):
    try:
        name = getattr(uploaded_file, "name", "") or ""
        if name.endswith(".csv"):
            df = pd.read_csv(uploaded_file, encoding="utf-8")
        else:
            df = pd.read_excel(uploaded_file, sheet_name=0)
    except Exception as e:
        return None, str(e)
    df = pd.DataFrame(df)
    df_q, col_to_cat_be, _ = clean_and_score(df)
    if df_q is None or df_q.empty:
        return None, "未识别到问卷题目列，请检查表头。"
    df_dims = compute_dimension_scores(df_q, col_to_cat_be)
    total, _ = get_person_total_and_dims(df_q, df_dims)
    name_col = next((c for c in ["填写人", "姓名", "学员姓名"] if c in df.columns), None)
    names = df[name_col].astype(str).tolist() if name_col else [f"学员{i+1}" for i in range(len(df))]
    for col in OPEN_QUESTION_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("无").astype(str).replace("nan", "无")
    return {"df": df, "df_q": df_q, "df_dims": df_dims, "col_to_cat_be": col_to_cat_be, "names": names, "total": total}, None

data, err = load_and_process(uploaded)
if err:
    st.error("❌ " + err)
    st.stop()

df = data["df"]
df_q = data["df_q"]
df_dims = data["df_dims"]
col_to_cat_be = data["col_to_cat_be"]
names = data["names"]
total = data["total"]

# ---------- 说明中间页：需管理者确认后进入报告（左文右表） ----------
if st.session_state.get("disclaimer_confirmed", False) is not True:
    st.markdown("## 在您阅读报告之前，请您知悉")
    st.markdown("")
    col_left, col_right = st.columns([1.1, 1])
    with col_left:
        left_html = """
        <div class="disclaimer-left-wrap">
        <div class="disclaimer-box">
        <p><span class="disclaimer-highlight">这不是一份领导力评估报告。</span>本报告旨在呈现新灵秀课程的学员在不同管理动作上的自我评估结果，我们在设计本期课程的重点强调内容时将进行参考。把调研结果同步给您是希望：</p>
        <ul>
        <li><strong>为您提供一个视角</strong>，即：学员们眼中的自己在团队中是否充分展现了各方面管理动作，以便您在帮助学员校准自我认知时能有的放矢；</li>
        <li><strong>帮助学员打开乔哈里窗盲区</strong>，结合您对学员们的了解，帮助大家看见一些他们自己没有察觉的优劣势，未来期待着您的点拨和指导；</li>
        <li><strong>请您知悉</strong>这些优秀的伙伴们踏上了成长为更优秀管理者的旅途，一路上期待有您的关注和陪伴。</li>
        </ul>
        <div class="tip">
        <strong>【温馨提示】</strong>本报告结果是根据员工的自陈得出，请结合具体情况，根据员工日常表现以及360评价对各项数据进行理性的阐释，而不是单纯以分数论事，绝不能作为给员工贴标签的依据。<br><br>
        此报告涉及好未来集团保密信息。未经许可，任何人禁止不当使用（包含但不限于泄露、散发、复制）相关内容。
        </div>
        </div>
        <div class="disclaimer-design-box">
        <h4>调研题本设计说明</h4>
        <p>本次调研在凯洛格（KeyLogic Group）金牌培养项目《新经理成长地图》的设计逻辑之上，融合好未来的集团特色，分别从管理角色认知、辅导、任务分配、激励和沟通 5 个维度对新任管理者的管理动作呈现情况进行调研。</p>
        <h4>赋分标准</h4>
        <p>每个行为项的评分范围为 1～5 分，分数越高则表示参调者们出现该类行为的频率越高，报告中【均分】代表多位参调者自我描述的平均。自评分数换算逻辑：</p>
        <table class="score-table">
        <thead><tr><th>自评选项</th><th>赋分</th></tr></thead>
        <tbody>
        <tr><td>总是如此</td><td>5</td></tr>
        <tr><td>经常如此</td><td>4</td></tr>
        <tr><td>有时如此</td><td>3</td></tr>
        <tr><td>很少如此</td><td>2</td></tr>
        <tr><td>从未展现</td><td>1</td></tr>
        </tbody>
        </table>
        </div>
        </div>
        """
        st.markdown(left_html, unsafe_allow_html=True)
    with col_right:
        questions_html = """
        <div class="disclaimer-questions-box disclaimer-right-wrap">
        <h4>调研题目设置</h4>
        <table class="questions-table">
        <thead><tr><th class="col-module">模块</th><th class="col-behavior">行为项</th><th>具体行为描述</th></tr></thead>
        <tbody>
        <tr><td class="col-module mod-role">管理角色认知</td><td class="col-behavior">工作理念</td><td class="col-desc">比起亲力亲为，花了更多时间帮助下属推动工作，相信只有伙伴们完成任务自己才能取得成功。</td></tr>
        <tr><td class="col-module mod-role">管理角色认知</td><td class="col-behavior">时间管理</td><td class="col-desc">担任管理者后，将更多时间放在目标规划、任务分配、团队协作和教练辅导等相关的工作上。</td></tr>
        <tr><td class="col-module mod-role">管理角色认知</td><td class="col-behavior">言行合一</td><td class="col-desc">作为团队管理者，保证自己的所言即所行，从而促进团队伙伴间的互信。</td></tr>
        <tr><td class="col-module mod-role">管理角色认知</td><td class="col-behavior">接受反馈</td><td class="col-desc">作为团队管理者，能以谦虚的态度倾听下属反馈，并能以开放的心态接纳待改善的反馈。</td></tr>
        <tr><td class="col-module mod-coach">辅导</td><td class="col-behavior">主动辅导</td><td class="col-desc">当发现下属的产出成果低于预期或工作状态不佳时，会主动关心并予以辅导。</td></tr>
        <tr><td class="col-module mod-coach">辅导</td><td class="col-behavior">及时反馈</td><td class="col-desc">当观察到下属好或不好的表现时，都会进行及时的、充分的反馈，这也是我工作的一部分。</td></tr>
        <tr><td class="col-module mod-coach">辅导</td><td class="col-behavior">确定方向</td><td class="col-desc">辅导下属前，搜集多方信息并结合下属实际工作表现进行分析和推断，从而确定辅导方向。</td></tr>
        <tr><td class="col-module mod-coach">辅导</td><td class="col-behavior">预先思考</td><td class="col-desc">辅导下属前，事先思考在帮助下属解决问题的过程中所需要的方法与资源。</td></tr>
        <tr><td class="col-module mod-coach">辅导</td><td class="col-behavior">巧妙提问</td><td class="col-desc">在辅导下属时，通过提问引导下属进行思考，与下属共同讨论现状和解决方案。</td></tr>
        <tr><td class="col-module mod-coach">辅导</td><td class="col-behavior">跟踪结果</td><td class="col-desc">辅导下属后，定期考察下属的表现是否有变化，并根据数据去衡量结果。</td></tr>
        <tr><td class="col-module mod-task">任务分配</td><td class="col-behavior">综合评估</td><td class="col-desc">选择任务的分配对象时，综合评估任务难度和下属的能力、意愿和信心。</td></tr>
        <tr><td class="col-module mod-task">任务分配</td><td class="col-behavior">授权下属</td><td class="col-desc">相信下属有完成任务的能力，授权下属让他们自己做决策，在必要时提供适当帮助。</td></tr>
        <tr><td class="col-module mod-task">任务分配</td><td class="col-behavior">清楚委任</td><td class="col-desc">分配任务时，清晰说明为什么要做这个任务和期望的成果等，并提供必要的支持。</td></tr>
        <tr><td class="col-module mod-task">任务分配</td><td class="col-behavior">跟踪进度</td><td class="col-desc">分配任务时，与下属确认后续的追踪方式以及衡量标准，定期跟踪计划进度。</td></tr>
        <tr><td class="col-module mod-motivate">激励</td><td class="col-behavior">激发热情</td><td class="col-desc">主动了解下属的兴趣和能力，安排工作时考虑下属的兴趣以及个人发展诉求。</td></tr>
        <tr><td class="col-module mod-motivate">激励</td><td class="col-behavior">认可价值</td><td class="col-desc">通过沟通帮助下属了解其工作对团队目标的贡献，理解其工作的价值和重要性，并在日常的工作中给予认可。</td></tr>
        <tr><td class="col-module mod-motivate">激励</td><td class="col-behavior">营造氛围</td><td class="col-desc">营造开放的、安全的、彼此依靠的团队氛围，鼓励下属进一步学习和展现新的技能。</td></tr>
        <tr><td class="col-module mod-motivate">激励</td><td class="col-behavior">规划发展</td><td class="col-desc">定期与下属就优势和待发展项进行开放的讨论，提供建设性的反馈并形成后续的发展计划。</td></tr>
        <tr><td class="col-module mod-comm">沟通</td><td class="col-behavior">认真倾听</td><td class="col-desc">在工作中，让伙伴们多表达，耐心的让对方充分表达观点，理解对方的动机和顾虑。</td></tr>
        <tr><td class="col-module mod-comm">沟通</td><td class="col-behavior">积极回应</td><td class="col-desc">与伙伴沟通时，通过眼神交流、点头或不断提出有启发性的问题等方式，表现出对话题的兴趣。</td></tr>
        <tr><td class="col-module mod-comm">沟通</td><td class="col-behavior">坦诚表达</td><td class="col-desc">开放地跟伙伴们分享自己的想法、理由和感受。</td></tr>
        <tr><td class="col-module mod-comm">沟通</td><td class="col-behavior">提问澄清</td><td class="col-desc">在沟通中遇到不确定的信息，会通过耐心提问来确认自己对其他伙伴观点的理解是否准确。</td></tr>
        </tbody>
        </table>
        </div>
        """
        st.markdown(questions_html, unsafe_allow_html=True)
    st.markdown("")
    if st.button("确认已阅读，进入报告", type="primary", use_container_width=False):
        st.session_state["disclaimer_confirmed"] = True
        st.rerun()
    st.stop()

with st.sidebar:
    st.markdown("---")
    st.markdown("### 👤 学员筛选")
    selected_name = st.selectbox("选择学员", names, key="sel_name", label_visibility="collapsed")

# ==================== 主布局 ====================
st.markdown(
    f'<div class="main-title-wrap">'
    f'<h1>管理者调研报告</h1>'
    f'<p class="main-caption">已加载 {len(df)} 条记录 · {len(col_to_cat_be)} 道题 · {len(names)} 位学员</p>'
    f'</div>',
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 全局概览",
    "🧩 维度深度分析",
    "👤 个人详细报告",
    "📝 开放反馈",
    "⚠️ 异常提醒",
])

# ---------- Tab 1: 全局概览（数据可视化 · 概览第一） ----------
with tab1:
    dim_means = df_dims[CATEGORY_ORDER].mean() if all(c in df_dims.columns for c in CATEGORY_ORDER) else df_dims.mean()
    summary = pd.DataFrame({"维度": dim_means.index.tolist(), "全员平均分": dim_means.values.round(2).tolist()})
    scores = summary["全员平均分"].values
    max_s, min_s = float(scores.max()), float(scores.min())

    # 核心数据（Ant：将最关键指标置于顶部）
    st.markdown(
        f'<p class="viz-section-title">核心数据</p>'
        f'<p style="font-size:14px; color:rgba(0,0,0,0.88); margin-bottom:0;">'
        f'共 <strong>{len(names)}</strong> 位学员 · <strong>{len(df)}</strong> 条有效记录 · <strong>{len(CATEGORY_ORDER)}</strong> 个维度</p>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # 指标卡模块（Ant：对数据高度概括时，指标卡+数值比图表更直接）
    st.markdown("#### 五维度全员平均分")
    st.caption("核心指标卡，可结合下方图表与「维度深度分析」查看细节。")
    cols = st.columns(5)
    for i, (dim, sc) in enumerate(zip(summary["维度"], summary["全员平均分"])):
        with cols[i]:
            color = COLOR_SCHEME.get(dim, "#64748b")
            badge = ""
            if sc == max_s:
                badge = '<span style="font-size:0.7rem; font-weight:600; color:#059669; margin-left:0.25rem;">最高</span>'
            elif sc == min_s:
                badge = '<span style="font-size:0.7rem; font-weight:600; color:#ea580c; margin-left:0.25rem;">最低</span>'
            st.markdown(
                f'<div class="kpi-card">'
                f'<div class="kpi-value">{sc:.2f}{badge}</div>'
                f'<p class="kpi-label" style="color:{color}">{dim}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown("#### 维度对比")
    st.caption("可切换「维度深度分析」页签查看各维度下的行为项得分。")
    bar_colors = ["#10b981" if s == max_s else "#f59e0b" if s == min_s else "#3498db" for s in scores]
    fig1 = go.Figure(data=[go.Bar(
        x=summary["全员平均分"], y=summary["维度"], orientation="h",
        marker_color=bar_colors, text=summary["全员平均分"], texttemplate="%{text:.2f}", textposition="outside",
    )])
    fig1.update_layout(xaxis_title="平均分", xaxis=dict(range=[0, 5.8]), height=320, margin=dict(l=120), showlegend=False)
    fig1.update_yaxes(showgrid=True, gridcolor="#F0F0F0")
    fig1 = apply_chart_style(fig1)
    st.plotly_chart(fig1, use_container_width=True, config=PLOTLY_CONFIG)
    st.caption("🟢 最高分维度　　🟠 最低分维度")

    st.markdown("---")
    dim_max = summary.loc[summary["全员平均分"].idxmax(), "维度"]
    dim_min = summary.loc[summary["全员平均分"].idxmin(), "维度"]
    overall_avg = float(scores.mean())
    insight_text = (
        f"<strong>表现最佳</strong>：{dim_max}（{max_s:.2f} 分）— 可总结经验、固化做法。<br>"
        f"<strong>最需关注</strong>：{dim_min}（{min_s:.2f} 分）— 建议在培训中优先加强。<br>"
        f"<strong>整体</strong>：五维度全员平均 <strong>{overall_avg:.2f}</strong> 分。"
        + (" 各维度相对均衡。" if max_s - min_s < 0.5 else f" 最高与最低相差 {max_s - min_s:.2f} 分，可重点补足短板。")
    )
    st.markdown("#### 简要洞察")
    st.markdown(
        f'<div class="insight-box neutral" style="margin-top:0;">{insight_text}</div>',
        unsafe_allow_html=True,
    )

# ---------- Tab 2: 维度深度分析（数据可视化 · 多维分析） ----------
with tab2:
    st.markdown("#### 各维度行为项得分（全员平均）")
    st.caption("针对同一主题的多个维度分析，便于发现各维度下的强弱行为项。左侧筛选器可切换学员，个人得分见「个人详细报告」。")
    behavior_avgs = get_behavior_avg_by_dimension(df_q, col_to_cat_be)
    dim_items = []
    for i, cat in enumerate(CATEGORY_ORDER):
        if cat not in behavior_avgs:
            continue
        be_dict = behavior_avgs[cat]
        be_names = list(be_dict.keys())
        be_scores = [round(be_dict[b], 2) for b in be_names]
        max_be, min_be = max(be_scores), min(be_scores)
        bar_colors = ["#10b981" if s == max_be else "#dc2626" if s == min_be else "#94a3b8" for s in be_scores]
        strong_be = be_names[be_scores.index(max_be)]
        weak_be = be_names[be_scores.index(min_be)]
        dim_items.append((cat, be_names, be_scores, bar_colors, strong_be, max_be, weak_be, min_be, i))

    # 分模块并列：两列排布，模块名+图表同一框体（与个人报告维度块一致）
    for k in range(0, len(dim_items), 2):
        col_a, col_b = st.columns(2)
        for j, col in enumerate([col_a, col_b]):
            idx = k + j
            if idx >= len(dim_items):
                continue
            cat, be_names, be_scores, bar_colors, strong_be, max_be, weak_be, min_be, i = dim_items[idx]
            color = COLOR_SCHEME.get(cat, "#333333")
            with col:
                st.markdown(
                    f'<div class="dim-depth-block-title">'
                    f'<span style="color: {color};">{cat}</span>'
                    f'<span style="font-size: 12px; color: rgba(0,0,0,0.45); margin-left: 8px; font-weight: 400;">'
                    f'🟢 最高 {strong_be} {max_be:.2f}　🔴 最低 {weak_be} {min_be:.2f}'
                    f'</span></div>',
                    unsafe_allow_html=True,
                )
                # Y 轴聚焦数据范围，增强子维度区分度
                y_min_data = min(be_scores)
                y_max_data = max(be_scores)
                span = max(1.0, (y_max_data - y_min_data) + 0.5)
                center = (y_min_data + y_max_data) / 2
                y_low = max(0, center - span / 2)
                y_high = min(5.5, center + span / 2)
                if y_high - y_low < 0.5:
                    y_low = max(0, y_min_data - 0.25)
                    y_high = min(5.5, y_max_data + 0.25)
                fig_dim = go.Figure(data=[go.Bar(
                    x=be_names, y=be_scores, marker_color=bar_colors,
                    text=be_scores, texttemplate="%{text:.2f}", textposition="outside",
                )])
                fig_dim.update_layout(
                    xaxis_title="", yaxis_title="平均分",
                    yaxis=dict(range=[y_low, y_high], showgrid=True, gridcolor="#F0F0F0", dtick=0.2),
                    xaxis=dict(tickangle=0),
                    height=max(220, len(be_names) * 36),
                    margin=dict(t=20, b=50, l=40, r=20),
                    showlegend=False,
                )
                fig_dim = apply_chart_style(fig_dim)
                st.plotly_chart(fig_dim, use_container_width=True, config=PLOTLY_CONFIG)

# ---------- Tab 3: 个人详细报告（详情页 · 层次分明、直截了当） ----------
with tab3:
    # 1. 人员筛选区域（保留）
    st.markdown("#### 选择员工")
    selected_for_tab3 = st.selectbox("学员", names, index=names.index(selected_name), key="sel_tab3", label_visibility="collapsed")

    idx = names.index(selected_for_tab3)
    row_index = df_q.index[idx]
    profile_row = df.iloc[idx]
    dim_cols = [c for c in CATEGORY_ORDER if c in df_dims.columns]
    row_dims = df_dims.loc[row_index, dim_cols] if dim_cols else pd.Series(dtype=float)
    total_person = float(total.loc[row_index])
    dim_means_all = df_dims[dim_cols].mean() if dim_cols else pd.Series(dtype=float)
    above = [c for c in dim_cols if row_dims[c] >= dim_means_all[c]] if dim_cols else []
    below = [c for c in dim_cols if row_dims[c] < dim_means_all[c]] if dim_cols else []

    # 2. 员工筛选下：三模块横向排布（员工信息与得分 | 总分 | 五维度得分 vs 全员均分），等高对齐
    st.markdown('<div class="report-three-modules-marker"></div>', unsafe_allow_html=True)
    col_info, col_score, col_radar = st.columns(3)
    with col_info:
        st.markdown("**员工信息与得分**")
        display_map = [
            ("部门", "部门"),
            ("工号", "工号"),
            ("管理年限", "您开始带团队有多久啦？"),
            ("团队规模", "向您直接汇报的伙伴有多少？"),
        ]
        for label, col_key in display_map:
            val = profile_row.get(col_key, "") if col_key in df.columns else ""
            if pd.isna(val) or val == "" or (isinstance(val, float) and math.isnan(val)):
                val = "-"
            else:
                val = str(val).strip()
            st.markdown(f"**{label}**：{val}")

    with col_score:
        st.markdown("**总分（题目平均）**")
        above_text = ""
        if above:
            dims_joined = "」「".join(above)
            above_text = f'<p style="margin:4px 0 0 0; font-size:14px; line-height:1.5;"><strong>💪 高于全员</strong>：{selected_for_tab3} 在「{dims_joined}」上达到或超过全员平均。</p>'
        st.markdown(
            f'<div style="margin:0;">'
            f'<p style="font-size:24px; font-weight:600; color:rgba(0,0,0,0.88); margin:4px 0 0 0;">{total_person:.2f}</p>'
            f'{above_text}'
            f'<hr style="margin:10px 0 10px 0; border:none; border-top:1px solid #f0f0f0;">'
            f'</div>',
            unsafe_allow_html=True,
        )
        if dim_cols:
            for c in dim_cols:
                st.write(f"**{c}**：{row_dims[c]:.2f}（全员均分 {dim_means_all[c]:.2f}）")
        if below:
            st.markdown('<div class="insight-box low">', unsafe_allow_html=True)
            st.markdown(f"**📈 建议关注**：在「{'」「'.join(below)}」上低于全员平均，建议重点提升。")
            st.markdown("</div>", unsafe_allow_html=True)

    with col_radar:
        st.markdown(f"**{selected_for_tab3}** · 五维度得分 vs 全员均分")
        theta_radar = dim_cols if dim_cols else []
        r_person = [float(row_dims[c]) for c in theta_radar]
        r_avg = [float(dim_means_all[c]) for c in theta_radar]
        if theta_radar and len(r_person) == 5:
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=r_person + [r_person[0]],
                theta=theta_radar + [theta_radar[0]],
                fill="toself",
                fillcolor="rgba(52, 152, 219, 0.35)",
                line=dict(color="#3498DB", width=2),
                name=selected_for_tab3,
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=r_avg + [r_avg[0]],
                theta=theta_radar + [theta_radar[0]],
                fill="toself",
                fillcolor="rgba(148, 163, 184, 0.2)",
                line=dict(color="#94a3b8", width=1.5, dash="dash"),
                name="全员均分",
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 5.5], tickfont=dict(size=11), gridcolor="#F0F0F0"), bgcolor="rgba(0,0,0,0)"),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                height=320,
                margin=dict(t=40, b=40),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_radar, use_container_width=True, config=PLOTLY_CONFIG)
        else:
            st.info("暂无足够维度数据，无法绘制雷达图。")

    # 4. 各维度详细得分（详情页：按相关性分组，卡片区隔）
    st.markdown("#### 各维度详细得分")
    st.caption("按维度分组展示行为项、描述与得分。🟢 ≥4.5 浅绿　🔴 ≤3.0 浅红　⚪ 其他白底")

    def highlight_score_cell(v):
        try:
            x = float(v)
        except (TypeError, ValueError):
            x = 0
        if x >= 4.5:
            return "background-color: #e6f4ea; color: #0d652d"
        if x <= 3.0:
            return "background-color: #fce8e6; color: #c5221f"
        return "background-color: #ffffff; color: #333333"

    def style_dim_table(df):
        def _apply(series):
            if series.name != "得分":
                return [""] * len(series)
            return [highlight_score_cell(v) for v in series]
        return df.style.apply(_apply, axis=0)

    dim_tables = []
    for dim in CATEGORY_ORDER:
        rows = []
        for col, (cat, be) in col_to_cat_be.items():
            if cat != dim:
                continue
            val = df_q.loc[row_index, col]
            val_f = float(val) if not math.isnan(val) else 0
            avg_f = float(df_q[col].mean())
            rows.append({
                "行为项": be,
                "行为描述": str(col).strip(),
                "得分": round(val_f, 2),
                "均分": round(avg_f, 2),
            })
        if rows:
            dim_tables.append((dim, pd.DataFrame(rows)))

    # 管理角色认知按 6 行展示，与右侧辅导等维度高度一致（不足补空行，超过截断）
    TARGET_ROWS = 6
    DIM_FIX_ROWS = "管理角色认知"
    fixed = []
    for dim, dim_df in dim_tables:
        if dim == DIM_FIX_ROWS:
            n = len(dim_df)
            if n >= TARGET_ROWS:
                dim_df = dim_df.head(TARGET_ROWS).copy()
            else:
                empty = pd.DataFrame([
                    {"行为项": "", "行为描述": "", "得分": "", "均分": ""}
                    for _ in range(TARGET_ROWS - n)
                ])
                dim_df = pd.concat([dim_df, empty], ignore_index=True)
        fixed.append((dim, dim_df))
    dim_tables = fixed

    col1, col2 = st.columns(2)
    for i, (dim, dim_df) in enumerate(dim_tables):
        target = col1 if i % 2 == 0 else col2
        with target:
            st.markdown(
                f'<div class="dim-score-block-title">{dim}</div>',
                unsafe_allow_html=True,
            )
            styled = style_dim_table(dim_df)
            st.dataframe(
                styled,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "行为项": st.column_config.TextColumn("行为项", width=100),
                    "行为描述": st.column_config.TextColumn("行为描述", width=420),
                    "得分": st.column_config.NumberColumn("得分", format="%.2f", width=65),
                    "均分": st.column_config.NumberColumn("均分", format="%.2f", width=65),
                },
            )

# ---------- Tab 4: 开放反馈 ----------
with tab4:
    st.markdown("#### 开放反馈汇总")
    name_col_df = next((c for c in ["填写人", "姓名", "学员姓名"] if c in df.columns), None)
    dept_col = "部门" if "部门" in df.columns else None
    open_cols = [c for c in OPEN_QUESTION_COLS if c in df.columns]
    if not name_col_df or not open_cols:
        st.info("当前数据中未找到「填写人」或开放性问题列（如「您对这次培训还有哪些期待？」），无法展示。")
    else:
        if dept_col:
            dept_options = ["全部"] + sorted(df[dept_col].dropna().astype(str).unique().tolist())
            selected_depts = st.selectbox("按部门筛选", dept_options, key="open_dept")
            if selected_depts == "全部":
                open_df = df[[name_col_df, dept_col] + open_cols].copy()
            else:
                open_df = df[df[dept_col].astype(str) == selected_depts][[name_col_df, dept_col] + open_cols].copy()
        else:
            open_df = df[[name_col_df] + open_cols].copy()
        open_df = open_df.fillna("无")
        for c in open_cols:
            open_df[c] = open_df[c].astype(str).replace("nan", "无")
        def has_content(val):
            s = str(val).strip()
            return s and s not in ("无", "-", "—")
        mask = open_df[open_cols].apply(lambda row: any(has_content(row[c]) for c in open_cols), axis=1)
        open_df = open_df[mask].reset_index(drop=True)
        if open_df.empty:
            st.caption("暂无有效开放反馈（已填写「无」或为空的记录不展示）")
        else:
            all_text_parts = []
            for _, row in open_df.iterrows():
                for c in open_cols:
                    val = str(row[c]).strip()
                    if val and val not in ("无", "-", "—"):
                        all_text_parts.append(val)
            combined_text = " ".join(all_text_parts)
            _app_dir = os.path.dirname(os.path.abspath(__file__))

            # 左右布局：左侧词云（略小），右侧填写明细，不再上下分屏
            col_wc, col_detail = st.columns([1, 2])
            with col_wc:
                st.markdown("##### 伙伴反馈词云")
                st.caption("根据开放反馈内容生成")
                wc_buf, top_keywords = build_wordcloud_image(
                    combined_text, width=420, height=320, mask_dir=_app_dir
                )
                if wc_buf:
                    st.image(wc_buf, use_container_width=True)
                    if top_keywords:
                        st.markdown(
                            '<p style="font-size:13px; color:rgba(0,0,0,0.45); margin-top:8px;">'
                            '<strong>高频词</strong>：' + "　".join(f'<span style="color:#1677ff;">{w}</span>' for w in top_keywords[:12]) +
                            '</p>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.caption("反馈内容过少，暂无法生成词云。")

            with col_detail:
                st.markdown("##### 填写明细")
                col_config = {
                    name_col_df: st.column_config.TextColumn(name_col_df, width=85),
                    **({dept_col: st.column_config.TextColumn(dept_col, width=85)} if dept_col else {}),
                    **{c: st.column_config.TextColumn(c, width="large") for c in open_cols},
                }
                st.dataframe(open_df, use_container_width=True, hide_index=True, column_config=col_config)

# ---------- Tab 5: 异常提醒 ----------
with tab5:
    st.markdown("#### 异常名单 · 建议管理者关注")
    st.caption("单选题（量表题）若全部为同一分值，则视为异常，可能存在应付填答，建议关注。")
    score_cols = list(col_to_cat_be.keys())
    anomaly_rows = []
    for idx in df_q.index:
        row = df_q.loc[idx, score_cols]
        valid = row.dropna()
        if len(valid) >= 1 and valid.nunique() == 1:
            uniform_score = float(valid.iloc[0])
            anomaly_rows.append((idx, uniform_score))
    name_col_anom = next((c for c in ["填写人", "姓名", "学员姓名"] if c in df.columns), None)
    dept_col_anom = "部门" if "部门" in df.columns else None
    if not anomaly_rows:
        st.success("✅ 当前无异常：未发现「全部题目同一分值」的填答。")
    else:
        rows_out = []
        for idx, uniform_score in anomaly_rows:
            r = {"填写人": df.loc[idx, name_col_anom] if name_col_anom else f"学员{idx+1}"}
            if dept_col_anom:
                r["部门"] = df.loc[idx, dept_col_anom]
            r["统一分值"] = round(uniform_score, 2)
            r["提醒说明"] = f"该伙伴所有题目均为 {uniform_score:.1f} 分，建议管理者关注。"
            rows_out.append(r)
        anomaly_df = pd.DataFrame(rows_out)
        col_config = {"填写人": st.column_config.TextColumn("填写人", width=120)}
        if dept_col_anom:
            col_config["部门"] = st.column_config.TextColumn("部门", width=100)
        col_config["统一分值"] = st.column_config.NumberColumn("统一分值", format="%.2f", width=90)
        col_config["提醒说明"] = st.column_config.TextColumn("提醒说明", width="large")
        st.dataframe(anomaly_df, use_container_width=True, hide_index=True, column_config=col_config)
