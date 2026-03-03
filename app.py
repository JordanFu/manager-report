# -*- coding: utf-8 -*-
"""
管理者调研报告 — 专业商业仪表盘
前端 UI/UX 重构，后端逻辑不变（config / data_processor 保持不变）。
"""

import io
import math
import os
import string
import subprocess
import sys
import tempfile
import urllib.request
from collections import Counter, defaultdict
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import jieba
from PIL import Image
from wordcloud import WordCloud

# 用于 PDF 图表导出：无界面后端，避免 kaleido 不可用时无图
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from app_config import (
    CATEGORY_ORDER,
    COLORS_BARS,
    COLOR_SCHEME,
    BASIC_INFO_COLS,
    BASIC_INFO_DISPLAY,
    OPEN_QUESTION_COLS,
    LEARNING_MODULE_COL,
    TENURE_COL,
    TEAM_SIZE_COL,
    EXCLUDE_PDF_ROLE_LABEL,
    SURVEY_QUESTIONS,
)
from data_processor import (
    clean_and_score,
    compute_dimension_scores,
    get_behavior_avg_by_dimension,
    get_person_behavior_scores,
    get_all_behavior_avgs,
    get_person_total_and_dims,
)
from pdf_generator import PDFReport, REPORTLAB_AVAILABLE, REPORTLAB_IMPORT_ERROR

# 中文停用词（词云过滤）
STOPWORDS_CN = {
    "的", "了", "是", "在", "我", "有", "和", "就", "不", "人", "都", "一", "一个",
    "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好",
    "自己", "这", "那", "等", "能", "与", "及", "或", "而", "把", "被", "让", "给",
    "无", "可以", "能够", "一些", "什么", "怎么", "如何", "为什么",
}

# 词云统计排除：标点符号（中英文及常见符号），不参与分词统计
PUNCT_FOR_WORDCLOUD = set(string.punctuation) | set("，。！？、；：""''（）【】《》…—·～ \t\n\r")

def _is_punctuation_only(s: str):
    """判断是否为纯标点/空白，此类不纳入词云统计。"""
    if not s or not s.strip():
        return True
    return all(c in PUNCT_FOR_WORDCLOUD for c in s)

def _strip_punctuation_for_word(s: str):
    """去掉首尾标点，便于「问题，」统计为「问题」。"""
    s = s.strip()
    while s and s[0] in PUNCT_FOR_WORDCLOUD:
        s = s[1:]
    while s and s[-1] in PUNCT_FOR_WORDCLOUD:
        s = s[:-1]
    return s.strip()

# 管理痛点/问题/期待相关触发词（用于从开放反馈中筛选有效表述）
PAIN_POINT_TRIGGERS = {
    "难", "不足", "缺乏", "希望", "需要", "问题", "挑战", "压力", "不够", "改善", "提升",
    "困惑", "不知道", "平衡", "时间", "精力", "带人", "管人", "辅导", "反馈", "授权",
    "激励", "任务", "沟通", "下属", "团队", "学习", "成长", "期待", "担心", "焦虑",
    "协调", "冲突", "效率", "方法", "技巧", "经验", "能力", "加强", "更多", "管理",
    "改进", "完善", "支持", "帮助", "指导", "培养", "发展", "角色", "转型",
}
# 按长度降序，用于分组时优先匹配长触发词（如「任务分配」先于「任务」）
TRIGGER_ORDER = sorted(PAIN_POINT_TRIGGERS, key=len, reverse=True)
# 触发词 -> 结论中的主题展示名
TRIGGER_DISPLAY = {
    "时间": "时间与精力分配",
    "精力": "时间与精力分配",
    "平衡": "时间与精力分配",
    "压力": "压力与心态",
    "焦虑": "压力与心态",
    "担心": "压力与心态",
    "辅导": "辅导与反馈",
    "反馈": "辅导与反馈",
    "沟通": "沟通与协作",
    "协调": "沟通与协作",
    "冲突": "沟通与协作",
    "授权": "授权与任务分配",
    "任务": "授权与任务分配",
    "激励": "激励与团队",
    "团队": "激励与团队",
    "下属": "激励与团队",
    "带人": "带人与管人",
    "管人": "带人与管人",
    "管理": "管理角色与转型",
    "角色": "管理角色与转型",
    "转型": "管理角色与转型",
    "学习": "学习与成长",
    "成长": "学习与成长",
    "能力": "能力与方法",
    "方法": "能力与方法",
    "技巧": "能力与方法",
    "经验": "能力与方法",
    "效率": "效率与改进",
    "改善": "效率与改进",
    "改进": "效率与改进",
    "提升": "提升与完善",
    "完善": "提升与完善",
    "希望": "期待与需求",
    "需要": "期待与需求",
    "期待": "期待与需求",
    "支持": "支持与指导",
    "帮助": "支持与指导",
    "指导": "支持与指导",
    "培养": "支持与指导",
    "发展": "支持与指导",
    "问题": "问题与挑战",
    "挑战": "问题与挑战",
    "困惑": "问题与挑战",
    "不知道": "问题与挑战",
    "难": "问题与挑战",
    "不足": "不足与缺乏",
    "缺乏": "不足与缺乏",
    "不够": "不足与缺乏",
    "更多": "更多诉求",
    "加强": "更多诉求",
}
# 主题在页面上的展示顺序（未出现在此列表中的主题排在最后，按条数降序）
THEME_DISPLAY_ORDER = [
    "问题与挑战",
    "时间与精力分配",
    "压力与心态",
    "辅导与反馈",
    "沟通与协作",
    "授权与任务分配",
    "激励与团队",
    "带人与管人",
    "管理角色与转型",
    "学习与成长",
    "能力与方法",
    "效率与改进",
    "提升与完善",
    "期待与需求",
    "支持与指导",
    "不足与缺乏",
    "更多诉求",
]

def _primary_trigger(phrase: str):
    """返回短语所属的主触发词（按 TRIGGER_ORDER 第一个匹配）。"""
    for t in TRIGGER_ORDER:
        if t in phrase:
            return t
    return None

def _dedupe_similar(phrases: list, max_repr: int = 2, sim_threshold: int = 0.6):
    """去重相似表述，保留最多 max_repr 条代表性表述。优先保留较完整（较长）的表述。"""
    if not phrases:
        return []
    sorted_p = sorted(phrases, key=len, reverse=True)
    kept = []
    for p in sorted_p:
        p_clean = p.strip()
        if len(p_clean) < 3:
            continue
        is_dup = False
        for k in kept:
            if p_clean in k or k in p_clean:
                is_dup = True
                break
            set_p, set_k = set(p_clean), set(k)
            overlap = len(set_p & set_k) / max(len(set_p), len(set_k), 1)
            if overlap >= sim_threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(p_clean)
        if len(kept) >= max_repr:
            break
    return kept[:max_repr]

def _summarise_pain_point_phrases(phrases: list):
    """
    将痛点相关表述按主题分组、去重后，生成结论式总结。
    返回 [(主题展示名, 该主题条数, 代表性表述列表), ...]，按条数降序。
    """
    if not phrases:
        return []
    by_trigger = defaultdict(list)
    for p in phrases:
        t = _primary_trigger(p)
        if t:
            by_trigger[t].append(p)
    # 合并到统一主题名
    theme_to_phrases = defaultdict(list)
    for t, plist in by_trigger.items():
        theme = TRIGGER_DISPLAY.get(t, t)
        theme_to_phrases[theme].extend(plist)
    # 每个主题去重、取代表
    out = []
    for theme, plist in theme_to_phrases.items():
        reprs = _dedupe_similar(plist, max_repr=4, sim_threshold=0.55)
        out.append((theme, len(plist), reprs))
    # 按 THEME_DISPLAY_ORDER 排序，未在列表中的主题按条数降序排在最后
    order_idx = {t: i for i, t in enumerate(THEME_DISPLAY_ORDER)}
    out.sort(key=lambda x: (order_idx.get(x[0], len(THEME_DISPLAY_ORDER)), -x[1]))
    return out

def _extract_pain_point_phrases(text: str, max_phrases: int = 30):
    """
    从反馈全文里筛出包含「管理痛点/问题/期待」相关词的句子或片段，用于聚焦呈现。
    按句切分（。！？；\\n），保留含触发词的片段，去重后返回列表。
    """
    if not (text or "").strip():
        return []
    import re
    raw = re.sub(r"[。！？；]", "\n", text)
    raw = re.sub(r"\n+", "\n", raw).strip()
    segments = [s.strip() for s in raw.split("\n") if len(s.strip()) >= 4]
    out = []
    seen = set()
    for s in segments:
        if len(out) >= max_phrases:
            break
        for t in PAIN_POINT_TRIGGERS:
            if t in s:
                key = s[:50]
                if key not in seen:
                    seen.add(key)
                    out.append(s)
                break
    return out

def _extract_pain_point_keywords(phrases: list, top_n: int = 20, min_word_len: int = 1):
    """仅在管理痛点相关片段中统计词频，返回 (词, 频次) 列表。标点不纳入统计。"""
    if not phrases:
        return []
    combined = " ".join(phrases)
    segs = jieba.lcut(combined)
    single_char_stop = {"的", "了", "是", "在", "我", "有", "和", "就", "不", "人", "都", "一", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没", "看", "好", "自", "这", "那", "等", "能", "与", "及", "或", "而", "把", "被", "让", "给", "无", "可", "以", "够", "些", "什", "么", "怎", "如", "为"}
    words = []
    for w in segs:
        w = _strip_punctuation_for_word(w)
        if _is_punctuation_only(w) or not w:
            continue
        if w in STOPWORDS_CN:
            continue
        if min_word_len <= 1:
            if len(w) >= 2 or (len(w) == 1 and w not in single_char_stop):
                words.append(w)
        else:
            if len(w) >= min_word_len:
                words.append(w)
    freq = Counter(words)
    return freq.most_common(top_n)

def _font_candidates_in_dir(directory: str):
    """在指定目录下生成 fonts/ 中字体候选路径。Pillow/WordCloud 仅可靠支持 TTF，优先 TTF。"""
    if not directory:
        return []
    return [os.path.join(directory, "fonts", name) for name in (
        "NotoSansSC-Regular.ttf", "font.ttf", "NotoSansCJK-Regular.ttc",
        "NotoSansSC-Regular.otf",
    )]


def _get_chinese_font_path(app_dir: str = None):
    """返回系统可用的中文字体路径，用于词云（兼容 macOS / Windows / Linux 线上环境）。"""
    # 1) 优先使用应用目录下捆绑字体（线下=__file__ 所在目录，线上=可能用 getcwd）
    for base in ([app_dir] if app_dir else []) + [os.getcwd()]:
        for path in _font_candidates_in_dir(base):
            if path and os.path.isfile(path):
                return path
    # 2) 系统字体路径
    candidates = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Supplemental/Songti.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
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
    # 3) matplotlib 字体列表中的 CJK
    try:
        import matplotlib.font_manager as fm
        for f in fm.fontManager.ttflist:
            path = getattr(f, "fname", None)
            if not path or not os.path.isfile(path):
                continue
            name = (f.name or "").lower()
            if "noto" in name or "cjk" in name or ("sans" in name and ("sc" in name or "tc" in name or "jp" in name or "kr" in name)):
                return path
    except Exception:
        pass
    # 4) 下载并缓存（多 URL、缓存字节，提高线上成功率）
    return _download_chinese_font_cached()


@st.cache_data(ttl=3600)
def _fetch_font_bytes():
    """下载中文字体 TTF 字节并缓存（Pillow/WordCloud 对 OTF 易报 unknown file format）。"""
    # 优先 TTF：Pillow/WordCloud 对 OTF 会报 unknown file format
    urls = [
        "https://cdn.jsdelivr.net/gh/jsntn/webfonts@master/NotoSansSC-Regular.ttf",
        "https://cdn.jsdelivr.net/gh/google/fonts@main/ofl/notosanssc/NotoSansSC%5Bwght%5D.ttf",
    ]
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; Streamlit)"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            if len(data) > 50000:
                return data
        except Exception:
            continue
    return None


def _download_chinese_font_cached():
    """无系统字体时下载并缓存中文字体到临时文件（TTF），返回路径；失败返回 None。"""
    cache_dir = tempfile.gettempdir()
    cache_path = os.path.join(cache_dir, "NotoSansSC-wordcloud.ttf")
    if os.path.isfile(cache_path):
        return cache_path
    data = _fetch_font_bytes()
    if data:
        try:
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


def build_wordcloud_image(text: str, width=900, height=380, mask_dir: str = None, use_mask: bool = True, min_word_length: int = 2):
    """
    根据反馈文本生成词云图：红/橙配色，可选文字围绕卡通形象（保持比例）。
    min_word_length：最小词长，1 时允许单字（会过滤无意义单字），便于呈现「难」「力」等与管理问题相关的词。
    返回 (PNG 字节流, 高频词列表, 错误信息)；成功时错误信息为 None。
    """
    text = (text or "").strip()
    if not text:
        return None, [], None
    segs = jieba.lcut(text)
    single_char_stop = {"的", "了", "是", "在", "我", "有", "和", "就", "不", "人", "都", "一", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没", "看", "好", "自", "这", "那", "等", "能", "与", "及", "或", "而", "把", "被", "让", "给", "无", "可", "以", "够", "些", "什", "么", "怎", "如", "为"}
    words = []
    for w in segs:
        w = _strip_punctuation_for_word(w)
        if _is_punctuation_only(w) or not w:
            continue
        if w in STOPWORDS_CN:
            continue
        if min_word_length <= 1:
            if len(w) >= 2 or (len(w) == 1 and w not in single_char_stop):
                words.append(w)
        else:
            if len(w) >= min_word_length:
                words.append(w)
    if not words and len(text) >= 2:
        for w in segs:
            w = _strip_punctuation_for_word(w)
            if _is_punctuation_only(w) or not w or w in STOPWORDS_CN:
                continue
            if len(w) >= min_word_length or (min_word_length <= 1 and len(w) == 1 and w not in single_char_stop):
                words.append(w)
    if not words:
        return None, [], None
    freq = Counter(words)
    top_words = [w for w, _ in freq.most_common(25)]
    # 线下=__file__ 目录，线上=再试 getcwd，保证能找到 fonts/
    font_path = _get_chinese_font_path(mask_dir)
    mask, overlay_img = None, None
    if mask_dir and use_mask:
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
    err_msg = None
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
        return buf, top_words, None
    except Exception as e:
        err_msg = str(e)
        kw.pop("mask", None)
        kw.pop("contour_width", None)
        kw.pop("contour_color", None)
        try:
            wc = WordCloud(**kw)
            wc.generate_from_frequencies(freq)
            buf = io.BytesIO()
            wc.to_image().save(buf, format="PNG")
            buf.seek(0)
            return buf, top_words, None
        except Exception as e2:
            return None, [], (err_msg + "; 无蒙版重试: " + str(e2))

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

def _set_matplotlib_chinese_font(app_dir=None):
    """
    为 matplotlib 设置中文字体，避免 PDF 导出图表在云端/无中文字体环境下乱码。
    优先使用应用目录 fonts/ 下字体（TTF/OTF），再回退到 _get_chinese_font_path。
    """
    candidates = []
    if app_dir:
        for name in ("NotoSansSC-Regular.ttf", "font.ttf", "NotoSansSC-Regular.otf", "NotoSansCJK-Regular.ttc"):
            p = os.path.join(app_dir, "fonts", name)
            if os.path.isfile(p):
                candidates.append(p)
    path = None
    for p in candidates:
        path = p
        break
    if not path:
        path = _get_chinese_font_path(app_dir)
    if not path or not os.path.isfile(path):
        return
    try:
        import matplotlib.font_manager as fm
        fm.fontManager.addfont(path)
        prop = fm.FontProperties(fname=path)
        name = prop.get_name()
        plt.rcParams["font.sans-serif"] = [name, "SimHei", "PingFang SC", "Microsoft YaHei", "DejaVu Sans", "sans-serif"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


def _radar_chart_matplotlib(dim_labels, person_vals, out_buffer, avg_vals=None, app_dir=None):
    """用 matplotlib 绘制五维度雷达图。可叠加全员平均线对比。dim_labels 与 person_vals 长度均为 5。"""
    _set_matplotlib_chinese_font(app_dir)
    plt.rcParams["font.sans-serif"] = plt.rcParams.get("font.sans-serif", ["SimHei", "PingFang SC", "Microsoft YaHei", "DejaVu Sans", "sans-serif"])
    plt.rcParams["axes.unicode_minus"] = False
    n = len(dim_labels)
    if n == 0:
        return
    angles = [2 * math.pi * i / n for i in range(n)]
    angles_close = angles + [angles[0]]

    def _safe_vals(vals, length):
        out = []
        for v in (vals or [])[:length]:
            try:
                x = float(v)
                out.append(x if x == x else 0.0)
            except (TypeError, ValueError):
                out.append(0.0)
        while len(out) < length:
            out.append(0.0)
        return out[:length]

    vals = _safe_vals(person_vals, n)
    vals_close = vals + [vals[0]]
    fig, ax = plt.subplots(figsize=(3.2, 3.2), subplot_kw=dict(projection="polar"))
    # 全员平均（先画，在底层）
    if avg_vals is not None and len(avg_vals) >= n:
        avg = _safe_vals(avg_vals, n)
        avg_close = avg + [avg[0]]
        ax.fill(angles_close, avg_close, alpha=0.15, color="#94a3b8")
        ax.plot(angles_close, avg_close, "o-", linewidth=1.5, color="#94a3b8", linestyle="--", markersize=3, label="全员平均")
    # 员工自评（后画，在上层）
    ax.fill(angles_close, vals_close, alpha=0.25, color="#3498DB")
    ax.plot(angles_close, vals_close, "o-", linewidth=2, color="#3498DB", markersize=4, label="员工自评")
    ax.set_xticks(angles)
    ax.set_xticklabels(dim_labels, size=9)
    ax.set_ylim(0, 5.5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_buffer, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    out_buffer.seek(0)


def _line_chart_behavior_matplotlib(labels, values, out_buffer, color_scheme=None, app_dir=None):
    """用 matplotlib 绘制模块+行为项得分折线图。x=模块-行为项，y=得分；按模块着色。PDF 导出时纵轴统一为 1～5、步长 1。"""
    _set_matplotlib_chinese_font(app_dir)
    plt.rcParams["font.sans-serif"] = plt.rcParams.get("font.sans-serif", ["SimHei", "PingFang SC", "Microsoft YaHei", "DejaVu Sans", "sans-serif"])
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x_pos = list(range(len(labels)))
    vals = [float(v) if v is not None and str(v) != "nan" else 0.0 for v in values]
    if not vals:
        out_buffer.seek(0)
        return
    scheme = color_scheme or {}
    modules = [lab.split("-", 1)[0] if "-" in lab else "" for lab in labels]
    colors = [scheme.get(m, "#2980B9") for m in modules]
    for i in range(len(x_pos) - 1):
        ax.plot(x_pos[i : i + 2], vals[i : i + 2], "-", color=colors[i], linewidth=2)
    ax.scatter(x_pos, vals, c=colors, s=28, zorder=5, edgecolors="white", linewidths=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("得分")
    ax.set_xlabel("模块-行为项")
    ax.set_ylim(1, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.grid(True, linestyle="--", alpha=0.6)
    seen = []
    for m in modules:
        if m and m not in seen and m in scheme:
            seen.append(m)
    if seen:
        from matplotlib.patches import Patch
        ax.legend(
            [Patch(facecolor=scheme[m], edgecolor="none") for m in seen],
            seen,
            loc="upper right",
            fontsize=8,
            ncol=2,
        )
    fig.tight_layout()
    fig.savefig(out_buffer, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    out_buffer.seek(0)


def _summary_chart_matplotlib(dims, scores, bar_colors, out_buffer, app_dir=None):
    """用 matplotlib 绘制五维度得分柱状图并写入 out_buffer（kaleido 不可用时的备选）。"""
    _set_matplotlib_chinese_font(app_dir)
    plt.rcParams["font.sans-serif"] = plt.rcParams.get("font.sans-serif", ["SimHei", "PingFang SC", "Microsoft YaHei", "DejaVu Sans", "sans-serif"])
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(10, 4))
    x_pos = range(len(dims))
    bars = ax.bar(x_pos, scores, color=bar_colors, edgecolor="gray", linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(dims, rotation=25, ha="right")
    ax.set_ylabel("得分")
    ax.set_xlabel("维度")
    ax.set_ylim(0, 5.5)
    ax.set_yticks([0, 1, 2, 3, 4, 5])
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    for b, v in zip(bars, scores):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.08, "%.2f" % v, ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_buffer, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    out_buffer.seek(0)


def _pie_chart_matplotlib(labels, values, colors, title: str, out_buffer, app_dir=None):
    """用 matplotlib 绘制饼图并写入 out_buffer，用于 PDF 摘要页。labels/values/colors 同长，title 为图标题。"""
    if not labels or not values or len(labels) != len(values):
        return
    _set_matplotlib_chinese_font(app_dir)
    plt.rcParams["font.sans-serif"] = plt.rcParams.get("font.sans-serif", ["SimHei", "PingFang SC", "Microsoft YaHei", "DejaVu Sans", "sans-serif"])
    plt.rcParams["axes.unicode_minus"] = False
    # 使用较大 figsize + 高 dpi，避免 PDF 中嵌入时模糊
    fig, ax = plt.subplots(figsize=(3.8, 3.6))
    cols = colors if len(colors) >= len(labels) else (colors * ((len(labels) // len(colors)) + 1))[:len(labels)]
    wedges, _, autotexts = ax.pie(values, labels=labels, colors=cols, autopct="%1.0f%%", startangle=90, textprops={"fontsize": 8})
    for t in autotexts:
        t.set_fontsize(7)
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_buffer, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    out_buffer.seek(0)


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

# 多选「希望深入学习的技能模块」统计（全局概览 + PDF 共用）
learning_col = LEARNING_MODULE_COL if LEARNING_MODULE_COL in df.columns else next(
    (c for c in df.columns if "技能模块" in str(c) and "深入" in str(c)), None
)
learning_module_votes = []
if learning_col is not None:
    _counts = Counter()
    for val in df[learning_col].dropna().astype(str):
        _v = val
        for sep in ["，", "、", "；", ";", ",", "\n"]:
            _v = _v.replace(sep, "\t")
        for part in _v.split("\t"):
            token = part.strip()
            if token in CATEGORY_ORDER:
                _counts[token] += 1
    learning_module_votes = sorted(_counts.items(), key=lambda x: -x[1])

# 管理年限分布统计（全局概览）
tenure_col = TENURE_COL if TENURE_COL in df.columns else next(
    (c for c in df.columns if "带团队" in str(c) and "多久" in str(c)), None
)
tenure_votes = []
if tenure_col is not None:
    s = df[tenure_col].fillna("未填写").astype(str).str.strip()
    s = s.replace("", "未填写").replace("nan", "未填写")
    vc = s.value_counts()
    tenure_votes = [(str(k), int(v)) for k, v in vc.items() if str(k).strip()]
    tenure_votes.sort(key=lambda x: -x[1])

# 团队规模分布统计（全局概览）
team_size_col = TEAM_SIZE_COL if TEAM_SIZE_COL in df.columns else next(
    (c for c in df.columns if "汇报" in str(c) and "伙伴" in str(c)), None
)
team_size_votes = []
if team_size_col is not None:
    s = df[team_size_col].fillna("未填写").astype(str).str.strip()
    s = s.replace("", "未填写").replace("nan", "未填写")
    vc = s.value_counts()
    team_size_votes = [(str(k), int(v)) for k, v in vc.items() if str(k).strip()]
    team_size_votes.sort(key=lambda x: -x[1])

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
    st.markdown("---")
    st.markdown("### 📥 生成 PDF 报告")
    if not REPORTLAB_AVAILABLE:
        err_detail = (" " + REPORTLAB_IMPORT_ERROR) if REPORTLAB_IMPORT_ERROR else ""
        st.warning(
            "PDF 功能需要 reportlab。请点击下方按钮用当前环境安装，安装后页面会自动刷新。"
            + (("\n\n导入报错：" + err_detail) if err_detail else "")
        )
        if st.button("用当前环境安装 reportlab", key="install_reportlab", use_container_width=True):
            with st.spinner("正在安装 reportlab..."):
                r = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "reportlab"],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
            if r.returncode == 0:
                st.success("安装成功，正在刷新页面…")
                st.rerun()
            else:
                st.error("安装失败。请在终端执行：**" + sys.executable + " -m pip install reportlab**\n\n" + (r.stderr or r.stdout or ""))
    elif st.button("📥 生成 PDF 报告", key="gen_pdf", use_container_width=True):
        with st.spinner("报告正在生成中，请稍候…"):
            _app_dir = os.path.dirname(os.path.abspath(__file__))
            # 导出 PDF 时排除选择「我走专业路线，没有带团队」的伙伴（他们无带团队数据）
            mask_exclude = pd.Series(False, index=df.index)
            if TENURE_COL in df.columns:
                mask_exclude |= df[TENURE_COL].astype(str).str.contains(EXCLUDE_PDF_ROLE_LABEL, na=False)
            if TEAM_SIZE_COL in df.columns:
                mask_exclude |= df[TEAM_SIZE_COL].astype(str).str.contains(EXCLUDE_PDF_ROLE_LABEL, na=False)
            keep_idx = df.index[~mask_exclude]
            if len(keep_idx) == 0:
                st.error("当前数据中所有伙伴均选择「我走专业路线，没有带团队」，无法生成 PDF。")
            else:
                df_q_pdf = df_q.loc[keep_idx]
                df_dims_pdf = df_dims.loc[keep_idx]
                _name_col = next((c for c in ["填写人", "姓名", "学员姓名"] if c in df.columns), None)
                names_pdf = df.loc[keep_idx, _name_col].astype(str).tolist()
                total_pdf = total.loc[keep_idx]
                selected_name_pdf = selected_name if selected_name in names_pdf else (names_pdf[0] if names_pdf else None)
                # 从过滤后的数据重新统计技能模块 / 管理年限 / 团队规模（PDF 中不再含「专业路线」选项）
                df_pdf = df.loc[keep_idx]
                learning_module_votes_pdf = []
                if LEARNING_MODULE_COL in df_pdf.columns:
                    _counts = Counter()
                    for val in df_pdf[LEARNING_MODULE_COL].dropna().astype(str):
                        _v = val
                        for sep in ["，", "、", "；", ";", ",", "\n"]:
                            _v = _v.replace(sep, "\t")
                        for part in _v.split("\t"):
                            token = part.strip()
                            if token in CATEGORY_ORDER:
                                _counts[token] += 1
                    learning_module_votes_pdf = sorted(_counts.items(), key=lambda x: -x[1])
                tenure_votes_pdf = []
                if TENURE_COL in df_pdf.columns:
                    s = df_pdf[TENURE_COL].fillna("未填写").astype(str).str.strip()
                    s = s.replace("", "未填写").replace("nan", "未填写")
                    vc = s.value_counts()
                    tenure_votes_pdf = [(str(k), int(v)) for k, v in vc.items() if str(k).strip()]
                    tenure_votes_pdf.sort(key=lambda x: -x[1])
                team_size_votes_pdf = []
                if TEAM_SIZE_COL in df_pdf.columns:
                    s = df_pdf[TEAM_SIZE_COL].fillna("未填写").astype(str).str.strip()
                    s = s.replace("", "未填写").replace("nan", "未填写")
                    vc = s.value_counts()
                    team_size_votes_pdf = [(str(k), int(v)) for k, v in vc.items() if str(k).strip()]
                    team_size_votes_pdf.sort(key=lambda x: -x[1])

                dim_cols = [c for c in CATEGORY_ORDER if c in df_dims_pdf.columns]
                dim_means = [(c, float(df_dims_pdf[c].mean())) for c in dim_cols]
                behavior_avgs = get_behavior_avg_by_dimension(df_q_pdf, col_to_cat_be)
                summary_chart_png = io.BytesIO()
                radar_png = io.BytesIO()
                summary = pd.DataFrame({"维度": [x[0] for x in dim_means], "全员平均分": [x[1] for x in dim_means]})
                dims = summary["维度"].tolist()
                scores = summary["全员平均分"].values
                bar_colors = [COLOR_SCHEME.get(d, "#3498db") for d in dims]
                # PDF 导出优先用 matplotlib 生成图表，确保云端/多设备下中文字体正确显示，避免 kaleido 无中文字体乱码
                try:
                    _summary_chart_matplotlib(dims, scores, bar_colors, summary_chart_png, app_dir=_app_dir)
                except Exception:
                    try:
                        fig_summary = go.Figure(data=[go.Bar(
                            x=dims,
                            y=summary["全员平均分"],
                            marker_color=bar_colors,
                            text=summary["全员平均分"].round(2),
                            texttemplate="%{text:.2f}",
                            textposition="outside",
                            textfont=dict(size=12),
                        )])
                        fig_summary.update_layout(
                            xaxis_title="维度",
                            yaxis_title="得分",
                            yaxis=dict(range=[0, 5.5], dtick=1, showgrid=True, gridcolor="#e8e8e8"),
                            height=340,
                            margin=dict(b=100, t=50, l=60, r=40),
                            showlegend=False,
                        )
                        fig_summary.update_xaxes(tickangle=-25, tickfont=dict(size=11))
                        fig_summary = apply_chart_style(fig_summary)
                        img_bytes = fig_summary.to_image(format="png", engine="kaleido")
                        summary_chart_png.write(img_bytes)
                    except Exception:
                        pass
                summary_chart_png.seek(0)
                if len(summary_chart_png.getvalue()) == 0:
                    try:
                        _summary_chart_matplotlib(dims, scores, bar_colors, summary_chart_png, app_dir=_app_dir)
                    except Exception:
                        pass
                summary_chart_png.seek(0)
                behavior_chart_png = io.BytesIO()
                try:
                    labels_avg, values_avg = get_all_behavior_avgs(df_q_pdf, col_to_cat_be)
                    if labels_avg and values_avg:
                        _line_chart_behavior_matplotlib(labels_avg, values_avg, behavior_chart_png, color_scheme=COLOR_SCHEME, app_dir=_app_dir)
                except Exception:
                    pass
                behavior_chart_png.seek(0)
                # 三个饼图（希望深入学习的技能模块、管理年限、团队规模），放在报告摘要柱状图下方
                pie_learning_png = io.BytesIO()
                pie_tenure_png = io.BytesIO()
                pie_team_png = io.BytesIO()
                if learning_module_votes_pdf:
                    mod_names = [x[0] for x in learning_module_votes_pdf]
                    mod_counts = [x[1] for x in learning_module_votes_pdf]
                    pie_colors = [COLOR_SCHEME.get(m, "#3498db") for m in mod_names]
                    try:
                        _pie_chart_matplotlib(mod_names, mod_counts, pie_colors, "", pie_learning_png, app_dir=_app_dir)
                    except Exception:
                        pass
                if tenure_votes_pdf:
                    tenure_labels = [x[0] for x in tenure_votes_pdf]
                    tenure_counts = [x[1] for x in tenure_votes_pdf]
                    tenure_colors = [COLORS_BARS[i % len(COLORS_BARS)] for i in range(len(tenure_labels))]
                    try:
                        _pie_chart_matplotlib(tenure_labels, tenure_counts, tenure_colors, "管理年限分布", pie_tenure_png, app_dir=_app_dir)
                    except Exception:
                        pass
                if team_size_votes_pdf:
                    team_labels = [x[0] for x in team_size_votes_pdf]
                    team_counts = [x[1] for x in team_size_votes_pdf]
                    team_colors = [COLORS_BARS[i % len(COLORS_BARS)] for i in range(len(team_labels))]
                    try:
                        _pie_chart_matplotlib(team_labels, team_counts, team_colors, "团队规模分布", pie_team_png, app_dir=_app_dir)
                    except Exception:
                        pass
                try:
                    idx = names_pdf.index(selected_name_pdf)
                    row_index = df_q_pdf.index[idx]
                    row_dims = df_dims_pdf.loc[row_index, dim_cols] if dim_cols else pd.Series(dtype=float)
                    dim_means_all = df_dims_pdf[dim_cols].mean() if dim_cols else pd.Series(dtype=float)
                    theta_radar = dim_cols
                    r_person = [float(row_dims[c]) for c in theta_radar]
                    r_avg = [float(dim_means_all[c]) for c in theta_radar]
                    if len(r_person) == 5:
                        _radar_chart_matplotlib(theta_radar, r_person, radar_png, avg_vals=r_avg, app_dir=_app_dir)
                    else:
                        radar_png.seek(0)
                except Exception:
                    radar_png = io.BytesIO()
                person_scores = list(zip(names_pdf, [float(total_pdf.loc[df_q_pdf.index[i]]) for i in range(len(df_q_pdf))]))
                person_scores.sort(key=lambda x: x[1], reverse=True)
                top3_high = person_scores[:3]
                top3_low = person_scores[-3:][::-1] if len(person_scores) >= 3 else person_scores[::-1]
                score_cols = list(col_to_cat_be.keys())
                anomaly_rows = []
                name_col_anom = next((c for c in ["填写人", "姓名", "学员姓名"] if c in df.columns), None)
                dept_col_anom = "部门" if "部门" in df.columns else None
                for idx in df_q_pdf.index:
                    row = df_q_pdf.loc[idx, score_cols]
                    valid = row.dropna()
                    if len(valid) >= 1 and valid.nunique() == 1:
                        uniform_score = float(valid.iloc[0])
                        name = df.loc[idx, name_col_anom] if name_col_anom else str(idx)
                        dept = df.loc[idx, dept_col_anom] if dept_col_anom else None
                        note = f"该伙伴所有题目均为 {uniform_score:.1f} 分，建议管理者关注。"
                        anomaly_rows.append((name, dept, uniform_score, note))
                summary_votes = learning_module_votes_pdf
                dim_means_all = df_dims_pdf[dim_cols].mean() if dim_cols else pd.Series(dtype=float)
                avg_dims = [float(dim_means_all[c]) for c in dim_cols] if len(dim_cols) == 5 else None
                person_details = []
                for i in range(len(names_pdf)):
                    name = names_pdf[i]
                    row_index = df_q_pdf.index[i]
                    radar_io = io.BytesIO()
                    line_io = io.BytesIO()
                    try:
                        row_dims = df_dims_pdf.loc[row_index, dim_cols] if dim_cols else pd.Series(dtype=float)
                        person_dims = [float(row_dims[c]) for c in dim_cols] if len(dim_cols) == 5 else []
                        if len(person_dims) == 5:
                            _radar_chart_matplotlib(dim_cols, person_dims, radar_io, avg_vals=avg_dims, app_dir=_app_dir)
                    except Exception:
                        pass
                    try:
                        labels, values = get_person_behavior_scores(df_q_pdf, col_to_cat_be, row_index)
                        if labels and values:
                            _line_chart_behavior_matplotlib(labels, values, line_io, color_scheme=COLOR_SCHEME, app_dir=_app_dir)
                    except Exception:
                        pass
                    person_details.append((name, radar_io, line_io))
                try:
                    report = PDFReport(app_dir=_app_dir, report_type="team")
                    pdf_buf = report.build(
                        dim_means=dim_means,
                        summary_chart_png=summary_chart_png,
                        pie_learning_png=pie_learning_png,
                        pie_tenure_png=pie_tenure_png,
                        pie_team_png=pie_team_png,
                        behavior_avgs=behavior_avgs,
                        behavior_chart_png=behavior_chart_png,
                        radar_images=[radar_png],
                        top3_high=top3_high,
                        top3_low=top3_low,
                        anomaly_rows=anomaly_rows,
                        names=names_pdf,
                        selected_name=selected_name_pdf,
                        summary_votes=summary_votes,
                        tenure_votes=tenure_votes_pdf,
                        team_size_votes=team_size_votes_pdf,
                        person_details=person_details,
                    )
                    st.session_state["pdf_report_bytes"] = pdf_buf.getvalue()
                    st.success("PDF 已生成，请点击下方下载。")
                except Exception as e:
                    st.error("PDF 生成失败：" + str(e))
    if "pdf_report_bytes" in st.session_state:
        st.download_button(
            "下载 好未来新灵秀报告.pdf",
            data=st.session_state["pdf_report_bytes"],
            file_name="好未来新灵秀报告.pdf",
            key="dl_pdf",
            use_container_width=True,
        )

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

    # 希望深入学习的技能模块 + 管理年限 + 团队规模（三列并列）
    st.markdown("---")
    col_learning, col_tenure, col_team = st.columns(3)
    # 图例放在下方，预留足够底部边距避免遮挡饼图；左右对称便于不同宽度自适应
    _pie_height = 440
    _pie_margin = dict(t=40, b=120, l=55, r=55)
    _pie_legend = dict(
        orientation="h",
        yanchor="top",
        y=0.08,
        xanchor="center",
        x=0.5,
        font=dict(size=9),
        tracegroupgap=10,
        itemwidth=24,
    )

    with col_learning:
        st.markdown("#### 希望深入学习的技能模块")
        st.caption("多选：「您希望在以下哪个技能模块进行深入的学习和研讨？」")
        if not learning_module_votes:
            st.info("未找到该多选题目或暂无有效选项。")
        else:
            mod_names = [x[0] for x in learning_module_votes]
            mod_counts = [x[1] for x in learning_module_votes]
            total_votes = sum(mod_counts)
            pie_colors = [COLOR_SCHEME.get(m, "#3498db") for m in mod_names]
            fig_learning = go.Figure(data=[go.Pie(
                labels=mod_names,
                values=mod_counts,
                marker_colors=pie_colors,
                textinfo="label+percent+value",
                texttemplate="%{label}<br>%{percent}（%{value} 票）",
                textposition="outside",
                hole=0.4,
            )])
            fig_learning.update_layout(
                height=_pie_height,
                margin=_pie_margin,
                showlegend=True,
                legend=_pie_legend,
            )
            fig_learning = apply_chart_style(fig_learning)
            st.plotly_chart(fig_learning, use_container_width=True, config=PLOTLY_CONFIG)
            if learning_module_votes:
                top_mod, top_cnt = learning_module_votes[0]
                st.caption(f"最受期待：**{top_mod}**（{top_cnt} 票，" + (f"{100*top_cnt/total_votes:.1f}%" if total_votes else "") + "）")

    with col_tenure:
        st.markdown("#### 管理年限分布")
        st.caption("「您开始带团队有多久啦？」")
        if not tenure_votes:
            st.info("未找到该题目或暂无有效选项。")
        else:
            tenure_labels = [x[0] for x in tenure_votes]
            tenure_counts = [x[1] for x in tenure_votes]
            total_tenure = sum(tenure_counts)
            n_tenure = len(tenure_labels)
            tenure_colors = [COLORS_BARS[i % len(COLORS_BARS)] for i in range(n_tenure)]
            fig_tenure = go.Figure(data=[go.Pie(
                labels=tenure_labels,
                values=tenure_counts,
                marker_colors=tenure_colors,
                textinfo="label+percent+value",
                texttemplate="%{label}<br>%{percent}（%{value} 人）",
                textposition="outside",
                hole=0.4,
            )])
            fig_tenure.update_layout(
                height=_pie_height,
                margin=_pie_margin,
                showlegend=True,
                legend=_pie_legend,
            )
            fig_tenure = apply_chart_style(fig_tenure)
            st.plotly_chart(fig_tenure, use_container_width=True, config=PLOTLY_CONFIG)
            if tenure_votes:
                top_tenure, top_n = tenure_votes[0]
                st.caption(f"人数最多：**{top_tenure}**（{top_n} 人，" + (f"{100*top_n/total_tenure:.1f}%" if total_tenure else "") + "）")

    with col_team:
        st.markdown("#### 团队规模分布")
        st.caption("「向您直接汇报的伙伴有多少？」")
        if not team_size_votes:
            st.info("未找到该题目或暂无有效选项。")
        else:
            team_labels = [x[0] for x in team_size_votes]
            team_counts = [x[1] for x in team_size_votes]
            total_team = sum(team_counts)
            n_team = len(team_labels)
            team_colors = [COLORS_BARS[i % len(COLORS_BARS)] for i in range(n_team)]
            fig_team = go.Figure(data=[go.Pie(
                labels=team_labels,
                values=team_counts,
                marker_colors=team_colors,
                textinfo="label+percent+value",
                texttemplate="%{label}<br>%{percent}（%{value} 人）",
                textposition="outside",
                hole=0.4,
            )])
            fig_team.update_layout(
                height=_pie_height,
                margin=_pie_margin,
                showlegend=True,
                legend=_pie_legend,
            )
            fig_team = apply_chart_style(fig_team)
            st.plotly_chart(fig_team, use_container_width=True, config=PLOTLY_CONFIG)
            if team_size_votes:
                top_team, top_n = team_size_votes[0]
                st.caption(f"人数最多：**{top_team}**（{top_n} 人，" + (f"{100*top_n/total_team:.1f}%" if total_team else "") + "）")

# ---------- Tab 2: 维度深度分析（数据可视化 · 多维分析） ----------
with tab2:
    st.markdown("#### 各维度行为项得分（全员平均）")
    st.caption("针对同一主题的多个维度分析，便于发现各维度下的强弱行为项。")
    # (模块, 行为项) -> 完整行为描述，用于柱状图 hover
    _behavior_desc = {(m, b): d for m, b, d in SURVEY_QUESTIONS}
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
                    hovertext=[f"{be}，{_behavior_desc.get((cat, be), '')}" for be in be_names],
                    hoverinfo="text",
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

# ---------- Tab 3: 个人详细报告（与左侧栏学员筛选联动） ----------
with tab3:
    # 与左侧栏「学员筛选」共用同一选择，无需重复选
    st.markdown(f"#### 当前学员：**{selected_name}**")
    st.caption("在左侧边栏「学员筛选」中切换学员，本页会同步更新。")

    idx = names.index(selected_name)
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
            above_text = f'<p style="margin:4px 0 0 0; font-size:14px; line-height:1.5;"><strong>💪 高于全员</strong>：{selected_name} 在「{dims_joined}」上达到或超过全员平均。</p>'
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
        st.markdown(f"**{selected_name}** · 五维度得分 vs 全员均分")
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
                name=selected_name,
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

    # 3. 各行为项得分折线图（整体布局：标题→说明→图表→图例，间距与字号统一）
    st.markdown("#### 各行为项得分")
    st.caption("模块名称居中显示在各色块正上方，中间为得分趋势，下方为行为项；y 轴 0～5.5。")
    st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)  # 标题/说明与图表间距
    try:
        labels_line, values_line = get_person_behavior_scores(df_q, col_to_cat_be, row_index)
        if labels_line and values_line:
            vals = [float(v) if v is not None and str(v) != "nan" else 0.0 for v in values_line]
            n = len(labels_line)
            ticktext = []
            for lab in labels_line:
                if "-" in lab:
                    _, be = lab.split("-", 1)
                    ticktext.append(be)
                else:
                    ticktext.append(lab)
            y_min, y_max = 0.0, 5.5
            fig_line = go.Figure()
            shapes = []
            annotations = []
            for mod in CATEGORY_ORDER:
                indices = [i for i in range(n) if labels_line[i].startswith(mod + "-")]
                if not indices:
                    continue
                y_mod = [vals[i] for i in indices]
                color = COLOR_SCHEME.get(mod, "#2980B9")
                text_mod = [f"{y_mod[j]:.2f}" for j in range(len(y_mod))]
                fig_line.add_trace(go.Scatter(
                    x=indices,
                    y=y_mod,
                    mode="lines+markers+text",
                    text=text_mod,
                    textposition="top center",
                    textfont=dict(size=9, color=color),
                    line=dict(color=color, width=2.5),
                    marker=dict(size=9, color=color, line=dict(width=0.8, color="white")),
                    name=mod,
                ))
                i0, i1 = min(indices), max(indices)
                x_center = (i0 + i1) / 2.0
                shapes.append(dict(
                    type="rect",
                    xref="x", yref="y",
                    x0=i0 - 0.5, x1=i1 + 0.5,
                    y0=y_min, y1=y_max,
                    fillcolor=color,
                    opacity=0.12,
                    layer="below",
                    line=dict(width=0),
                ))
                # 模块名：字体放大加粗、向上贴顶一大块
                annotations.append(dict(
                    x=x_center,
                    y=0.998,
                    xref="x",
                    yref="paper",
                    text=mod,
                    showarrow=False,
                    font=dict(size=18, color=color, family="SimHei, Microsoft YaHei Bold, PingFang SC, sans-serif"),
                    xanchor="center",
                    yanchor="bottom",
                ))
            # 四块分层：①顶部模块名 ②折线图 ③行为项 ④图例紧贴行为项下方
            fig_line.update_layout(
                xaxis=dict(
                    tickmode="array",
                    tickvals=list(range(n)),
                    ticktext=ticktext,
                    tickangle=-32,
                    tickfont=dict(size=9),
                    ticklen=4,
                    title=None,
                ),
                yaxis=dict(
                    range=[y_min, y_max],
                    dtick=1,
                    showgrid=True,
                    gridcolor="#F0F0F0",
                    title="得分",
                    title_font=dict(size=11),
                ),
                height=550,
                margin=dict(l=54, r=40, t=75, b=130),
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=0.002,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=10),
                    bgcolor="rgba(248,248,248,0.95)",
                    bordercolor="rgba(200,200,200,0.6)",
                    borderwidth=1,
                ),
                shapes=shapes,
                annotations=annotations,
            )
            fig_line = apply_chart_style(fig_line)
            st.plotly_chart(fig_line, use_container_width=True, config=PLOTLY_CONFIG)
        else:
            st.info("暂无行为项得分数据。")
    except Exception as e:
        st.warning("折线图生成失败，请确认数据完整。")

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
    # 兼容列名标点差异（全角/半角）：若配置列未匹配，则用包含关键字的列
    if not open_cols:
        for col in df.columns:
            if isinstance(col, str) and ("培训" in col and "期待" in col) or "开放" in col or "反馈" in col:
                open_cols.append(col)
                break
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
