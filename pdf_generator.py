# -*- coding: utf-8 -*-
"""
好未来新灵秀报告 PDF 生成模块
支持个人报告与团队报告，中文显示，专业排版。所有输出使用 BytesIO，不写临时文件。
"""

import io
import os
from datetime import datetime
from xml.sax.saxutils import escape

REPORTLAB_IMPORT_ERROR = None
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether,
    )
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
    REPORTLAB_AVAILABLE = True
    REPORTLAB_IMPORT_ERROR = None
    # 摘要页左右边距与 SimpleDocTemplate 一致，用于表格总宽=可用宽度，保证与段落左对齐
    _FRAME_WIDTH_CM = 21 - 2 - 2  # A4 21cm，左右边距各 2cm
except ImportError as e:
    REPORTLAB_AVAILABLE = False
    REPORTLAB_IMPORT_ERROR = str(e)

try:
    from config import COLOR_SCHEME, SURVEY_QUESTIONS, SCORE_MAP
except ImportError:
    COLOR_SCHEME = {}
    SURVEY_QUESTIONS = []
    SCORE_MAP = {}

# 分值 -> 量表选项文案（用于摘要第一段，根据实际均分区间呈现）
SCORE_TO_LABEL = {v: k for k, v in (SCORE_MAP if SCORE_MAP else {"总是如此": 5, "经常如此": 4, "有时如此": 3, "很少如此": 2, "从未展现": 1}).items()}


# 页眉左侧文案
HEADER_LEFT = "Talent AI Insights"

# 第一页：第一部分标红文案
PREFACE_RED_TITLE = "！！！很重要！！！在您阅读报告之前，我们希望您能知悉"
PREFACE_RED_LINE = "这不是一份领导力评估报告"
# 第一页：温馨提示（单独模块标黄）
PREFACE_TIP = "【温馨提示】本报告结果是根据员工的自陈得出，请结合具体情况，根据员工日常表现以及360评价对各项数据进行理性的阐释，而不是单纯以分数论事，绝不能作为给员工贴标签的依据。"

# 自评分数换算规则（第一页表格）
SCORE_RULES_DATA = [
    ["分数", "含义"],
    ["5", "总是如此"],
    ["4", "经常如此"],
    ["3", "有时如此"],
    ["2", "很少如此"],
    ["1", "从未展现"],
]


def _register_chinese_font(app_dir: str = None):
    """注册中文字体，优先 TTF。返回注册后的字体名称，失败返回 None（调用方用 Helvetica 回退）。"""
    if not REPORTLAB_AVAILABLE:
        return None
    font_name = "ChineseFont"
    candidates = []
    if app_dir:
        for name in ("NotoSansSC-Regular.ttf", "font.ttf", "NotoSansSC-Regular.otf"):
            path = os.path.join(app_dir, "fonts", name)
            if os.path.isfile(path):
                candidates.append(path)
    for base in ([app_dir] if app_dir else []) + [os.getcwd()]:
        if not base:
            continue
        for name in ("NotoSansSC-Regular.ttf", "font.ttf"):
            path = os.path.join(base, "fonts", name)
            if path not in candidates and os.path.isfile(path):
                candidates.append(path)
    system_paths = [
        "C:/Windows/Fonts/msyh.ttf",
        "C:/Windows/Fonts/simhei.ttf",
        "/System/Library/Fonts/PingFang.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ]
    for path in candidates + system_paths:
        if not path or not os.path.isfile(path):
            continue
        try:
            pdfmetrics.registerFont(TTFont(font_name, path))
            return font_name
        except Exception:
            continue
    return None


def _get_header_date():
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def _mean_score_to_label(mean_score):
    """将维度均分（1～5）映射为量表选项文案，用于摘要第一段。"""
    s = max(1, min(5, round(mean_score)))
    return SCORE_TO_LABEL.get(int(s), "—")


def _lighten_hex(hex_color: str, blend_white: float = 0.75):
    """将十六进制颜色与白色混合，得到更浅的底色，便于深色文字阅读。"""
    if not hex_color or not hex_color.startswith("#"):
        return "#f8f8f8"
    hex_color = hex_color.strip("#")
    if len(hex_color) != 6:
        return "#f8f8f8"
    try:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        r = int(r + (255 - r) * blend_white)
        g = int(g + (255 - g) * blend_white)
        b = int(b + (255 - b) * blend_white)
        return "#%02x%02x%02x" % (r, g, b)
    except Exception:
        return "#f8f8f8"


class PDFReport:
    """好未来新灵秀报告 PDF 生成器。所有内容写入内存 BytesIO，不落盘。"""

    def __init__(self, app_dir: str = None, report_type: str = "team"):
        """
        app_dir: 应用根目录，用于查找 fonts/ 下中文字体。
        report_type: "team" | "personal"
        """
        self.app_dir = app_dir
        self.report_type = report_type
        self.buffer = io.BytesIO()
        self.font_name = _register_chinese_font(app_dir)
        self.styles = None
        self.doc = None

    def _build_styles(self):
        base = getSampleStyleSheet()
        self.styles = {
            "title": ParagraphStyle(
                name="CustomTitle",
                parent=base["Title"],
                fontName=self.font_name or "Helvetica",
                fontSize=18,
                spaceAfter=12,
                alignment=TA_CENTER,
            ),
            "heading1": ParagraphStyle(
                name="H1",
                parent=base["Heading1"],
                fontName=self.font_name or "Helvetica",
                fontSize=14,
                spaceBefore=14,
                spaceAfter=8,
                alignment=TA_LEFT,
                leftIndent=0,
                firstLineIndent=0,
            ),
            "heading2": ParagraphStyle(
                name="H2",
                parent=base["Heading2"],
                fontName=self.font_name or "Helvetica",
                fontSize=12,
                spaceBefore=10,
                spaceAfter=6,
                alignment=TA_LEFT,
                leftIndent=0,
                firstLineIndent=0,
            ),
            "body": ParagraphStyle(
                name="Body",
                parent=base["Normal"],
                fontName=self.font_name or "Helvetica",
                fontSize=10,
                leading=14,
                spaceAfter=6,
                alignment=TA_LEFT,
                leftIndent=0,
                firstLineIndent=0,
            ),
            "subtitle": ParagraphStyle(
                name="Subtitle",
                parent=base["Normal"],
                fontName=self.font_name or "Helvetica",
                fontSize=14,
                spaceAfter=12,
                alignment=TA_CENTER,
            ),
            "table_cell": ParagraphStyle(
                name="TableCell",
                parent=base["Normal"],
                fontName=self.font_name or "Helvetica",
                fontSize=9,
                leading=11,
                spaceAfter=0,
                spaceBefore=0,
                leftIndent=0,
                rightIndent=0,
                alignment=TA_LEFT,
            ),
            "table_cell_center": ParagraphStyle(
                name="TableCellCenter",
                parent=base["Normal"],
                fontName=self.font_name or "Helvetica",
                fontSize=9,
                leading=11,
                spaceAfter=0,
                spaceBefore=0,
                alignment=TA_CENTER,
            ),
            "table_cell_tight": ParagraphStyle(
                name="TableCellTight",
                parent=base["Normal"],
                fontName=self.font_name or "Helvetica",
                fontSize=8,
                leading=9,
                spaceAfter=0,
                spaceBefore=0,
                leftIndent=0,
                rightIndent=0,
                alignment=TA_LEFT,
            ),
            "table_cell_center_tight": ParagraphStyle(
                name="TableCellCenterTight",
                parent=base["Normal"],
                fontName=self.font_name or "Helvetica",
                fontSize=8,
                leading=9,
                spaceAfter=0,
                spaceBefore=0,
                alignment=TA_CENTER,
            ),
            "table_cell_survey": ParagraphStyle(
                name="TableCellSurvey",
                parent=base["Normal"],
                fontName=self.font_name or "Helvetica",
                fontSize=10,
                leading=12,
                spaceAfter=0,
                spaceBefore=0,
                leftIndent=0,
                rightIndent=0,
                alignment=TA_LEFT,
            ),
            "table_cell_center_survey": ParagraphStyle(
                name="TableCellCenterSurvey",
                parent=base["Normal"],
                fontName=self.font_name or "Helvetica",
                fontSize=10,
                leading=12,
                spaceAfter=0,
                spaceBefore=0,
                alignment=TA_CENTER,
            ),
            "note_red": ParagraphStyle(
                name="NoteRed",
                parent=base["Normal"],
                fontName=self.font_name or "Helvetica",
                fontSize=9,
                leading=11,
                spaceAfter=4,
                spaceBefore=0,
                textColor=colors.HexColor("#c00000"),
            ),
        }
        return self.styles

    def _resolve_background_path(self):
        """解析底图路径：优先 assets/background.png，其次 background.png。"""
        if not self.app_dir:
            return None
        for name in ("assets/background.png", "background.png"):
            path = os.path.join(self.app_dir, name)
            if os.path.isfile(path):
                return path
        return None

    def _canvas_background(self, canvas, doc):
        """在页面最底层绘制底图（铺满 A4），降低不透明度避免遮挡正文。"""
        path = getattr(self, "_background_path", None)
        if not path or not os.path.isfile(path):
            return
        try:
            canvas.saveState()
            try:
                canvas.setOpacity(0.18)
            except Exception:
                pass
            canvas.drawImage(path, 0, 0, width=A4[0], height=A4[1])
            canvas.restoreState()
        except Exception:
            pass

    def _canvas_header(self, canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 9)
        canvas.drawString(2 * cm, A4[1] - 1.2 * cm, HEADER_LEFT)
        canvas.drawRightString(A4[0] - 2 * cm, A4[1] - 1.2 * cm, _get_header_date())
        canvas.restoreState()

    def build(
        self,
        preface_text: str = None,
        dim_means: list = None,
        summary_chart_png: io.BytesIO = None,
        pie_learning_png: io.BytesIO = None,
        pie_tenure_png: io.BytesIO = None,
        pie_team_png: io.BytesIO = None,
        behavior_avgs: dict = None,
        behavior_chart_png: io.BytesIO = None,
        radar_images: list = None,
        top3_high: list = None,
        top3_low: list = None,
        anomaly_rows: list = None,
        names: list = None,
        selected_name: str = None,
        summary_votes: list = None,
        tenure_votes: list = None,
        team_size_votes: list = None,
        person_details: list = None,
    ):
        """
        生成 PDF 到 self.buffer。
        dim_means: [(维度名, 均分), ...]
        behavior_avgs: { "管理角色认知": {"工作理念": 4.2, ...}, ... }
        radar_images: [BytesIO, ...] 雷达图 PNG 流（个人报告一个，团队报告可多个或占位）
        top3_high / top3_low: [(姓名, 总分), ...]
        anomaly_rows: [(姓名, 部门, 统一分值, 说明), ...]，部门可为 None
        summary_votes: [(模块名, 票数), ...] 希望重点学习的模块得票，按票数降序；无则传 None 或 []
        tenure_votes: [(选项, 人数), ...] 管理年限分布，按人数降序；无则传 None 或 []
        team_size_votes: [(选项, 人数), ...] 团队规模分布，按人数降序；无则传 None 或 []
        person_details: [(姓名, 雷达图BytesIO, 折线图BytesIO), ...] 学员自陈结果细则，每人一行
        """
        if not REPORTLAB_AVAILABLE:
            raise RuntimeError("reportlab 未安装，请执行: pip install reportlab")
        self._build_styles()
        self.doc = SimpleDocTemplate(
            self.buffer,
            pagesize=A4,
            rightMargin=2 * cm,
            leftMargin=2 * cm,
            topMargin=2.2 * cm,
            bottomMargin=1.5 * cm,
        )
        story = []

        # 第一页：封面 + 团队报告（居中）+ 第一部分（标红）+ 温馨提示（标黄）+ 第二部分 + 自评分数换算表
        story.append(Paragraph("好未来新灵秀报告", self.styles["title"]))
        story.append(Spacer(1, 0.5 * cm))
        report_subtitle = "个人报告" if self.report_type == "personal" else "团队报告"
        story.append(Paragraph(report_subtitle, self.styles["subtitle"]))
        story.append(Spacer(1, 0.5 * cm))

        # 第一部分（标红）
        story.append(Paragraph('<font color="red">%s</font>' % PREFACE_RED_TITLE, self.styles["body"]))
        story.append(Paragraph('<font color="red">%s</font>' % PREFACE_RED_LINE, self.styles["body"]))
        story.append(Spacer(1, 0.2 * cm))
        story.append(Paragraph(
            "本报告旨在呈现新灵秀课程的学员在不同管理动作上的自我评估结果，我们在设计本期课程的重点强调内容时将进行参考。把调研结果同步给您是希望：",
            self.styles["body"],
        ))
        story.append(Paragraph(
            "1. 为您提供一个视角，即：学员们眼中的自己在团队中是否充分展现了各方面管理动作，以便您在帮助学员校准自我认知时能有的放矢",
            self.styles["body"],
        ))
        story.append(Paragraph(
            "2. 帮助学员打开乔哈里窗盲区，结合您对学员们的了解，帮助大家看见一些他们自己没有察觉的优劣势，未来期待着您的点拨和指导",
            self.styles["body"],
        ))
        story.append(Paragraph(
            "3. 请您知悉这些优秀的伙伴们踏上了成长为更优秀管理者的旅途，一路上期待有您的关注和陪伴",
            self.styles["body"],
        ))
        story.append(Spacer(1, 0.2 * cm))
        # 温馨提示（单独模块标黄）
        tip_para = Paragraph(PREFACE_TIP, self.styles["body"])
        tip_table = Table([[tip_para]], colWidths=[14 * cm])
        tip_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fffacd")),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(tip_table)
        story.append(Spacer(1, 0.4 * cm))

        # 第二部分：调研题本设计说明
        story.append(Paragraph("第二部分  调研题本设计说明", self.styles["heading2"]))
        story.append(Paragraph(
            "本次调研在凯洛格（KeyLogic Group）金牌培养项目&lt;新经理成长地图&gt;的设计逻辑之上，融合好未来的集团特色，分别从管理角色认知、辅导、任务分配、激励和沟通 5 个维度对新任管理者的管理动作呈现情况进行调研。",
            self.styles["body"],
        ))
        story.append(Paragraph("赋分标准", self.styles["heading2"]))
        story.append(Paragraph(
            "每个行为项的评分范围为 1～5 分，分数越高则表示参调者们出现该类行为的频率越高，报告中【均分】代表多位参调者自我描述的平均。自评分数换算逻辑：",
            self.styles["body"],
        ))
        score_table = Table(SCORE_RULES_DATA, colWidths=[4 * cm, 8 * cm])
        score_table.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), self.font_name or "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f0f0")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ALIGN", (0, 0), (0, -1), "CENTER"),
            ("ALIGN", (1, 0), (1, -1), "LEFT"),
        ]))
        story.append(score_table)
        story.append(PageBreak())

        # 第二页：调研题目设置（整表铺满一页、不跨页；按可用高度均分行高，消除底部空白）
        if SURVEY_QUESTIONS:
            ps_cell = self.styles["table_cell_survey"]
            ps_center = self.styles["table_cell_center_survey"]
            q_data = [[
                Paragraph("模块", ps_center),
                Paragraph("行为项", ps_center),
                Paragraph("具体行为描述", ps_center),
            ]]
            row_modules = []
            for mod, be, desc in SURVEY_QUESTIONS:
                row_modules.append(mod)
                desc_safe = desc.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                q_data.append([
                    Paragraph(mod, ps_center),
                    Paragraph(be, ps_center),
                    Paragraph(desc_safe, ps_cell),
                ])
            # 第二页可用高度：页高 - 上下边距 - 标题与间距，均分给表头+数据行；略留余量避免跨页
            page_h_pt = A4[1]
            top_pt, bottom_pt = 2.2 * cm, 1.5 * cm
            heading_spacer_pt = 1.0 * cm + 0.3 * cm
            table_available_pt = (page_h_pt - top_pt - bottom_pt - heading_spacer_pt) * 0.94
            num_rows = len(q_data)
            row_height_pt = max(16, table_available_pt / num_rows)
            row_heights = [row_height_pt] * num_rows
            q_table = Table(
                q_data,
                colWidths=[3 * cm, 3 * cm, 10 * cm],
                rowHeights=row_heights,
                repeatRows=0,
            )
            styles = [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e0e0e0")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
            for r in range(1, len(q_data)):
                hex_color = COLOR_SCHEME.get(row_modules[r - 1], "#f0f0f0")
                light = _lighten_hex(hex_color, blend_white=0.78)
                try:
                    bg = colors.HexColor(light)
                except Exception:
                    bg = colors.HexColor("#f8f8f8")
                styles.append(("BACKGROUND", (0, r), (-1, r), bg))
            q_table.setStyle(TableStyle(styles))
            second_page_block = [
                Paragraph("调研题目设置", self.styles["heading1"]),
                Spacer(1, 0.3 * cm),
                q_table,
            ]
            story.append(KeepTogether(second_page_block))
        story.append(PageBreak())

        # 第三页：报告摘要（所有文字与表格左对齐，用单列全宽表格包裹）
        summary_rows = [Paragraph("一、报告摘要", self.styles["heading1"])]
        if dim_means:
            scores = [s for _, s in dim_means]
            min_s, max_s = min(scores), max(scores)
            low_label = _mean_score_to_label(min_s)
            high_label = _mean_score_to_label(max_s)
            if low_label == high_label:
                p1 = "管理者们（指受测人员）在 5 个维度上的自评行为展现基本都在&lt;%s&gt;水平。" % (low_label if low_label != "—" else "有时如此")
            elif low_label == "—" or high_label == "—":
                p1 = "管理者们（指受测人员）在 5 个维度上的自评行为展现基本在&lt;%s&gt;和&lt;%s&gt;之间。" % (low_label if low_label != "—" else "很少如此", high_label if high_label != "—" else "总是如此")
            else:
                p1 = "管理者们（指受测人员）在 5 个维度上的自评行为展现基本都在&lt;%s&gt;和&lt;%s&gt;之间。" % (low_label, high_label)
            summary_rows.append(Paragraph(p1, self.styles["body"]))
            dim_min_name = min(dim_means, key=lambda x: x[1])[0]
            dim_max_name = max(dim_means, key=lambda x: x[1])[0]
            p2 = (
                "横向比较来看，管理者们自我评价在【%s】维度展现的行为稍显不足，在大家看来自己在这部分的管理动作展现不是特别的充分，"
                "而在【%s】的运用上相对优于其他部分。"
            ) % (dim_min_name, dim_max_name)
            summary_rows.append(Paragraph(p2, self.styles["body"]))
            summary_rows.append(Spacer(1, 0.4 * cm))
        # 第一行左右结构：维度均分表 | 柱状图
        if dim_means:
            max_s = max(s for _, s in dim_means) if dim_means else 0
            min_s = min(s for _, s in dim_means) if dim_means else 0
            data = [["维度", "全员平均分", "备注"]]
            for dim, sc in dim_means:
                note = "最高" if sc == max_s else ("最低" if sc == min_s else "")
                data.append([dim, "%.2f" % sc, note])
            bar_chart_height_cm = 4.8
            n_rows = len(data)  # 1 表头 + 5 数据行
            row_heights = [bar_chart_height_cm / n_rows * cm] * n_rows
            col_width_left_cm = 6.2
            dim_col_width = col_width_left_cm / 3
            dim_table = Table(data, colWidths=[dim_col_width * cm, dim_col_width * cm, dim_col_width * cm], rowHeights=row_heights)
            dim_table.setStyle(TableStyle([
                ("FONTNAME", (0, 0), (-1, -1), self.font_name or "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f0f0")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            bar_cell = Paragraph("（暂无图表）", self.styles["table_cell_center"])
            if summary_chart_png and summary_chart_png.getvalue():
                try:
                    summary_chart_png.seek(0)
                    bar_cell = Image(summary_chart_png, width=7.2 * cm, height=bar_chart_height_cm * cm)
                except Exception:
                    pass
            row1 = Table(
                [[dim_table, bar_cell]],
                colWidths=[col_width_left_cm * cm, (_FRAME_WIDTH_CM - col_width_left_cm) * cm],
            )
            row1.setStyle(TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ALIGN", (0, 0), (0, 0), "LEFT"),
                ("ALIGN", (1, 0), (1, 0), "CENTER"),
                ("LEFTPADDING", (0, 0), (0, -1), 0),
                ("RIGHTPADDING", (0, 0), (0, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                ("LEFTPADDING", (1, 0), (1, 0), 8),
            ]))
            summary_rows.append(row1)
            summary_rows.append(Spacer(1, 0.4 * cm))
        # 希望深入学习的技能模块：副标题与“管理者们……”放入同一左栏；右侧仅饼图
        p3_para = Paragraph("（本期调研未采集「希望重点学习的模块」相关选项数据。）", self.styles["body"])
        if summary_votes and len(summary_votes) > 0:
            main_mod, main_cnt = summary_votes[0]
            others = ["【%s】（%d 票）" % (m, c) for m, c in summary_votes[1:]]
            p3_text = "管理者们主要希望在【%s】（%d 票）进行深入的学习和研讨。" % (main_mod, main_cnt)
            if others:
                p3_text += "其他选项：" + " ".join(others) + "。"
            p3_para = Paragraph(p3_text, self.styles["body"])
        left_block = Table([
            [Paragraph("希望深入学习的技能模块", self.styles["heading2"])],
            [p3_para],
        ], colWidths=[8.2 * cm])
        left_block.setStyle(TableStyle([
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 1), (0, 1), 4),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ]))
        pie_learning_cell = Paragraph("（暂无数据）", self.styles["table_cell_center"])
        if pie_learning_png and getattr(pie_learning_png, "getvalue", None) and pie_learning_png.getvalue():
            try:
                pie_learning_png.seek(0)
                pie_learning_cell = Image(pie_learning_png, width=5.5 * cm, height=5.2 * cm)
            except Exception:
                pass
        row2 = Table(
            [[left_block, pie_learning_cell]],
            colWidths=[8.2 * cm, (_FRAME_WIDTH_CM - 8.2) * cm],
        )
        row2.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ALIGN", (0, 0), (0, 0), "LEFT"),
            ("ALIGN", (1, 0), (1, 0), "CENTER"),
            ("LEFTPADDING", (0, 0), (0, -1), 0),
            ("RIGHTPADDING", (0, 0), (0, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ]))
        summary_rows.append(row2)
        summary_rows.append(Spacer(1, 0.4 * cm))
        if tenure_votes and len(tenure_votes) > 0:
            main_t, cnt_t = tenure_votes[0]
            others_t = ["【%s】（%d 人）" % (o, c) for o, c in tenure_votes[1:3]]
            p_tenure = "管理年限方面，多数伙伴为【%s】（%d 人）。" % (main_t, cnt_t)
            if others_t:
                p_tenure += "其次：" + "、".join(others_t) + "。"
            summary_rows.append(Spacer(1, 0.4 * cm))
            summary_rows.append(Paragraph(p_tenure, self.styles["body"]))
        if team_size_votes and len(team_size_votes) > 0:
            main_s, cnt_s = team_size_votes[0]
            others_s = ["【%s】（%d 人）" % (o, c) for o, c in team_size_votes[1:3]]
            p_team = "团队规模方面，多数伙伴为【%s】（%d 人）。" % (main_s, cnt_s)
            if others_s:
                p_team += "其次：" + "、".join(others_s) + "。"
            summary_rows.append(Paragraph(p_team, self.styles["body"]))
        pie_tenure_cell = None
        pie_team_cell = None
        if pie_tenure_png and getattr(pie_tenure_png, "getvalue", None) and pie_tenure_png.getvalue():
            try:
                pie_tenure_png.seek(0)
                pie_tenure_cell = Image(pie_tenure_png, width=5.5 * cm, height=5.2 * cm)
            except Exception:
                pie_tenure_cell = Paragraph("管理年限分布（图略）", self.styles["table_cell_center"])
        else:
            pie_tenure_cell = Paragraph("管理年限分布（暂无数据）", self.styles["table_cell_center"])
        if pie_team_png and getattr(pie_team_png, "getvalue", None) and pie_team_png.getvalue():
            try:
                pie_team_png.seek(0)
                pie_team_cell = Image(pie_team_png, width=5.5 * cm, height=5.2 * cm)
            except Exception:
                pie_team_cell = Paragraph("团队规模分布（图略）", self.styles["table_cell_center"])
        else:
            pie_team_cell = Paragraph("团队规模分布（暂无数据）", self.styles["table_cell_center"])
        summary_rows.append(Spacer(1, 0.4 * cm))
        pie_two_table = Table(
            [[pie_tenure_cell, pie_team_cell]],
            colWidths=[(_FRAME_WIDTH_CM / 2) * cm, (_FRAME_WIDTH_CM / 2) * cm],
        )
        pie_two_table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("LEFTPADDING", (0, 0), (0, -1), 0),
            ("LEFTPADDING", (1, 0), (1, -1), 0),
        ]))
        summary_rows.append(pie_two_table)
        summary_table = Table([[f] for f in summary_rows], colWidths=[_FRAME_WIDTH_CM * cm])
        summary_table.setStyle(TableStyle([
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ]))
        story.append(summary_table)
        story.append(PageBreak())

        # 第三部分：各维度行为项平均分（第四页；三列表格 + 下方折线图）
        story.append(Paragraph("三、各维度行为项平均分", self.styles["heading1"]))
        data = [["模块", "行为项", "平均分"]]
        if behavior_avgs and SURVEY_QUESTIONS:
            for mod, be, _ in SURVEY_QUESTIONS:
                sc = (behavior_avgs.get(mod) or {}).get(be)
                data.append([mod, be, "%.2f" % round(sc, 2)] if sc is not None else [mod, be, "-"])
        if len(data) > 1:
            t = Table(data, colWidths=[3.5 * cm, 5 * cm, 2.5 * cm])
            t.setStyle(TableStyle([
                ("FONTNAME", (0, 0), (-1, -1), self.font_name or "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8f4fc")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ]))
            story.append(t)
            story.append(Spacer(1, 0.5 * cm))
        if behavior_chart_png and getattr(behavior_chart_png, "getvalue", None) and behavior_chart_png.getvalue():
            try:
                behavior_chart_png.seek(0)
                img = Image(behavior_chart_png, width=14 * cm, height=7 * cm)
                story.append(img)
            except Exception:
                pass
        story.append(PageBreak())

        # 附录：学员自陈结果细则（一页放多人，姓名与雷达图、折线图 KeepTogether 不分离）
        if person_details:
            story.append(Paragraph("附录：学员自陈结果细则", self.styles["heading1"]))
            story.append(Paragraph(
                '这一部分，我们主要关注每个学员自身数据横向对比，看"跟自己比"的高分项和低分项。',
                self.styles["note_red"],
            ))
            story.append(Spacer(1, 0.3 * cm))
            for name, radar_io, line_io in person_details:
                row_cells = []
                if radar_io and getattr(radar_io, "getvalue", None) and radar_io.getvalue():
                    try:
                        radar_io.seek(0)
                        row_cells.append(Image(radar_io, width=5.5 * cm, height=5 * cm))
                    except Exception:
                        row_cells.append(Paragraph("（雷达图）", self.styles["body"]))
                else:
                    row_cells.append(Paragraph("（雷达图）", self.styles["body"]))
                if line_io and getattr(line_io, "getvalue", None) and line_io.getvalue():
                    try:
                        line_io.seek(0)
                        row_cells.append(Image(line_io, width=10.5 * cm, height=6 * cm))
                    except Exception:
                        row_cells.append(Paragraph("（折线图）", self.styles["body"]))
                else:
                    row_cells.append(Paragraph("（折线图）", self.styles["body"]))
                name_para = Paragraph(name, self.styles["heading2"])
                if len(row_cells) == 2:
                    t = Table([row_cells], colWidths=[5.5 * cm, 10.5 * cm])
                    t.setStyle(TableStyle([
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("ALIGN", (0, 0), (0, 0), "CENTER"),
                        ("ALIGN", (1, 0), (1, 0), "CENTER"),
                    ]))
                    story.append(KeepTogether([name_para, Spacer(1, 0.2 * cm), t, Spacer(1, 0.4 * cm)]))
                else:
                    story.append(KeepTogether([name_para, Spacer(1, 0.4 * cm)]))
        story.append(PageBreak())

        # 第四部分：异常提醒
        story.append(Paragraph("四、异常提醒", self.styles["heading1"]))
        story.append(Paragraph("单选题若全部为同一分值，则视为异常，建议管理者关注。", self.styles["body"]))
        if anomaly_rows:
            cell_style = self.styles["table_cell"]
            cell_center = self.styles["table_cell_center"]
            data = [
                [
                    Paragraph(escape("姓名"), cell_center),
                    Paragraph(escape("部门"), cell_center),
                    Paragraph(escape("统一分值"), cell_center),
                    Paragraph(escape("说明"), cell_center),
                ]
            ]
            for row in anomaly_rows:
                name = row[0] if len(row) > 0 else ""
                dept = row[1] if len(row) > 1 and row[1] is not None else "-"
                score = row[2] if len(row) > 2 else ""
                note = row[3] if len(row) > 3 else ""
                score_str = "%.2f" % score if isinstance(score, (int, float)) else str(score)
                data.append([
                    Paragraph(escape(str(name)), cell_style),
                    Paragraph(escape(str(dept)), cell_style),
                    Paragraph(escape(score_str), cell_center),
                    Paragraph(escape(str(note)), cell_style),
                ])
            t = Table(data, colWidths=[3 * cm, 3 * cm, 2.5 * cm, 6 * cm])
            t.setStyle(TableStyle([
                ("FONTNAME", (0, 0), (-1, -1), self.font_name or "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#fff0f0")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]))
            story.append(t)
        else:
            story.append(Paragraph("当前无异常：未发现「全部题目同一分值」的填答。", self.styles["body"]))

        self._background_path = self._resolve_background_path()

        def _on_page(canvas, doc):
            self._canvas_background(canvas, doc)
            self._canvas_header(canvas, doc)
        self.doc.build(story, onFirstPage=_on_page, onLaterPages=_on_page)
        self.buffer.seek(0)
        return self.buffer
