# -*- coding: utf-8 -*-
"""用原始底表生成一份静态 HTML 报告，便于查看输出效果（不依赖 Streamlit 运行）"""

import os
import json
import pandas as pd
import plotly.graph_objects as go
from config import CATEGORY_ORDER, COLORS_BARS
from data_processor import (
    clean_and_score,
    compute_dimension_scores,
    get_behavior_avg_by_dimension,
    get_person_behavior_scores,
    get_all_behavior_avgs,
    get_person_total_and_dims,
)


def fig_to_json_safe(fig):
    """Plotly 图转成可 JSON 序列化的 dict（处理 numpy 等类型）"""
    d = fig.to_dict()
    return json.loads(json.dumps(d, default=lambda x: float(x) if hasattr(x, "item") else str(x)))


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, "原始底表.xlsx")
    if not os.path.isfile(path):
        print("未找到 原始底表.xlsx")
        return
    df = pd.read_excel(path, sheet_name=0)
    df_q, col_to_cat_be, _ = clean_and_score(df)
    if df_q is None:
        print("数据解析失败")
        return
    df_dims = compute_dimension_scores(df_q, col_to_cat_be)
    total, _ = get_person_total_and_dims(df_q, df_dims)
    name_col = "填写人" if "填写人" in df.columns else None
    names = df[name_col].astype(str).tolist() if name_col else [f"学员{i+1}" for i in range(len(df))]

    dim_means = df_dims[CATEGORY_ORDER].mean() if all(c in df_dims.columns for c in CATEGORY_ORDER) else df_dims.mean()
    summary = pd.DataFrame({"维度": dim_means.index, "全员平均分": dim_means.values.round(2)})
    behavior_avgs = get_behavior_avg_by_dimension(df_q, col_to_cat_be)
    row_index = df_q.index[0]
    labels, person_vals = get_person_behavior_scores(df_q, col_to_cat_be, row_index)
    _, all_vals = get_all_behavior_avgs(df_q, col_to_cat_be)
    dim_cols = [c for c in CATEGORY_ORDER if c in df_dims.columns]
    total_person = total.loc[row_index]

    parts = []
    parts.append("""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <title>管理者调研报告 - 示例输出</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    body { font-family: "PingFang SC", "Microsoft YaHei", sans-serif; max-width: 1000px; margin: 0 auto; padding: 24px; }
    h1 { border-bottom: 2px solid #3498DB; padding-bottom: 8px; }
    h2 { color: #2C3E50; margin-top: 32px; }
    h3 { color: #34495E; margin-top: 24px; }
    table { border-collapse: collapse; width: 100%%; margin: 16px 0; }
    th, td { border: 1px solid #ddd; padding: 10px 12px; text-align: left; }
    th { background: #3498DB; color: #fff; }
    .chart { margin: 24px 0; }
    .section { margin-bottom: 40px; }
  </style>
</head>
<body>
  <h1>管理者调研报告自动生成工具 · 示例输出</h1>
  <p>数据来源：原始底表.xlsx，共 %d 条记录，%d 道题。</p>
""" % (len(df), len(col_to_cat_be)))

    # 一、摘要报告
    parts.append('<div class="section"><h2>一、摘要报告：五维度全员平均分</h2>')
    parts.append('<table><tr><th>维度</th><th>全员平均分</th></tr>')
    for _, r in summary.iterrows():
        parts.append("<tr><td>%s</td><td>%.2f</td></tr>" % (r["维度"], r["全员平均分"]))
    parts.append("</table>")
    fig0 = go.Figure(data=[go.Bar(x=summary["全员平均分"].tolist(), y=summary["维度"].tolist(), orientation="h", marker_color=COLORS_BARS[0])])
    fig0.update_layout(title="五维度全员平均分", xaxis_title="平均分", yaxis_title="", height=380, margin=dict(l=120))
    fd0 = fig_to_json_safe(fig0)
    parts.append('<div class="chart" id="chart_summary"></div><script>Plotly.newPlot("chart_summary", %s, %s, {responsive: true});</script></div>' % (json.dumps(fd0["data"]), json.dumps(fd0["layout"])))

    # 二、模块报告
    parts.append('<div class="section"><h2>二、模块报告：各维度行为项得分（全员平均）</h2>')
    for i, cat in enumerate(CATEGORY_ORDER):
        if cat not in behavior_avgs:
            continue
        be_dict = behavior_avgs[cat]
        be_names = list(be_dict.keys())
        be_scores = [round(be_dict[b], 2) for b in be_names]
        color = COLORS_BARS[i % len(COLORS_BARS)]
        fig = go.Figure(data=[go.Bar(x=be_scores, y=be_names, orientation="h", marker_color=color)])
        fig.update_layout(title=cat, xaxis_title="平均分", xaxis=dict(range=[0, 5.5]), height=max(260, len(be_names) * 40), margin=dict(l=140), showlegend=False)
        fd = fig_to_json_safe(fig)
        parts.append('<h3>%s</h3><div class="chart" id="chart_dim_%d"></div><script>Plotly.newPlot("chart_dim_%d", %s, %s, {responsive: true});</script>' % (cat, i, i, json.dumps(fd["data"]), json.dumps(fd["layout"])))
    parts.append("</div>")

    # 三、学员详细报告
    parts.append('<div class="section"><h2>三、学员详细报告（示例：%s）</h2>' % names[0])
    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(x=labels, y=[float(v) for v in person_vals], mode="lines+markers", name="该学员得分", line=dict(color="#E74C3C", width=2), marker=dict(size=8)))
    fig_p.add_trace(go.Scatter(x=labels, y=[float(v) for v in all_vals], mode="lines+markers", name="全员平均分", line=dict(color="#3498DB", width=2, dash="dash"), marker=dict(size=8)))
    fig_p.update_layout(title="%s 各行为项得分 vs 全员平均" % names[0], xaxis_title="行为项", yaxis_title="得分", yaxis=dict(range=[0.5, 5.5]), height=500, legend=dict(orientation="h", y=1.02), margin=dict(b=140), xaxis_tickangle=-45)
    fpd = fig_to_json_safe(fig_p)
    parts.append('<div class="chart" id="chart_person"></div><script>Plotly.newPlot("chart_person", %s, %s, {responsive: true});</script>' % (json.dumps(fpd["data"]), json.dumps(fpd["layout"])))
    parts.append("<p><strong>总分（全部题目平均）：</strong> %.2f</p>" % float(total_person))
    parts.append("<p><strong>各维度平均分：</strong> " + "；".join("%s %.2f" % (c, float(df_dims.loc[row_index, c])) for c in dim_cols) + "</p>")
    parts.append("</div></body></html>")

    out_path = os.path.join(base, "报告示例_原始底表.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print("已生成报告示例：", out_path)


if __name__ == "__main__":
    main()
