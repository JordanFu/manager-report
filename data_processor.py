# -*- coding: utf-8 -*-
"""数据清洗、分值转换、维度与全员/个人得分计算"""

import pandas as pd
import numpy as np
from config import SCORE_MAP, QUESTION_MAPPING, CATEGORY_ORDER


def find_question_columns(df: pd.DataFrame):
    """根据关键词匹配列名，返回 [(列名, 维度, 行为), ...]"""
    result = []
    for keyword, category, behavior in QUESTION_MAPPING:
        for col in df.columns:
            if keyword in str(col).strip():
                result.append((col, category, behavior))
                break
    return result


def clean_and_score(df: pd.DataFrame):
    """
    清洗并转换分值。
    返回：(得分 DataFrame 仅题目列, 列->(维度,行为) 映射, 保留的元信息列)
    """
    cols_mapping = find_question_columns(df)
    if not cols_mapping:
        return None, [], []

    # 姓名列：优先「填写人」
    name_col = None
    for c in ["填写人", "姓名", "学员姓名"]:
        if c in df.columns:
            name_col = c
            break
    meta_cols = [c for c in df.columns if c not in [x[0] for x in cols_mapping]]
    if name_col and name_col not in meta_cols:
        meta_cols.append(name_col)

    # 只保留问卷题列
    question_cols = [x[0] for x in cols_mapping]
    df_q = df[question_cols].copy()

    # 文本去空格、统一
    for c in question_cols:
        if df_q[c].dtype == object:
            df_q[c] = df_q[c].astype(str).str.strip()

    # 分值转换
    df_q = df_q.replace(SCORE_MAP)
    # 无法识别的转为 NaN，统一为 float
    for c in question_cols:
        df_q[c] = pd.to_numeric(df_q[c], errors="coerce").astype(float)

    col_to_cat_be = {col: (cat, be) for col, cat, be in cols_mapping}
    return df_q, col_to_cat_be, meta_cols


def compute_dimension_scores(df_scores: pd.DataFrame, col_to_cat_be: dict):
    """
    df_scores: 仅题目列，已为数值。
    返回：每人每维度平均分 DataFrame (index=原行索引, columns=维度名)
    """
    rows = []
    for idx in df_scores.index:
        row = {}
        for col, (cat, _) in col_to_cat_be.items():
            if cat not in row:
                row[cat] = []
            v = df_scores.loc[idx, col]
            if not pd.isna(v):
                row[cat].append(v)
        for cat in row:
            row[cat] = np.mean(row[cat]) if row[cat] else np.nan
        rows.append(row)
    return pd.DataFrame(rows, index=df_scores.index)


def compute_behavior_scores(df_scores: pd.DataFrame, col_to_cat_be: dict):
    """
    返回：每个维度下各行为项的全员平均分。
    { "管理角色认知": {"工作理念": 4.2, ...}, ... }
    """
    out = {}
    for col, (cat, be) in col_to_cat_be.items():
        if cat not in out:
            out[cat] = {}
        out[cat][be] = df_scores[col].mean()
    return out


def get_person_total_and_dims(df_scores: pd.DataFrame, df_dims: pd.DataFrame):
    """每人总分、各维度分（按 CATEGORY_ORDER）。"""
    total = df_scores.mean(axis=1)
    dim_cols = [c for c in CATEGORY_ORDER if c in df_dims.columns]
    dim_scores = df_dims[dim_cols] if dim_cols else pd.DataFrame()
    return total, dim_scores


def get_behavior_avg_by_dimension(df_scores: pd.DataFrame, col_to_cat_be: dict):
    """各维度下各行为项的全员平均分，用于条形图。"""
    return compute_behavior_scores(df_scores, col_to_cat_be)


def get_person_behavior_scores(df_scores: pd.DataFrame, col_to_cat_be: dict, person_idx):
    """某学员在各行为项上的得分（用于折线图）。顺序与维度、行为一致。"""
    labels = []
    values = []
    for col, (cat, be) in col_to_cat_be.items():
        labels.append(f"{cat}-{be}")
        values.append(df_scores.loc[person_idx, col])
    return labels, values


def get_all_behavior_avgs(df_scores: pd.DataFrame, col_to_cat_be: dict):
    """全员各行为项平均分，顺序与 get_person_behavior_scores 一致。"""
    labels = []
    values = []
    for col, (cat, be) in col_to_cat_be.items():
        labels.append(f"{cat}-{be}")
        values.append(df_scores[col].mean())
    return labels, values
