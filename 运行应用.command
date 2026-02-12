#!/bin/bash
cd "$(dirname "$0")"
echo "正在启动「管理者调研报告」..."
echo "启动后请在浏览器中打开显示的本地地址（一般为 http://localhost:8501）"
echo ""
python3 -m pip install -q streamlit pandas plotly openpyxl reportlab kaleido 2>/dev/null
python3 -m streamlit run app.py --server.port 8501
read -p "按回车键关闭窗口..."
