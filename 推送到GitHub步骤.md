# 推送到 GitHub 步骤

本地已做好首次提交，按下面步骤即可把代码放到 GitHub。

## 1. 在 GitHub 上新建仓库

1. 打开 [https://github.com/new](https://github.com/new)
2. **Repository name** 填：`survey-report-app`（或你喜欢的英文名）
3. **Description** 可选填：`管理者调研报告 Streamlit 应用`
4. 选择 **Public**
5. **不要**勾选 "Add a README"（本地已有）
6. 点击 **Create repository**

## 2. 在本地添加远程并推送

创建好空仓库后，GitHub 会显示仓库地址，形如：

- HTTPS：`https://github.com/你的用户名/survey-report-app.git`
- SSH：`git@github.com:你的用户名/survey-report-app.git`

在**本项目的终端**里执行（把下面的地址换成你仓库的真实地址）：

```bash
cd "/Users/tal/Desktop/cursor 尝试/新经理报告app/基础信息/Survey_Analysis_Tool"

# 添加远程（二选一，HTTPS 或 SSH）
git remote add origin https://github.com/你的用户名/survey-report-app.git
# 或
# git remote add origin git@github.com:你的用户名/survey-report-app.git

# 推送到 GitHub
git branch -M main
git push -u origin main
```

若用 HTTPS，推送时可能要求输入 GitHub 用户名和密码（密码处填 **Personal Access Token**）。  
若用 SSH，需先在 GitHub 添加 SSH 公钥。

## 3. 完成后

- 在浏览器打开 `https://github.com/你的用户名/survey-report-app` 即可看到代码。
- 若要部署为公开页面，按 **部署说明.md** 用 Streamlit Cloud 连接该仓库即可。
