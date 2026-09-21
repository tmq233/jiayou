# Git 速查表（我的练习笔记）

## 日常四步：改 → 存 → 传

```bash
git add .                      # 1. 暂存所有改动
git commit -m "写清楚改了什么"   # 2. 存档到本地
git push                       # 3. 上传到 GitHub
```

## 常用查询

```bash
git status          # 看当前状态
git log --oneline   # 看历史记录
git diff            # 看具体改了什么
git remote -v       # 看远程仓库地址
```

## 撤销

```bash
git checkout -- 文件名   # 撤销未提交的改动
```

## 提交信息规范

```
feat:     新增功能
fix:      修复 bug
docs:     改文档
refactor: 重构代码
```
