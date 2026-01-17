# 🚀 快速演示指南（5分钟上手）

## 第一步：启动服务器

```bash
# 方式1：直接运行
python RAG.py

# 方式2：使用批处理文件（Windows）
快速启动.bat
```

**看到以下输出表示启动成功：**
```
============================================================
企业级 RAG + Agent 服务启动
配置: LLM=DeepSeek, Embedding=本地BGE（离线模式）
接口文档: http://127.0.0.1:8000/docs
健康检查: http://127.0.0.1:8000/health
============================================================
```

---

## 第二步：打开 API 文档界面

在浏览器中访问：**http://127.0.0.1:8000/docs**

---

## 第三步：获取 Token（授权）

1. 在 API 文档中找到 `/auth/dev_login` 接口
2. 点击 "Try it out"
3. 输入：
```json
{
  "user": "demo_user",
  "roles": ["public", "admin"],
  "department": "技术部"
}
```
4. 点击 "Execute"
5. **复制返回的 `access_token`**

---

## 第四步：授权 API 调用

1. 点击页面右上角的 **"Authorize"** 按钮
2. 输入：`Bearer <你的access_token>`（注意 Bearer 后面有空格）
3. 点击 "Authorize"，然后 "Close"

---

## 第五步：导入文档

1. 找到 `/ingest` 接口
2. 点击 "Try it out"
3. 输入：
```json
{
  "paths": ["./demo_docs"],
  "department": "技术部"
}
```
4. 点击 "Execute"
5. 查看结果，应该显示成功导入了 5 个文档

---

## 第六步：测试功能

### 6.1 搜索文档（/search）
```json
{
  "query": "年假政策",
  "k": 5
}
```

### 6.2 智能问答（/query）
```json
{
  "question": "员工年假有多少天？"
}
```

### 6.3 Agent 对话（/agent_chat）
```json
{
  "message": "搜索一下关于报销流程的文档"
}
```

---

## 🎬 自动化演示（推荐）

如果想快速测试所有功能，运行：

```bash
python demo_script.py
```

这个脚本会自动：
- ✅ 检查服务器状态
- ✅ 获取 Token
- ✅ 导入文档
- ✅ 测试搜索功能
- ✅ 测试问答功能
- ✅ 测试 Agent 对话

---

## 📋 详细文档

- **完整运行指南**：查看 `运行指南.md`
- **演示检查清单**：查看 `演示检查清单.md`

---

## ⚠️ 常见问题

**Q: 启动失败？**  
A: 检查 `.env` 文件中的 `OPENAI_API_KEY` 是否配置正确

**Q: 导入文档失败？**  
A: 确认 `demo_docs` 文件夹中有 5 个 txt 文件

**Q: Token 过期？**  
A: 重新调用 `/auth/dev_login` 获取新 token

---

**准备好了吗？开始演示吧！** 🎉
