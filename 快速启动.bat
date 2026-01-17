@echo off
chcp 65001 >nul
echo ============================================================
echo 企业级 RAG 系统 - 快速启动脚本
echo ============================================================
echo.

cd /d %~dp0

echo [1/3] 检查 Python 环境...
python --version
if errorlevel 1 (
    echo 错误: 未找到 Python，请先安装 Python
    pause
    exit /b 1
)

echo.
echo [2/3] 检查依赖...
python -c "import fastapi, uvicorn" 2>nul
if errorlevel 1 (
    echo 警告: 部分依赖可能未安装
    echo 请运行: pip install -r requirements_v2.txt
    echo.
)

echo.
echo [3/3] 启动服务器...
echo.
echo ============================================================
echo 服务将在以下地址启动:
echo   - API 文档: http://127.0.0.1:8000/docs
echo   - 健康检查: http://127.0.0.1:8000/health
echo ============================================================
echo.
echo 按 Ctrl+C 停止服务器
echo.

python RAG.py

pause
