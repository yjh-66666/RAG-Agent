"""
自动化演示脚本 - 企业级 RAG 系统
用于快速测试所有功能，适合演示视频录制
"""

import json
import time
import requests
from typing import Dict, Any

# 配置
BASE_URL = "http://127.0.0.1:8000"
DEMO_USER = "demo_user"
DEMO_ROLES = ["public", "admin"]
DEMO_DEPARTMENT = "技术部"

# 颜色输出
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(text: str):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text:^60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")

def print_step(step: int, text: str):
    print(f"{Colors.OKBLUE}[步骤 {step}] {text}{Colors.ENDC}")

def print_success(text: str):
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")

def print_error(text: str):
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")

def print_info(text: str):
    print(f"{Colors.OKCYAN}ℹ {text}{Colors.ENDC}")

def print_warning(text: str):
    print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")

def check_server() -> bool:
    """检查服务器是否运行"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print_success(f"服务器运行正常")
            print_info(f"LLM 状态: {data['data']['llm']}")
            print_info(f"Embedding 状态: {data['data']['embedding']}")
            return True
    except Exception as e:
        print_error(f"无法连接到服务器: {e}")
        print_info("请确保已运行: python RAG.py")
        return False
    return False

def get_token() -> str:
    """获取 JWT Token"""
    print_step(1, "获取访问令牌...")
    try:
        payload = {
            "user": DEMO_USER,
            "roles": DEMO_ROLES,
            "department": DEMO_DEPARTMENT,
            "exp_minutes": 1440
        }
        response = requests.post(
            f"{BASE_URL}/auth/dev_login",
            json=payload,
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        if data.get("success"):
            token = data["data"]["access_token"]
            print_success(f"Token 获取成功 (用户: {DEMO_USER})")
            return token
        else:
            print_error("Token 获取失败")
            return None
    except Exception as e:
        print_error(f"获取 Token 失败: {e}")
        return None

def ingest_documents(token: str) -> bool:
    """导入文档"""
    print_step(2, "导入文档到知识库...")
    try:
        headers = {"Authorization": f"Bearer {token}"}
        payload = {
            "paths": ["./demo_docs"],
            "department": DEMO_DEPARTMENT,
            "access_control": ["public"]
        }
        response = requests.post(
            f"{BASE_URL}/ingest",
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        data = response.json()
        if data.get("success"):
            ingest_info = data["data"]["ingest"]
            print_success(f"文档导入成功！")
            print_info(f"  文档数量: {ingest_info.get('documents', 0)}")
            print_info(f"  文本块数量: {ingest_info.get('chunks', 0)}")
            return True
        else:
            print_error("文档导入失败")
            return False
    except Exception as e:
        print_error(f"导入文档失败: {e}")
        return False

def test_search(token: str, query: str, k: int = 3):
    """测试搜索功能"""
    print_step(3, f"搜索: '{query}'")
    try:
        headers = {"Authorization": f"Bearer {token}"}
        payload = {
            "query": query,
            "k": k,
            "freshness_weight": 0.3
        }
        response = requests.post(
            f"{BASE_URL}/search",
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        if data.get("success"):
            results = data["data"]["results"]
            print_success(f"找到 {len(results)} 个相关结果")
            for i, result in enumerate(results[:2], 1):  # 只显示前2个
                print_info(f"\n  结果 {i}:")
                print(f"    相似度: {result['similarity_score']:.4f}")
                content = result['content'][:100] + "..." if len(result['content']) > 100 else result['content']
                print(f"    内容: {content}")
            return True
        else:
            print_error("搜索失败")
            return False
    except Exception as e:
        print_error(f"搜索失败: {e}")
        return False

def test_query(token: str, question: str):
    """测试问答功能"""
    print_step(4, f"提问: '{question}'")
    try:
        headers = {"Authorization": f"Bearer {token}"}
        payload = {"question": question}
        response = requests.post(
            f"{BASE_URL}/query",
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        data = response.json()
        if data.get("success"):
            answer = data["data"]["answer"]
            sources = data["data"]["sources"]
            print_success("AI 回答生成成功")
            print_info(f"\n  答案:")
            # 格式化输出答案（每行最多80字符）
            for line in answer.split('\n'):
                if len(line) > 80:
                    words = line.split()
                    current_line = ""
                    for word in words:
                        if len(current_line + word) > 80:
                            print(f"    {current_line}")
                            current_line = word + " "
                        else:
                            current_line += word + " "
                    if current_line:
                        print(f"    {current_line}")
                else:
                    print(f"    {line}")
            print_info(f"\n  引用来源: {len(sources)} 个文档")
            for source in sources[:3]:  # 只显示前3个
                file_name = source.get("source_file", "").split("/")[-1]
                print(f"    [{source['ref']}] {file_name}")
            return True
        else:
            print_error("问答失败")
            return False
    except Exception as e:
        print_error(f"问答失败: {e}")
        return False

def test_agent_chat(token: str, message: str):
    """测试 Agent 对话"""
    print_step(5, f"Agent 对话: '{message}'")
    try:
        headers = {"Authorization": f"Bearer {token}"}
        payload = {"message": message}
        response = requests.post(
            f"{BASE_URL}/agent_chat",
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        data = response.json()
        if data.get("success"):
            action = data["data"].get("action", "unknown")
            print_success(f"Agent 识别意图: {action}")
            
            if action == "search":
                results = data["data"].get("results", [])
                print_info(f"找到 {len(results)} 个相关文档")
                for i, result in enumerate(results[:2], 1):
                    file_name = result.get("source_file", "").split("/")[-1]
                    print(f"  [{i}] {file_name}")
                    print(f"     预览: {result.get('preview', '')[:80]}...")
            
            elif action == "rag_answer":
                answer = data["data"].get("answer", "")
                sources = data["data"].get("sources", [])
                print_info(f"\n  AI 回答:")
                # 简化输出
                preview = answer[:200] + "..." if len(answer) > 200 else answer
                for line in preview.split('\n')[:3]:
                    print(f"    {line}")
                print_info(f"\n  引用来源: {len(sources)} 个文档")
            
            elif action == "doc_info":
                doc_info = data["data"].get("document", {})
                print_info(f"文档信息:")
                print(f"  文件路径: {doc_info.get('file_path', 'N/A')}")
                print(f"  上传者: {doc_info.get('uploaded_by', 'N/A')}")
                print(f"  版本: {doc_info.get('version', 'N/A')}")
            
            return True
        else:
            print_error(f"Agent 对话失败: {data.get('error', {})}")
            return False
    except Exception as e:
        print_error(f"Agent 对话失败: {e}")
        return False

def main():
    """主演示流程"""
    print_header("企业级 RAG 系统 - 自动化演示")
    
    # 检查服务器
    if not check_server():
        return
    
    time.sleep(1)
    
    # 获取 Token
    token = get_token()
    if not token:
        return
    
    time.sleep(1)
    
    # 导入文档
    if not ingest_documents(token):
        print_warning("文档导入失败，但可以继续演示（如果之前已导入）")
    
    time.sleep(2)
    
    # 测试搜索
    print_header("功能演示 - 搜索")
    test_search(token, "年假政策")
    time.sleep(1)
    test_search(token, "报销流程")
    time.sleep(1)
    test_search(token, "代码评审")
    
    time.sleep(2)
    
    # 测试问答
    print_header("功能演示 - 智能问答")
    test_query(token, "员工年假有多少天？")
    time.sleep(2)
    test_query(token, "新员工如何申请账号？")
    time.sleep(2)
    test_query(token, "报销需要准备哪些材料？")
    
    time.sleep(2)
    
    # 测试 Agent
    print_header("功能演示 - Agent 智能对话")
    test_agent_chat(token, "搜索一下关于代码评审的文档")
    time.sleep(2)
    test_agent_chat(token, "新员工入职需要什么流程？")
    time.sleep(2)
    test_agent_chat(token, "年假可以结转吗？")
    
    print_header("演示完成！")
    print_success("所有功能测试完成")
    print_info(f"API 文档: {BASE_URL}/docs")
    print_info(f"健康检查: {BASE_URL}/health")
    print_info(f"监控指标: {BASE_URL}/metrics")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}演示被用户中断{Colors.ENDC}")
    except Exception as e:
        print_error(f"演示过程中出现错误: {e}")
