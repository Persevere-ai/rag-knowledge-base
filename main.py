"""
RAG 企业知识库 —— 项目入口

用法（在项目根目录执行）：
    python main.py            # 交互式菜单，选择对话或入库
    python main.py ingest     # 批量入库 datas/md 下的文档
    python main.py rag2       # 直接启动自适应 RAG 对话
    python main.py rag1       # 直接启动自我纠正 RAG 对话
"""

import sys


def _check_config():
    """启动前检查必要的配置是否已填写"""
    from utils.env_utils import OPENAI_API_KEY
    if not OPENAI_API_KEY:
        print('=' * 60)
        print('[提示] 尚未配置 OPENAI_API_KEY。')
        print('请先把项目根目录的 .env.example 复制为 .env，')
        print('并在其中填入你的 API Key 后重试。')
        print('=' * 60)
        return False
    return True


def run_ingest():
    """批量入库 datas/md 下的 Markdown 文档"""
    if not _check_config():
        return
    from documents.write_milvus import ingest
    ingest()


def run_rag1():
    """自我纠正 RAG（Corrective RAG）"""
    from graph.graph1 import run
    run()


def run_rag2():
    """自适应 RAG（Adaptive RAG）"""
    from graph2.graph_2 import run
    run()


def main():
    # 支持命令行参数快速启动
    if len(sys.argv) > 1:
        cmd = sys.argv[1].lower()
        if cmd == 'ingest':
            run_ingest()
            return
        elif cmd == 'rag1':
            if _check_config():
                run_rag1()
            return
        elif cmd == 'rag2':
            if _check_config():
                run_rag2()
            return

    # 交互式菜单
    print('=' * 60)
    print('RAG 企业知识库系统')
    print('=' * 60)
    print('  1. 自适应 RAG 对话（推荐，含路由/改写/幻觉检测）')
    print('  2. 自我纠正 RAG 对话（含查询改写重试）')
    print('  3. 批量入库 datas/md 下的文档')
    print('  q. 退出')
    print('=' * 60)

    choice = input('请选择操作: ').strip()
    if choice == '1':
        if _check_config():
            run_rag2()
    elif choice == '2':
        if _check_config():
            run_rag1()
    elif choice == '3':
        run_ingest()
    elif choice.lower() == 'q':
        print('再见！')
    else:
        print('无效输入，请重新运行。')


if __name__ == '__main__':
    main()
