import multiprocessing
import os
from multiprocessing import Queue

from documents.markdown_parser import MarkdownParser
from documents.milvus_db import MilvusVectorSave
from utils.log_utils import log


# 采用分布式，多进程的方式把海量数据写入Milvus数据库

def file_parser_process(dir_path: str, output_queue: Queue, batch_size: int = 20):
    """进程1：解析目录下所有md文件并分批放入队列"""
    log.info(f"解析进程开始扫描目录: {dir_path}")

    # 获取目录下所有.md文件
    md_files = [
        os.path.join(dir_path, f)
        for f in os.listdir(dir_path)
        if f.endswith('.md')
    ]

    if not md_files:
        log.warning("警告：未找到任何.md文件")
        output_queue.put(None)  # 发送终止信号
        return

    parser = MarkdownParser()
    doc_batch = []
    for file_path in md_files:
        try:
            docs = parser.parse_markdown_to_documents(file_path)
            if docs:
                doc_batch.extend(docs)

            # 达到批次大小时发送 到队列中
            if len(doc_batch) >= batch_size:
                output_queue.put(doc_batch.copy())
                doc_batch.clear()  # 清空当前缓冲区的所有批次数据
        except Exception as e:
            log.error(f"解析失败 {file_path}: {str(e)}")
            log.exception(e)

    # 发送剩余文档
    if doc_batch:
        output_queue.put(doc_batch)

    # 发送终止信号
    output_queue.put(None)
    log.info(f"解析完成，共处理{len(md_files)}个文件")


def milvus_writer_process(input_queue: Queue):
    """进程2：从队列读取并写入Milvus"""
    log.info("Milvus写入进程启动...")


    mv = MilvusVectorSave()
    mv.create_connection()
    total_count = 0
    while True:
        try:
            datas = input_queue.get()  # 阻塞的函数
            if datas is None:  # 收到了终止的信号
                break

            if isinstance(datas, list):
                mv.add_documents(datas)
                total_count += len(datas)
                log.info(f"累计已写入: {total_count} 个文档")
        except Exception as e:
            log.error("写入数据失败：")
            log.exception(e)

    log.info(f"写入进程结束，总计写入 {total_count} 个文档")


def ingest(md_dir: str = None, queue_maxsize: int = 20):
    """把指定目录下的 Markdown 文档批量写入 Milvus 知识库

    参数:
        md_dir: Markdown 文件所在目录，默认取项目下的 datas/md（语料目录）
        queue_maxsize: 进程间队列最大容量（防止内存溢出）
    """
    if md_dir is None:
        # 默认使用项目根目录下的 datas/md（语料目录）
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        md_dir = os.path.join(root_dir, 'datas', 'md')

    # 先创建集合（若已存在会重建，注意会清空旧数据）
    mv = MilvusVectorSave()
    mv.create_collection()

    # 创建进程间通信队列
    docs_queue = Queue(maxsize=queue_maxsize)

    # 启动子进程：一个负责解析，一个负责写入
    parser_proc = multiprocessing.Process(
        target=file_parser_process,
        args=(md_dir, docs_queue)
    )
    writer_proc = multiprocessing.Process(
        target=milvus_writer_process,
        args=(docs_queue,)
    )

    parser_proc.start()
    writer_proc.start()

    # 等待进程结束
    parser_proc.join()
    writer_proc.join()

    log.info("系统提示：所有任务完成")


if __name__ == '__main__':
    # 默认把项目下 datas/md 目录的文档批量入库
    # 也可传入自定义目录：ingest(r'C:/your_docs')
    ingest()