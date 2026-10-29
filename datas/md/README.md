# 语料目录（datas/md）

把你要入库的 Markdown 文档放到这个目录下，然后运行入库脚本：

```bash
python main.py ingest
```

当前目录保留了几个样例文件（Milvus FAQ 与两篇半导体技术报告），你可以直接删除，换成你自己的文档。

## 注意事项

- 入库脚本会解析本目录下**所有** `.md` 文件。
- 集合（Collection）已存在时，入库会**先清空旧数据再写入**（见 `documents/milvus_db.py` 的 `create_collection`），请确认无误后再执行。
- 文档建议使用带标题层级的 Markdown，解析与语义切割效果更好。
