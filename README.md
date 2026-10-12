# RepoMind

RepoMind 是面向 GitHub 代码仓库的只读理解与分析系统。当前完成仓库导入、Python AST 结构索引，以及 **Symbol + FTS5/BM25 + Dense + RRF** 混合检索。尚未提供 LLM 问答或 MCP 服务；完整路线见 [schedule.md](schedule.md)。

## 环境与安装

需要 Python 3.12+、Git 和 [uv](https://docs.astral.sh/uv/)。在项目根目录运行：

```bash
uv sync
uv run pytest
```

运行时仅使用 Python 标准库和系统 Git，开发测试使用 pytest。工作区默认是当前目录下的 `.repomind/`，其中有 `index.sqlite3` 与保留的裸 Git 快照。可用 `--workspace PATH` 指定其他目录。不要把工作区放进待扫描的本地仓库内。

## 第一阶段使用方法

导入本地 **Git 仓库已提交的 `HEAD`**：

```bash
uv run repomind import /path/to/local/repository
```

导入公开 GitHub 仓库的默认分支：

```bash
uv run repomind import https://github.com/OWNER/REPO
```

命令会打印 `repository` ID、`snapshot` ID、完整 commit SHA、Python 文件数、符号数和解析错误数。本地未提交或未跟踪文件不会进入快照，CLI 会给出提示。非 Git 本地目录暂不支持：这样可保证所有源码位置都对应一个可重读的固定版本。

列出仓库、检索并从保留快照读取源码：

```bash
uv run repomind list
uv run repomind search REPOSITORY_ID Worker --mode symbol
uv run repomind search REPOSITORY_ID "how is connection closed?" --mode hybrid --limit 5
uv run repomind search REPOSITORY_ID "handleRead" --mode bm25 --snapshot FULL_COMMIT_SHA
uv run repomind search REPOSITORY_ID --file pkg/mod.py
uv run repomind show SYMBOL_ID
```

`search` 默认使用符号与 BM25 融合；配置 embedding 后可加入 Dense。原来的 `search REPO NAME` 仍能搜索名称；需要纯结构搜索时加 `--mode symbol`。不带查询的 `search REPO --file PATH` 保留第一阶段的文件符号列表。查询输出排名、路径、行号、得分、通道、commit 和 chunk ID。`show` 会核对 Git blob 的内容哈希和行范围，再显示带行号的原始源码。检索命中还会校验 chunk 哈希。重复导入同一 commit 不会新增索引记录；旧快照仍可按完整 commit SHA 或 snapshot ID 查询。

### Dense 向量索引

可选配置 `DASHSCOPE_API_KEY`；有区域或工作空间端点时，另设 `DASHSCOPE_BASE_URL` 为兼容 API 的 `/v1` 根路径。密钥由环境变量提供，不要提交 `.env`。先显式构建向量，再搜索：

```bash
uv run repomind index-embeddings REPOSITORY_ID --model text-embedding-v4 --dimension 1024
uv run repomind search REPOSITORY_ID "close a connection" --mode dense
uv run repomind search REPOSITORY_ID "close a connection" --mode hybrid
```

也可用 `--snapshot FULL_COMMIT_SHA` 指定旧版本。未设置 API Key 时，Symbol、BM25 和双路 RRF 仍可用；`dense` 或 `index-embeddings` 会给出明确错误。构建向量会把选中仓库的符号名和最多 3000 字符源码发送给配置的云服务。无 Key 时不会生成伪向量。离线测试使用可替换的 Fake Provider。

每个 chunk 的向量按 `(chunk_id, model_id)` 存在同一 SQLite 数据库，包含输入哈希与维度；相同输入重复构建不再请求 embedding。向量检索按仓库、快照、模型过滤，使用标准库做精确余弦扫描；当前适合单人开发与中小型仓库。若需改变 embedding 输入格式，应更新模型/输入版本标识并重建向量。

## 当前结构

```text
ingestion  GitHub URL 校验、裸 Git 克隆、固定 commit 与文件筛选
parsing    Python ast 定义、签名、docstring、导入与源码跨度
models     稳定 ID 和可序列化的数据记录
storage    SQLite 仓库、快照、文件、符号、chunk、导入关系、FTS5 与向量
retrieval  标识符拆词、符号匹配、BM25、embedding 适配、余弦扫描和 RRF
service    导入/解析/发布与按快照读取和校验源码
cli        import、list、search、show、index-embeddings
```

索引基于 Git blob，不 checkout 或执行目标仓库代码。FTS5 缓存经过拆词的源码文本，检索结果仍从保留的 Git 快照重读并校验文件及 chunk 哈希。第一阶段的旧 SQLite 工作区会在首次 BM25 检索或向量索引时回填 FTS。FTS5 使用 `unicode61`，同时索引完整标识符和 `snake_case`、`CamelCase`、缩写拆分词；用户查询先安全拆词，再以参数化 `MATCH` 执行。SQLite `bm25()` **越小越相关**，融合时只使用每路排名，默认 `k=60`。

## 边界与限制

- 只接受公开 GitHub `https://github.com/OWNER/REPO` URL 和带已提交 `HEAD` 的本地 Git 仓库；不支持认证、指定分支、非 Git 目录和未提交内容。
- 第一阶段只扫描 `.py`；跳过常见依赖/缓存目录、符号链接与过大文件。最多索引 2,000 个 Python 文件、每文件 1 MB、总 Python 源码 20 MB。克隆有 120 秒超时并在完成后检查本地副本大小；**目前无法对网络传输字节数提供硬上限**，因此只用于明确选择的公开仓库。
- AST 只提取语法结构与导入语句，不推断动态调用目标。长函数尚未细分，当前每个符号对应一个结构 chunk；模块顶层非定义文本不会作为独立 chunk 检索。
- Dense 需要显式建立向量；真实 DashScope API 尚未在自动化测试中验证。纯 Python 精确扫描会随向量数增长。FTS5 的 IDF 统计来自整个数据库，跨版本加入数据可能改变 BM25 原始分数，正式实验应冻结数据库。
- 当前没有 LangGraph 问答、引用语义验证、关系扩展或 MCP。它们属于后续阶段。
- 当前导入在解析完成后以 SQLite 事务发布结构索引；快照文件与 SQLite 不共享事务。若索引写入失败，可能留下未引用的裸 Git 快照，后续可重新导入。

## 测试

`uv run pytest` 使用本地临时 Git 仓库与 Fake Embedding Provider，不依赖 GitHub 网络或云 API。测试覆盖递归扫描、AST、SQLite、重复导入、版本隔离、符号排序、FTS5 标识符、特殊查询、向量持久化和维度、RRF、降级及源码行号。GitHub 在线导入与真实 DashScope 调用应作为独立联通性检查。
