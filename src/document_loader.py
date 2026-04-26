"""
文档加载和处理模块

功能：
1. 从PDF文件中提取文本内容
2. 将长文本切分成适合检索的小块（切片）
"""

import os
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
import config


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    从PDF文件中提取全部文本内容

    参数:
        pdf_path: PDF文件的路径

    返回:
        提取出的全部文本内容
    """
    all_text = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                all_text.append(text)

    full_text = "\n".join(all_text)
    return full_text


def split_text_into_chunks(text: str) -> list[dict]:
    """
    将长文本切分成小块

    参数:
        text: 要切分的长文本

    返回:
        切片列表，每个切片是一个字典，包含:
        - "content": 切片的文本内容
        - "chunk_id": 切片的编号（从0开始）
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", ".", " ", ""],
    )

    raw_chunks = splitter.split_text(text)

    chunks = []
    for i, chunk_text in enumerate(raw_chunks):
        chunks.append({"content": chunk_text, "chunk_id": i})

    return chunks


def load_and_process_pdf(pdf_path: str) -> list[dict]:
    """
    一站式处理：读取PDF → 提取文本 → 切片

    参数:
        pdf_path: PDF文件路径

    返回:
        处理好的切片列表
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"找不到文件: {pdf_path}")

    if not pdf_path.lower().endswith(".pdf"):
        raise ValueError(f"不是PDF文件: {pdf_path}")

    print(f"正在提取文本: {pdf_path}")
    text = extract_text_from_pdf(pdf_path)

    if not text.strip():
        raise ValueError(f"PDF文件中没有提取到文本内容: {pdf_path}")

    print(f"提取到 {len(text)} 个字符")

    print("正在切分文本...")
    chunks = split_text_into_chunks(text)
    print(f"切分为 {len(chunks)} 个文本块")

    return chunks


# ============================================================
# 测试代码
# ============================================================
if __name__ == "__main__":
    test_pdf = os.path.join(config.DATA_DIR, "test.pdf")

    if os.path.exists(test_pdf):
        chunks = load_and_process_pdf(test_pdf)
        for chunk in chunks[:3]:
            print(f"\n--- 切片 {chunk['chunk_id']} ---")
            print(chunk["content"][:200])
            print(f"(共 {len(chunk['content'])} 个字符)")
    else:
        print(f"请先在 {config.DATA_DIR} 目录下放一个名为 test.pdf 的文件")