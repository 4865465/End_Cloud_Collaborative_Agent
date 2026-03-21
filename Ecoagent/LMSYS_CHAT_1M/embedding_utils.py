"""
Embedding工具：支持两种方式生成文本嵌入
1. 远程API方式：使用 OpenAI 兼容接口调用远程 Embedding API
2. 本地模型方式：使用 SentenceTransformer 在本地加载模型
通过配置 EMBEDDING_USE_LOCAL 控制使用哪种方式
"""
import time
from typing import List, Union, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

from config import (
    EMBEDDING_MODEL,
    EMBEDDING_API_KEY,
    EMBEDDING_API_BASE,
    EMBEDDING_USE_LOCAL,
    EMBEDDING_GPU_ID,
)


class QwenEmbedding:
    """Embedding 模型封装类（支持API和本地模型两种方式）"""

    def __init__(self, model: Optional[str] = None, use_local: Optional[bool] = None):
        """
        初始化嵌入模型客户端

        Args:
            model: 模型名称，默认使用配置中的 EMBEDDING_MODEL
            use_local: 是否使用本地模型，默认使用配置中的 EMBEDDING_USE_LOCAL
        """
        self.model_name = model or EMBEDDING_MODEL
        self.use_local = use_local if use_local is not None else EMBEDDING_USE_LOCAL
        self._embedding_dim: Optional[int] = None
        
        if self.use_local:
            # 本地模型模式
            model_to_load = self.model_name
            gpu_id_to_use = EMBEDDING_GPU_ID
            
            # 确定设备
            if gpu_id_to_use is not None and gpu_id_to_use >= 0:
                device = f"cuda:{gpu_id_to_use}"
                print(f"正在加载本地嵌入模型: {model_to_load} (GPU {gpu_id_to_use})")
            else:
                device = "cpu"
                print(f"正在加载本地嵌入模型: {model_to_load} (CPU)")
            
            self.model = SentenceTransformer(model_to_load, device=device)
            # 获取模型维度
            self._embedding_dim = 4096
            print(f"嵌入模型加载完成，维度: {self._embedding_dim}, 设备: {device}")
        else:
            # API模式
            print(f"使用远程API嵌入模型: {self.model_name}")
            self.client = OpenAI(
                api_key=EMBEDDING_API_KEY,
                base_url=EMBEDDING_API_BASE,
            )
            print(f"嵌入模型API客户端初始化完成")
    
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        max_retries: int = 9,
    ) -> np.ndarray:
        """
        将文本编码为嵌入向量
        
        Args:
            texts: 单个文本或文本列表
            batch_size: 批处理大小
            show_progress_bar: 是否显示进度条
            max_retries: 最大重试次数（API模式使用，本地模式通常不需要重试）
        
        Returns:
            嵌入向量数组，形状为 (n_texts, embedding_dim)
        """
        # 确保texts是列表
        if isinstance(texts, str):
            texts = [texts]
        
        if len(texts) == 0:
            return np.array([])
            
        target_dim = 4096
            
        if self.use_local:
            # 本地模型模式
            try:
                embeddings = self.model.encode(
                    texts,
                    batch_size=batch_size,
                    show_progress_bar=show_progress_bar,
                    convert_to_numpy=True,
                    normalize_embeddings=False  # 根据模型需要可以设置为True
                )
                embeddings = np.array(embeddings, dtype=float)
                
                # Check and pad if needed
                if embeddings.shape[1] < target_dim:
                    padding = np.zeros((embeddings.shape[0], target_dim - embeddings.shape[1]), dtype=embeddings.dtype)
                    embeddings = np.hstack([embeddings, padding])
                    
                return embeddings
            except Exception as e:
                print(f"生成嵌入向量时出错: {e}")
                # 如果失败，返回零向量（使用已知维度或默认1024）
                embedding_dim = self._embedding_dim or 4096
                return np.zeros((len(texts), embedding_dim), dtype=float)
        else:
            # API模式
            all_embeddings: List[np.ndarray] = []
            # 分批处理
            total_batches = (len(texts) + batch_size - 1) // batch_size
            
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(texts))
                batch_texts = texts[start_idx:end_idx]
                
                if show_progress_bar:
                    print(f"处理批次 {batch_idx + 1}/{total_batches} ({len(batch_texts)} 个文本)...")
                
                # 重试机制
                base_delay = 15
                for attempt in range(max_retries):
                    try:
                        # 调用远程 Embedding API
                        resp = self.client.embeddings.create(
                            model=self.model_name,
                            input=batch_texts,
                            encoding_format="float",
                        )
                        batch_embeddings = np.array([d.embedding for d in resp.data], dtype=float)

                        # 如果这是第一批，记录维度（用于错误处理）
                        if self._embedding_dim is None and len(batch_embeddings) > 0:
                            self._embedding_dim = 4096 # Force 4096

                        all_embeddings.extend(batch_embeddings)
                        break  # 成功，跳出重试循环
                        
                    except Exception as e:
                        if attempt < max_retries - 1:
                            wait_time = base_delay * (2 ** attempt)  # 指数退避：15s, 30s, 60s...
                            print(f"生成嵌入向量时出错，等待 {wait_time:.1f}s 后重试（第 {attempt + 1}/{max_retries} 次）: {e}")
                            time.sleep(wait_time)
                            continue
                        else:
                            print(f"生成嵌入向量时出错: {e}")
                            # 如果失败，返回零向量（使用已知维度或默认1024）
                            embedding_dim = self._embedding_dim or 4096  # Qwen3-Embedding-0.6B 通常是1024维
                            batch_embeddings = np.zeros((len(batch_texts), embedding_dim), dtype=float)
                            all_embeddings.extend(batch_embeddings)
                            break
            
            # Combine all
            result = np.array(all_embeddings)
            
            # Pad to 4096 if needed
            target_dim = 4096
            if result.shape[1] < target_dim:
                padding = np.zeros((result.shape[0], target_dim - result.shape[1]), dtype=result.dtype)
                result = np.hstack([result, padding])
            
            return result
    
    def encode_single(self, text: str) -> np.ndarray:
        """
        编码单个文本
        
        Args:
            text: 输入文本
        
        Returns:
            嵌入向量，形状为 (embedding_dim,)
        """
        embeddings = self.encode([text])
        return embeddings[0] if len(embeddings) > 0 else np.array([])


