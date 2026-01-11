import pandas as pd
import numpy as np
import os
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# 設定路徑
INPUT_CSV = "data/raw/news/news_for_finbert.csv" 
OUTPUT_CSV = "data/processed/news_emb_hourly.csv"
# 🚀 修改：改用金融專用 FinBERT，維度為 768
MODEL_NAME = "ProsusAI/finbert" 

def generate_embeddings():
    if not os.path.exists(INPUT_CSV):
        print(f"❌ 錯誤: 找不到檔案 {INPUT_CSV}")
        return

    print(f"1. 載入 FinBERT 模型: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # 🚀 強制使用 safetensors 避開 torch.load 漏洞報錯
    model = AutoModel.from_pretrained(MODEL_NAME, use_safetensors=True).cuda()
    model.eval()

    print("2. 讀取並清洗新聞資料...")
    df = pd.read_csv(INPUT_CSV)
    df['datetime'] = pd.to_datetime(df['published_at']).dt.floor('H')
    # FinBERT 對標題通常最敏感
    df['text'] = df['title'].fillna("") 
    
    grouped = df.groupby('datetime')
    hourly_vectors = {}
    
    print(f"3. 計算 FinBERT 向量 - 處理 {len(grouped)} 個原始新聞小時...")
    for dt, group in tqdm(grouped):
        texts = group['text'].astype(str).tolist()
        # 🚀 BoEC 思想：對該小時內所有新聞進行 Embedding 並取平均質心
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to("cuda")
        
        with torch.no_grad():
            outputs = model(**inputs)
            # 取 CLS token 作為特徵 (維度 768)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            mean_emb = np.mean(embeddings, axis=0)
            hourly_vectors[dt] = mean_emb

    print("4. 執行時間軸對齊與情緒延續...")
    emb_df = pd.DataFrame.from_dict(hourly_vectors, orient='index')
    emb_df.index = pd.to_datetime(emb_df.index)
    
    full_range = pd.date_range(start=emb_df.index.min(), end=emb_df.index.max(), freq='H')
    emb_df = emb_df.reindex(full_range)
    
    # 模擬新聞情緒在 8 小時內持續發酵
    emb_df = emb_df.ffill(limit=8)
    emb_df = emb_df.fillna(0)
    
    emb_df.index.name = 'datetime'
    emb_df.columns = [f'emb_{i}' for i in range(emb_df.shape[1])]
    
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    emb_df.to_csv(OUTPUT_CSV)
    print(f"✅ 最終樣本矩陣形狀: {emb_df.shape} (維度已轉為 768)")

if __name__ == "__main__":
    generate_embeddings()