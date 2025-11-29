import os
import subprocess
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from huggingface_hub import snapshot_download
from pyserini.search.lucene import LuceneSearcher
from pyserini.index.lucene import LuceneIndexReader
from peft import PeftModel

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-21-openjdk-amd64"
os.environ["PATH"] += os.pathsep + os.path.join(os.environ["JAVA_HOME"], "bin")

print(subprocess.check_output(["java","-version"], stderr=subprocess.STDOUT).decode())
print(subprocess.check_output(["javac","-version"]).decode())


class ModelLoader:
    """Handles loading and initialization of all models and indexes."""
    
    def __init__(self, device):
        BASE_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        LORA_DIR = "TinyLlama-1.1B-Chat-v1.0-SRE-LoRA"
        RET_MODEL_NAME = "facebook/contriever-msmarco"
        
        self.device = device
        print("Using device:", self.device)
        
        # Load Wiki18 (BM25) 
        print("Loading BM25 index...")
        local_dir = snapshot_download(
            repo_id="MinaGabriel/wiki18-bm25-index",
            repo_type="dataset"
        )
        self.index_dir = os.path.join(local_dir, "bm25_index")
        self.searcher = LuceneSearcher(self.index_dir)
        self.reader = LuceneIndexReader(self.index_dir)
        
        # Load Classifier Model (LoRA)
        print("Loading classifier model...")
        self.cls_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        if self.cls_tokenizer.pad_token is None:
            self.cls_tokenizer.pad_token = self.cls_tokenizer.eos_token
        self.cls_tokenizer.padding_side = "right"

        self.label2id = {"No": 0, "Yes": 1}
        self.id2label = {0: "No", 1: "Yes"}
        base_model = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL_NAME,
            num_labels=2,
            label2id=self.label2id,
            id2label=self.id2label,
        )
        self.cls_model = PeftModel.from_pretrained(base_model, LORA_DIR).to(self.device)
        self.cls_model.eval()

        # Load Dense Retrieval Model (Contriever)
        print("Loading dense retrieval model...")
        self.ret_tokenizer = AutoTokenizer.from_pretrained(RET_MODEL_NAME)
        self.dense_model = AutoModel.from_pretrained(RET_MODEL_NAME).to(self.device)
        self.dense_model.eval()
        
        print("All models loaded successfully!")