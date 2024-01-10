import argparse 
import os 

from huggingface_hub import snapshot_download, hf_hub_url 
""" 
    python huggingface_download.py --model facebook/opt-125m
"""
if __name__ == "__main__": 
    parser = argparse.ArgumentParser() 
    parser.add_argument("--model", type=str, required=True) 
    args = parser.parse_args() 
    print(f"Downloading {args.model}...") 
    snapshot_download(repo_id=args.model, ignore_patterns=["*.h5", "*.ot", "*.msgpack"])