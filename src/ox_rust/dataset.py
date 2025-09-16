from datasets import load_dataset, Dataset

def create_dataset(path: str, system_prompt: str) -> Dataset:
    """Create dataset for training"""
    data = load_dataset("parquet", data_files={"train": path})["train"]
    data = data.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': x['rust_prompt']}
        ],
        "test_list": x.get('rust_test_list', [])
    })
    return data