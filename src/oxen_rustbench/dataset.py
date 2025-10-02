from datasets import load_dataset, Dataset

def create_dataset(mode: str, system_prompt: str) -> Dataset:
    """Create dataset for training"""
    assert mode in ["train", "test", "eval"], f"Invalid mode: {mode}"
    path = f"src/oxen_rustbench/data/cargo_test_passed_{mode}.parquet"
    data = load_dataset("parquet", data_files={"train": path})["train"]
    data = data.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': x['rust_prompt']}
        ],
        "test_list": x.get('rust_test_list', [])
    })
    return data