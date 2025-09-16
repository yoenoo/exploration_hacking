import subprocess
from pathlib import Path

class RustTool:
    """Tool for running Rust cargo commands (build, test, clippy)"""
    
    def __init__(self, name: str):
        self.name = name

    def run(self, results: dict, project_dir: Path) -> dict:
        """Run cargo command and update results dict"""
        try:
            result = subprocess.run(
                ["cargo", self.name, "--quiet"],
                cwd=project_dir,
                capture_output=True,
                text=True,
                timeout=60
            )
            results[f'{self.name}_passed'] = result.returncode == 0
            results[f'{self.name}_stderr'] = str(result.stderr)
        except Exception as e:
            results[f'{self.name}_passed'] = False
            results[f'{self.name}_stderr'] = f"{e}"
        return results