import re
from typing import Optional
from uuid import uuid4
from pathlib import Path
import shutil
from typing import Callable
from src.ox_rust.tools import RustTool


class RustCodeEvaluator:
    """Evaluates Rust code using cargo toolchain and various heuristics"""
    
    @staticmethod
    def extract_regex(text: str, pattern: str) -> Optional[str]:
        """Extract text using regex pattern"""
        # match = re.search(pattern, text, re.DOTALL)
        # return match.group(1) if match else None
        match = re.findall(pattern, text, re.DOTALL)
        match = match[-1] if match else None
        return match

    @staticmethod
    def extract_rust_code(response: str) -> str:
        """Extract Rust code from markdown code blocks"""
        pattern = r'```rust\n(.*?)\n```'
        code = RustCodeEvaluator.extract_regex(response, pattern)
        return code if code else response

    @staticmethod
    def extract_test_code(response: str) -> str:
        """Extract test module from Rust code"""
        pattern = r'(#\[cfg\(test\)\]\s*mod\s+tests\s*\{.*?\})'
        return RustCodeEvaluator.extract_regex(response, pattern)

    @staticmethod
    def get_cargo_template() -> str:
        """Get Cargo.toml template"""
        return """
[package]
name = "rust-program"
version = "0.1.0"
edition = "2021"

[dependencies]
"""

    @staticmethod
    def get_rust_template() -> str:
        """Get main.rs template"""
        return """
#![allow(dead_code)]
// {code}

// Need basic main function for the code to compile
fn main() {
  println!("Hello World");
}
"""

    def setup_and_test_rust_project(self, row: dict, tools: list) -> dict:
        """
        Set up temporary Rust project and run tests
        
        This is where the actual Rust code execution happens:
        1. Creates a temporary Rust project structure
        2. Injects the model-generated code into a main.rs file
        3. Runs cargo commands (build/test/clippy) on the real code
        4. Returns results to be used as RL rewards
        """
        # Create temporary project directory with unique name
        project_dir = Path("outputs") / "tests" / f"temp_rust_project_{uuid4()}"
        project_dir_src = project_dir / "src"
        project_dir_src.mkdir(parents=True, exist_ok=True)

        # Extract and prepare the model-generated Rust code
        template = self.get_rust_template()
        rust_code = self.extract_rust_code(row['rust_code'])  # Extract from ```rust``` blocks
        template = template.replace("// {code}", rust_code)

        # Write the actual Rust source file
        main_rs_path = project_dir_src / "main.rs"
        with open(main_rs_path, "w") as f:
            f.write(template)
            
        # print(f"Created Rust project at: {project_dir}")
        # print(f"Generated code:\n{template}")

        # Write Cargo.toml to make it a valid Rust project
        cargo_file_path = project_dir / "Cargo.toml"
        with open(cargo_file_path, "w") as f:
            f.write(self.get_cargo_template())

        # Actually run the cargo tools on the generated code
        results = {}
        for tool in tools:
            # print(f"Running: cargo {tool.name}")
            results = tool.run(results, project_dir)
            # print(f"Result: {results}")

        # Clean up the temporary project
        shutil.rmtree(project_dir)
        return results


class RewardFunctions:
    """Collection of reward functions for GRPO training"""
    
    def __init__(self, evaluator: RustCodeEvaluator):
        self.evaluator = evaluator

    def create_reward_function(self, name: str, func: Callable):
        """Create a logged reward function"""
        def reward_func(prompts, completions, **kwargs) -> list[float]:
            contents = [completion[0]["content"] for completion in completions]
            return func(contents, **kwargs)
        
        reward_func.__name__ = name
        return reward_func

    def non_empty_reward(self, contents: list[str], **kwargs) -> list[float]:
        """Reward for non-empty responses with proper structure"""
        results = []
        for content in contents:
            if not (self._has_code_block(content) and self._has_test_block(content)):
                results.append(0.0)
                continue
                
            code = self.evaluator.extract_rust_code(content)
            num_non_empty = 0
            for line in code.split("\n"):
                line = line.strip()
                if line.startswith("//") or len(line) < 2:
                    continue
                num_non_empty += 1
            results.append(1.0 if num_non_empty >= 3 else 0.0)
        return results

    def test_asserts_reward(self, contents: list[str], **kwargs) -> list[float]:
        """Reward for having assert statements in tests"""
        results = []
        for content in contents:
            test_code = self.evaluator.extract_test_code(content)
            if not test_code:
                results.append(0.0)
                continue

            unique_asserts = set()
            for line in test_code.split("\n"):
                line = line.strip()
                if line.startswith("assert!(") or line.startswith("assert_eq!("):
                    unique_asserts.add(line)
            
            if len(unique_asserts) >= 4:
                results.append(1.0)
            else:
                results.append(0.25 * len(unique_asserts))
        return results

    def code_block_reward(self, contents: list[str], **kwargs) -> list[float]:
        """Reward for having proper code blocks"""
        return [0.5 if self._has_code_block(content) else 0.0 for content in contents]

    def test_block_reward(self, contents: list[str], **kwargs) -> list[float]:
        """Reward for having test blocks"""
        return [0.5 if self._has_test_block(content) else 0.0 for content in contents]

    def cargo_build_reward(self, contents: list[str], **kwargs) -> list[float]:
        """Reward for passing cargo build"""
        results = []
        for content in contents:
            code = self.evaluator.extract_rust_code(content)
            data = {'rust_code': code}
            tools = [RustTool("build")]
            cargo_results = self.evaluator.setup_and_test_rust_project(data, tools)
            score = 1.0 if cargo_results['build_passed'] else 0.0
            results.append(score)
        return results

    def cargo_clippy_reward(self, contents: list[str], **kwargs) -> list[float]:
        """Reward for passing cargo clippy"""
        results = []
        for content in contents:
            code = self.evaluator.extract_rust_code(content)
            data = {'rust_code': code}
            tools = [RustTool("clippy")]
            cargo_results = self.evaluator.setup_and_test_rust_project(data, tools)
            score = 1.0 if cargo_results['clippy_passed'] else 0.0
            results.append(score)
        return results

    def cargo_test_reward(self, contents: list[str], **kwargs) -> list[float]:
        """Reward for passing cargo test (weighted higher)"""
        results = []
        for content in contents:
            code = self.evaluator.extract_rust_code(content)
            test_code = self.evaluator.extract_test_code(code)
            
            score = 0.0
            if test_code:
                data = {'rust_code': code}
                tools = [RustTool("test")]
                cargo_results = self.evaluator.setup_and_test_rust_project(data, tools)
                # Higher reward for passing tests
                score = 2.0 if cargo_results['test_passed'] else 0.0
            results.append(score)
        print(f"cargo_test_reward: {results}")
        return results

    def _has_code_block(self, content: str) -> bool:
        """Check if response has proper code block structure"""
        return bool(self.evaluator.extract_rust_code(content) and "fn " in content)

    def _has_test_block(self, content: str) -> bool:
        """Check if response has test block"""
        return bool(self.evaluator.extract_test_code(content))