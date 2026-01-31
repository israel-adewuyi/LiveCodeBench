import base64
import json
import pickle
import zlib
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from datasets import load_dataset
from huggingface_hub import hf_hub_download


class Platform(Enum):
    LEETCODE = "leetcode"
    CODEFORCES = "codeforces"
    ATCODER = "atcoder"


class Difficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class TestType(Enum):
    STDIN = "stdin"
    FUNCTIONAL = "functional"


@dataclass
class Test:
    input: str
    output: str
    testtype: TestType

    def __post_init__(self):
        self.testtype = TestType(self.testtype)
        # if self.testtype == TestType.FUNCTIONAL:
        #     self.input = json.loads(self.input)
        #     self.output = json.loads(self.output)


@dataclass
class CodeGenerationProblem:
    question_title: str
    question_content: str
    platform: Platform
    question_id: str
    contest_id: str
    contest_date: datetime
    starter_code: str
    difficulty: Difficulty
    public_test_cases: list[Test]
    private_test_cases: list[Test]
    metadata: dict

    def __post_init__(self):
        self.platform = Platform(self.platform)
        self.difficulty = Difficulty(self.difficulty)
        self.contest_date = datetime.fromisoformat(self.contest_date)

        self.public_test_cases = json.loads(self.public_test_cases)  # type: ignore
        self.public_test_cases = [Test(**t) for t in self.public_test_cases]

        try:
            self.private_test_cases = json.loads(self.private_test_cases)  # type: ignore
        except Exception:
            self.private_test_cases = json.loads(
                pickle.loads(
                    zlib.decompress(
                        base64.b64decode(self.private_test_cases.encode("utf-8"))  # type: ignore
                    )
                )
            )  # type: ignore
        self.private_test_cases = [Test(**t) for t in self.private_test_cases]

        self.metadata = json.loads(self.metadata)  # type: ignore

    def insert_output(self, output_list: list[str], code_list: list[str]) -> dict:
        return {
            "question_title": self.question_title,
            "question_content": self.question_content,
            "platform": self.platform.value,
            "question_id": self.question_id,
            "contest_id": self.contest_id,
            "contest_date": self.contest_date.isoformat(),
            "starter_code": self.starter_code,
            "difficulty": self.difficulty.value,
            "output_list": output_list,
            "code_list": code_list,
        }

    def insert_output_evaluation(
        self,
        output_list: list[str],
        code_list: list[str],
        graded_list: list[bool],
        **kwargs,
    ) -> dict:
        output = self.insert_output(output_list, code_list)
        output["graded_list"] = graded_list
        output["pass@1"] = graded_list.count(True) / len(graded_list)
        for k, v in kwargs.items():
            output[k] = v
        return output

    def get_evaluation_sample(self):
        return {
            "input_output": json.dumps(
                {
                    "inputs": [
                        t.input
                        for t in self.public_test_cases + self.private_test_cases
                    ],
                    "outputs": [
                        t.output
                        for t in self.public_test_cases + self.private_test_cases
                    ],
                    "fn_name": self.metadata.get("func_name", None),
                }
            ),
        }


_ALLOWED_FILES = {
    1: "test.jsonl",
    2: "test2.jsonl",
    3: "test3.jsonl",
    4: "test4.jsonl",
    5: "test5.jsonl",
    6: "test6.jsonl",
}


def _parse_version_token(token: str) -> int | None:
    if token.startswith("v") and token[1:].isdigit():
        return int(token[1:])
    return None


def _normalize_release_version(release_version: str) -> str:
    if release_version == "release_latest":
        return "v6"
    if release_version.startswith("release_"):
        return release_version.replace("release_", "", 1)
    return release_version


def _resolve_jsonl_files(release_version: str) -> list[str]:
    rv = _normalize_release_version(release_version)

    if "_" in rv:
        start_token, end_token = rv.split("_", 1)
        start = _parse_version_token(start_token)
        end = _parse_version_token(end_token)
        if start is None or end is None:
            raise ValueError(f"Invalid release_version: {release_version}")
        if start > end:
            start, end = end, start
        return [_ALLOWED_FILES[i] for i in range(start, end + 1)]

    version = _parse_version_token(rv)
    if version is None:
        raise ValueError(f"Invalid release_version: {release_version}")
    return [_ALLOWED_FILES[i] for i in range(1, version + 1)]


def _load_code_generation_lite_dataset(release_version: str):
    jsonl_files = _resolve_jsonl_files(release_version)
    file_paths = [
        hf_hub_download(
            repo_id="livecodebench/code_generation_lite",
            filename=jsonl_file,
            repo_type="dataset",
        )
        for jsonl_file in jsonl_files
    ]
    return load_dataset("json", data_files=file_paths, split="train")


def load_code_generation_dataset(
    release_version="release_v1", start_date=None, end_date=None
) -> list[CodeGenerationProblem]:
    dataset = _load_code_generation_lite_dataset(release_version)
    dataset = [CodeGenerationProblem(**p) for p in dataset]  # type: ignore
    if start_date is not None:
        p_start_date = datetime.strptime(start_date, "%Y-%m-%d")
        dataset = [e for e in dataset if p_start_date <= e.contest_date]

    if end_date is not None:
        p_end_date = datetime.strptime(end_date, "%Y-%m-%d")
        dataset = [e for e in dataset if e.contest_date <= p_end_date]

    print(f"Loaded {len(dataset)} problems")
    return dataset


def load_code_generation_dataset_not_fast(release_version="release_v1") -> list[CodeGenerationProblem]:
    dataset = load_dataset("livecodebench/code_generation", split="test")
    dataset = [CodeGenerationProblem(**p) for p in dataset]  # type: ignore
    print(f"Loaded {len(dataset)} problems")
    return dataset


if __name__ == "__main__":
    dataset = load_code_generation_dataset()
