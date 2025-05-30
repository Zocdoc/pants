# Copyright 2021 Pants project contributors (see CONTRIBUTORS.md).
# Licensed under the Apache License, Version 2.0 (see LICENSE).
from __future__ import annotations

import json
from dataclasses import dataclass

from pants.backend.go.go_sources import load_go_binary
from pants.backend.go.go_sources.load_go_binary import LoadedGoBinaryRequest, setup_go_binary
from pants.engine.addresses import Address
from pants.engine.engine_aware import EngineAwareParameter
from pants.engine.fs import Digest, MergeDigests
from pants.engine.internals.native_engine import EMPTY_DIGEST
from pants.engine.intrinsics import execute_process, merge_digests
from pants.engine.process import Process
from pants.engine.rules import collect_rules, implicitly, rule
from pants.util.logging import LogLevel
from pants.util.ordered_set import FrozenOrderedSet


@dataclass(frozen=True)
class GenerateTestMainRequest(EngineAwareParameter):
    digest: Digest
    test_paths: FrozenOrderedSet[str]
    xtest_paths: FrozenOrderedSet[str]
    import_path: str
    register_cover: bool
    address: Address

    def debug_hint(self) -> str:
        return self.address.spec


@dataclass(frozen=True)
class GeneratedTestMain:
    digest: Digest
    has_tests: bool
    has_xtests: bool
    failed_exit_code_and_stderr: tuple[int, str] | None

    TEST_MAIN_FILE = "testmain.go"
    TEST_PKG = "_test"
    XTEST_PKG = "_xtest"


@rule
async def generate_testmain(request: GenerateTestMainRequest) -> GeneratedTestMain:
    generator_binary_name = "./generator"
    analyzer = await setup_go_binary(
        LoadedGoBinaryRequest("generate_testmain", ("main.go",), generator_binary_name),
        **implicitly(),
    )
    input_digest = await merge_digests(MergeDigests([request.digest, analyzer.digest]))

    test_paths = tuple(f"{GeneratedTestMain.TEST_PKG}:{path}" for path in request.test_paths)
    xtest_paths = tuple(f"{GeneratedTestMain.XTEST_PKG}:{path}" for path in request.xtest_paths)

    env = {}
    if request.register_cover:
        env["GENERATE_COVER"] = "1"

    result = await execute_process(
        Process(
            argv=(
                generator_binary_name,
                GeneratedTestMain.TEST_MAIN_FILE,
                request.import_path,
                *test_paths,
                *xtest_paths,
            ),
            input_digest=input_digest,
            env=env,
            description=f"Analyze Go test sources for {request.address}",
            level=LogLevel.DEBUG,
            output_files=(GeneratedTestMain.TEST_MAIN_FILE,),
        ),
        **implicitly(),
    )

    if result.exit_code != 0:
        return GeneratedTestMain(
            digest=EMPTY_DIGEST,
            has_tests=False,
            has_xtests=False,
            failed_exit_code_and_stderr=(result.exit_code, result.stderr.decode("utf-8")),
        )

    metadata = json.loads(result.stdout.decode("utf-8"))
    return GeneratedTestMain(
        digest=result.output_digest,
        has_tests=metadata["has_tests"],
        has_xtests=metadata["has_xtests"],
        failed_exit_code_and_stderr=None,
    )


def rules():
    return (*collect_rules(), *load_go_binary.rules())
