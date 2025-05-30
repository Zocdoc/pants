# Copyright 2021 Pants project contributors (see CONTRIBUTORS.md).
# Licensed under the Apache License, Version 2.0 (see LICENSE).

from __future__ import annotations

import os.path
from dataclasses import dataclass
from typing import Any

from pants.backend.go.lint.vet.skip_field import SkipGoVetField
from pants.backend.go.lint.vet.subsystem import GoVetSubsystem
from pants.backend.go.target_types import GoPackageSourcesField
from pants.backend.go.util_rules.go_mod import (
    GoModInfoRequest,
    OwningGoModRequest,
    determine_go_mod_info,
    find_owning_go_mod,
)
from pants.backend.go.util_rules.sdk import GoSdkProcess
from pants.core.goals.lint import LintResult, LintTargetsRequest
from pants.core.util_rules.partitions import PartitionerType
from pants.core.util_rules.source_files import SourceFilesRequest, determine_source_files
from pants.engine.fs import MergeDigests
from pants.engine.internals.selectors import concurrently
from pants.engine.intrinsics import execute_process, merge_digests
from pants.engine.rules import collect_rules, implicitly, rule
from pants.engine.target import FieldSet, Target
from pants.util.logging import LogLevel
from pants.util.strutil import pluralize


@dataclass(frozen=True)
class GoVetFieldSet(FieldSet):
    required_fields = (GoPackageSourcesField,)

    sources: GoPackageSourcesField

    @classmethod
    def opt_out(cls, tgt: Target) -> bool:
        return tgt.get(SkipGoVetField).value


class GoVetRequest(LintTargetsRequest):
    field_set_type = GoVetFieldSet
    tool_subsystem = GoVetSubsystem
    partitioner_type = PartitionerType.DEFAULT_SINGLE_PARTITION


@rule(level=LogLevel.DEBUG)
async def run_go_vet(request: GoVetRequest.Batch[GoVetFieldSet, Any]) -> LintResult:
    source_files = await determine_source_files(
        SourceFilesRequest(field_set.sources for field_set in request.elements)
    )

    owning_go_mods = await concurrently(
        find_owning_go_mod(OwningGoModRequest(field_set.address), **implicitly())
        for field_set in request.elements
    )
    owning_go_mod_addresses = {x.address for x in owning_go_mods}
    go_mod_infos = await concurrently(
        determine_go_mod_info(GoModInfoRequest(address)) for address in owning_go_mod_addresses
    )

    input_digest = await merge_digests(
        MergeDigests([source_files.snapshot.digest, *(info.digest for info in set(go_mod_infos))])
    )
    package_dirs = sorted({os.path.dirname(f) for f in source_files.snapshot.files})
    process_result = await execute_process(
        **implicitly(
            GoSdkProcess(
                ("vet", *(f"./{p}" for p in package_dirs)),
                input_digest=input_digest,
                description=f"Run `go vet` on {pluralize(len(source_files.snapshot.files), 'file')}.",
            )
        )
    )
    return LintResult.create(request, process_result)


def rules():
    return (
        *collect_rules(),
        *GoVetRequest.rules(),
    )
