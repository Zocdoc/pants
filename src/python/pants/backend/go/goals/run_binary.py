# Copyright 2021 Pants project contributors (see CONTRIBUTORS.md).
# Licensed under the Apache License, Version 2.0 (see LICENSE).

import os.path

from pants.backend.go.goals.package_binary import GoBinaryFieldSet, package_go_binary
from pants.core.goals.run import RunRequest
from pants.engine.rules import collect_rules, rule


@rule
async def create_go_binary_run_request(field_set: GoBinaryFieldSet) -> RunRequest:
    binary = await package_go_binary(field_set)
    artifact_relpath = binary.artifacts[0].relpath
    assert artifact_relpath is not None
    return RunRequest(digest=binary.digest, args=(os.path.join("{chroot}", artifact_relpath),))


def rules():
    return [
        *collect_rules(),
        *GoBinaryFieldSet.rules(),
    ]
