# Copyright 2019 Pants project contributors (see CONTRIBUTORS.md).
# Licensed under the Apache License, Version 2.0 (see LICENSE).

from __future__ import annotations

from pants.backend.python.subsystems.setup import PythonSetup
from pants.backend.python.subsystems.setuptools import PythonDistributionFieldSet
from pants.backend.python.util_rules.dists import run_pep517_build
from pants.backend.python.util_rules.package_dists import create_dist_build_request
from pants.backend.python.util_rules.package_dists import rules as package_dists_rules
from pants.core.goals.package import BuiltPackage, BuiltPackageArtifact, PackageFieldSet
from pants.engine.intrinsics import digest_to_snapshot
from pants.engine.rules import collect_rules, implicitly, rule
from pants.engine.unions import UnionMembership, UnionRule


@rule
async def package_python_dist(
    field_set: PythonDistributionFieldSet,
    python_setup: PythonSetup,
    union_membership: UnionMembership,
) -> BuiltPackage:
    dist_build_request = await create_dist_build_request(
        dist_target_address=field_set.address,
        python_setup=python_setup,
        union_membership=union_membership,
        # raises an error if both dist_tgt.wheel and dist_tgt.sdist are False
        validate_wheel_sdist=True,
    )
    setup_py_result = await run_pep517_build(dist_build_request, **implicitly())
    dist_snapshot = await digest_to_snapshot(setup_py_result.output)
    return BuiltPackage(
        setup_py_result.output,
        tuple(BuiltPackageArtifact(path) for path in dist_snapshot.files),
    )


def rules():
    return [
        *package_dists_rules(),
        *collect_rules(),
        UnionRule(PackageFieldSet, PythonDistributionFieldSet),
    ]
