# Copyright 2021 Pants project contributors (see CONTRIBUTORS.md).
# Licensed under the Apache License, Version 2.0 (see LICENSE).
from __future__ import annotations

from dataclasses import dataclass

from pants.backend.codegen.thrift.scrooge.java import symbol_mapper
from pants.backend.codegen.thrift.scrooge.rules import (
    GenerateScroogeThriftSourcesRequest,
    generate_scrooge_thrift_sources,
)
from pants.backend.codegen.thrift.target_types import (
    ThriftDependenciesField,
    ThriftSourceField,
    ThriftSourcesGeneratorTarget,
    ThriftSourceTarget,
)
from pants.backend.java.target_types import JavaSourceField
from pants.backend.scala.subsystems.scala import ScalaSubsystem
from pants.build_graph.address import Address
from pants.engine.fs import AddPrefix
from pants.engine.intrinsics import digest_to_snapshot
from pants.engine.rules import collect_rules, implicitly, rule
from pants.engine.target import (
    FieldSet,
    GeneratedSources,
    GenerateSourcesRequest,
    InferDependenciesRequest,
    InferredDependencies,
)
from pants.engine.unions import UnionRule
from pants.jvm.dependency_inference import artifact_mapper
from pants.jvm.dependency_inference.artifact_mapper import (
    AllJvmArtifactTargets,
    UnversionedCoordinate,
    find_jvm_artifacts_or_raise,
)
from pants.jvm.subsystems import JvmSubsystem
from pants.jvm.target_types import JvmResolveField, PrefixedJvmJdkField, PrefixedJvmResolveField
from pants.source.source_root import SourceRootRequest, get_source_root
from pants.util.logging import LogLevel


class GenerateJavaFromThriftRequest(GenerateSourcesRequest):
    input = ThriftSourceField
    output = JavaSourceField


@dataclass(frozen=True)
class ScroogeThriftJavaDependenciesInferenceFieldSet(FieldSet):
    required_fields = (
        ThriftDependenciesField,
        JvmResolveField,
    )

    dependencies: ThriftDependenciesField
    resolve: JvmResolveField


class InferScroogeThriftJavaDependencies(InferDependenciesRequest):
    infer_from = ScroogeThriftJavaDependenciesInferenceFieldSet


@rule(desc="Generate Java from Thrift with Scrooge", level=LogLevel.DEBUG)
async def generate_java_from_thrift_with_scrooge(
    request: GenerateJavaFromThriftRequest,
) -> GeneratedSources:
    result = await generate_scrooge_thrift_sources(
        GenerateScroogeThriftSourcesRequest(
            thrift_source_field=request.protocol_target[ThriftSourceField],
            lang_id="java",
            lang_name="Java",
        ),
        **implicitly(),
    )

    source_root = await get_source_root(SourceRootRequest.for_target(request.protocol_target))

    source_root_restored = (
        await digest_to_snapshot(**implicitly(AddPrefix(result.snapshot.digest, source_root.path)))
        if source_root.path != "."
        else await digest_to_snapshot(result.snapshot.digest)
    )
    return GeneratedSources(source_root_restored)


@dataclass(frozen=True)
class ScroogeThriftJavaRuntimeForResolveRequest:
    resolve_name: str


@dataclass(frozen=True)
class ScroogeThriftJavaRuntimeForResolve:
    addresses: frozenset[Address]


@rule
async def resolve_scrooge_thrift_java_runtime_for_resolve(
    request: ScroogeThriftJavaRuntimeForResolveRequest,
    jvm_artifact_targets: AllJvmArtifactTargets,
    jvm: JvmSubsystem,
    scala_subsystem: ScalaSubsystem,
) -> ScroogeThriftJavaRuntimeForResolve:
    scala_version = scala_subsystem.version_for_resolve(request.resolve_name)
    scala_binary_version = scala_version.binary
    addresses = find_jvm_artifacts_or_raise(
        required_coordinates=[
            UnversionedCoordinate(
                group="org.apache.thrift",
                artifact="libthrift",
            ),
            UnversionedCoordinate(
                group="com.twitter",
                artifact=f"scrooge-core_{scala_binary_version}",
            ),
        ],
        resolve=request.resolve_name,
        jvm_artifact_targets=jvm_artifact_targets,
        jvm=jvm,
        subsystem="the Scrooge Java Thrift runtime",
        target_type="thrift_sources",
    )
    return ScroogeThriftJavaRuntimeForResolve(addresses)


@rule
async def inject_scrooge_thrift_java_dependencies(
    request: InferScroogeThriftJavaDependencies, jvm: JvmSubsystem
) -> InferredDependencies:
    resolve = request.field_set.resolve.normalized_value(jvm)
    dependencies_info = await resolve_scrooge_thrift_java_runtime_for_resolve(
        ScroogeThriftJavaRuntimeForResolveRequest(resolve), **implicitly()
    )
    return InferredDependencies(dependencies_info.addresses)


def rules():
    return (
        *collect_rules(),
        *symbol_mapper.rules(),
        UnionRule(GenerateSourcesRequest, GenerateJavaFromThriftRequest),
        UnionRule(InferDependenciesRequest, InferScroogeThriftJavaDependencies),
        ThriftSourceTarget.register_plugin_field(PrefixedJvmJdkField),
        ThriftSourcesGeneratorTarget.register_plugin_field(PrefixedJvmJdkField),
        ThriftSourceTarget.register_plugin_field(PrefixedJvmResolveField),
        ThriftSourcesGeneratorTarget.register_plugin_field(PrefixedJvmResolveField),
        # Rules to avoid rule graph errors.
        *artifact_mapper.rules(),
    )
