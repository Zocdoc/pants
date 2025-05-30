# Copyright 2020 Pants project contributors (see CONTRIBUTORS.md).
# Licensed under the Apache License, Version 2.0 (see LICENSE).
from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePath

from pants.backend.codegen.protobuf import protoc
from pants.backend.codegen.protobuf.java import dependency_inference, symbol_mapper
from pants.backend.codegen.protobuf.java.subsystem import JavaProtobufGrpcSubsystem
from pants.backend.codegen.protobuf.protoc import Protoc
from pants.backend.codegen.protobuf.target_types import (
    ProtobufGrpcToggleField,
    ProtobufSourceField,
    ProtobufSourcesGeneratorTarget,
    ProtobufSourceTarget,
)
from pants.backend.experimental.java.register import rules as java_backend_rules
from pants.backend.java.target_types import JavaSourceField
from pants.core.goals.resolves import ExportableTool
from pants.core.util_rules.external_tool import download_external_tool
from pants.core.util_rules.source_files import SourceFilesRequest
from pants.core.util_rules.stripped_source_files import strip_source_roots
from pants.engine.fs import (
    AddPrefix,
    CreateDigest,
    Digest,
    Directory,
    FileEntry,
    MergeDigests,
    RemovePrefix,
)
from pants.engine.internals.graph import transitive_targets
from pants.engine.intrinsics import (
    create_digest,
    digest_to_snapshot,
    get_digest_entries,
    merge_digests,
    remove_prefix,
)
from pants.engine.platform import Platform
from pants.engine.process import Process, fallible_to_exec_result_or_raise
from pants.engine.rules import collect_rules, concurrently, implicitly, rule
from pants.engine.target import GeneratedSources, GenerateSourcesRequest, TransitiveTargetsRequest
from pants.engine.unions import UnionRule
from pants.jvm.resolve.coursier_fetch import ToolClasspathRequest, materialize_classpath_for_tool
from pants.jvm.resolve.jvm_tool import GenerateJvmLockfileFromTool
from pants.jvm.target_types import PrefixedJvmJdkField, PrefixedJvmResolveField
from pants.source.source_root import SourceRootRequest, get_source_root
from pants.util.logging import LogLevel


class GenerateJavaFromProtobufRequest(GenerateSourcesRequest):
    input = ProtobufSourceField
    output = JavaSourceField


@dataclass(frozen=True)
class ProtobufJavaGrpcPlugin:
    digest: Digest
    path: str


@rule
async def resolve_protobuf_java_grpc_plugin(
    platform: Platform,
    tool: JavaProtobufGrpcSubsystem,
) -> ProtobufJavaGrpcPlugin:
    lockfile_request = GenerateJvmLockfileFromTool.create(tool)
    classpath = await materialize_classpath_for_tool(
        ToolClasspathRequest(lockfile=lockfile_request)
    )

    # TODO: Improve `ToolClasspath` API so that the filenames corresponding to a coordinate are identified by a
    # mapping. Work-around the lack of such information by looking for a platform-specific string in the filenames
    # provided in the classpath.
    platform_part = {
        Platform.macos_arm64: "exe_osx-aarch_64",
        Platform.macos_x86_64: "exe_osx-x86_64",
        Platform.linux_arm64: "exe_linux-aarch_64",
        Platform.linux_x86_64: "exe_linux-x86_64",
    }[platform]

    classpath_entries = await get_digest_entries(classpath.digest)
    candidate_plugin_entries = []
    for classpath_entry in classpath_entries:
        if isinstance(classpath_entry, FileEntry):
            path = PurePath(classpath_entry.path)
            if platform_part in path.name:
                candidate_plugin_entries.append(classpath_entry)

    assert len(candidate_plugin_entries) == 1

    plugin_digest = await create_digest(
        CreateDigest(
            [
                FileEntry(
                    path="protoc-gen-grpc-java",
                    file_digest=candidate_plugin_entries[0].file_digest,
                    is_executable=True,
                )
            ]
        )
    )

    return ProtobufJavaGrpcPlugin(digest=plugin_digest, path="protoc-gen-grpc-java")


@rule(desc="Generate Java from Protobuf", level=LogLevel.DEBUG)
async def generate_java_from_protobuf(
    request: GenerateJavaFromProtobufRequest,
    protoc: Protoc,
    grpc_plugin: ProtobufJavaGrpcPlugin,  # TODO: Don't access grpc plugin unless gRPC codegen is enabled.
    platform: Platform,
) -> GeneratedSources:
    download_protoc_request = download_external_tool(protoc.get_request(platform))

    output_dir = "_generated_files"
    create_output_dir_request = create_digest(CreateDigest([Directory(output_dir)]))

    # Protoc needs all transitive dependencies on `protobuf_source` to work properly. It won't
    # actually generate those dependencies; it only needs to look at their .proto files to work
    # with imports.
    transitive_targets_for_protobuf_source = await transitive_targets(
        TransitiveTargetsRequest([request.protocol_target.address]), **implicitly()
    )

    # NB: By stripping the source roots, we avoid having to set the value `--proto_path`
    # for Protobuf imports to be discoverable.
    all_stripped_sources_request = strip_source_roots(
        **implicitly(
            SourceFilesRequest(
                tgt[ProtobufSourceField]
                for tgt in transitive_targets_for_protobuf_source.closure
                if tgt.has_field(ProtobufSourceField)
            )
        )
    )
    target_stripped_sources_request = strip_source_roots(
        **implicitly(SourceFilesRequest([request.protocol_target[ProtobufSourceField]]))
    )

    (
        downloaded_protoc_binary,
        empty_output_dir,
        all_sources_stripped,
        target_sources_stripped,
    ) = await concurrently(
        download_protoc_request,
        create_output_dir_request,
        all_stripped_sources_request,
        target_stripped_sources_request,
    )

    unmerged_digests = [
        all_sources_stripped.snapshot.digest,
        downloaded_protoc_binary.digest,
        empty_output_dir,
    ]
    input_digest = await merge_digests(MergeDigests(unmerged_digests))

    immutable_input_digests = {}
    if request.protocol_target.get(ProtobufGrpcToggleField).value:
        immutable_input_digests["__grpc"] = grpc_plugin.digest

    argv = [downloaded_protoc_binary.exe, "--java_out", output_dir]
    if request.protocol_target.get(ProtobufGrpcToggleField).value:
        argv.extend(
            [
                f"--plugin=protoc-gen-grpc-java=./__grpc/{grpc_plugin.path}",
                f"--grpc-java_out={output_dir}",
            ]
        )

    argv.extend(target_sources_stripped.snapshot.files)
    result = await fallible_to_exec_result_or_raise(
        **implicitly(
            Process(
                argv,
                input_digest=input_digest,
                immutable_input_digests=immutable_input_digests,
                description=f"Generating Java sources from {request.protocol_target.address}.",
                level=LogLevel.DEBUG,
                output_directories=(output_dir,),
            )
        )
    )

    normalized_digest, source_root = await concurrently(
        remove_prefix(RemovePrefix(result.output_digest, output_dir)),
        get_source_root(SourceRootRequest.for_target(request.protocol_target)),
    )

    source_root_restored = (
        await digest_to_snapshot(**implicitly(AddPrefix(normalized_digest, source_root.path)))
        if source_root.path != "."
        else await digest_to_snapshot(normalized_digest)
    )
    return GeneratedSources(source_root_restored)


def rules():
    return [
        *collect_rules(),
        *dependency_inference.rules(),
        *symbol_mapper.rules(),
        *protoc.rules(),
        UnionRule(GenerateSourcesRequest, GenerateJavaFromProtobufRequest),
        UnionRule(ExportableTool, JavaProtobufGrpcSubsystem),
        ProtobufSourceTarget.register_plugin_field(PrefixedJvmJdkField),
        ProtobufSourcesGeneratorTarget.register_plugin_field(PrefixedJvmJdkField),
        ProtobufSourceTarget.register_plugin_field(PrefixedJvmResolveField),
        ProtobufSourcesGeneratorTarget.register_plugin_field(PrefixedJvmResolveField),
        # Bring in the Java backend (since this backend compiles Java code) to avoid rule graph errors.
        # TODO: Figure out whether a subset of rules can be brought in to still avoid rule graph errors.
        *java_backend_rules(),
    ]
