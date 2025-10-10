# Copyright 2023 Pants project contributors (see CONTRIBUTORS.md).
# Licensed under the Apache License, Version 2.0 (see LICENSE).
from __future__ import annotations

import itertools
import logging
import os.path
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import DefaultDict
from dataclasses import dataclass
from pathlib import PurePath

from pants.backend.javascript import package_json
from pants.core.util_rules.stripped_source_files import StrippedFileNameRequest, strip_file_name
from pants.engine.target import AllTargets, Target
from pants.util.frozendict import FrozenDict
from pants.util.logging import LogLevel
from pants.backend.javascript.package_json import (
    FirstPartyNodePackageTargets,
    NodePackageDependenciesField,
    NodePackageNameField,
    NodeThirdPartyPackageNameField,
    OwningNodePackage,
    OwningNodePackageRequest,
    PackageJsonImports,
    PackageJsonSourceField,
)
from pants.backend.javascript.package_json import find_owning_package
from pants.backend.javascript.package_json import find_owning_package as find_owning_package_get
from pants.backend.javascript.package_json import (
    script_entrypoints_for_source,
    subpath_imports_for_source,
)
from pants.backend.javascript.subsystems.nodejs_infer import NodeJSInfer
from pants.backend.javascript.target_types import (
    JS_FILE_EXTENSIONS,
    JSRuntimeDependenciesField,
    JSRuntimeSourceField,
)
from pants.backend.jsx.target_types import JSX_FILE_EXTENSIONS
from pants.backend.tsx.target_types import TSX_FILE_EXTENSIONS
from pants.backend.typescript import tsconfig
from pants.backend.typescript.target_types import TS_FILE_EXTENSIONS
from pants.backend.typescript.tsconfig import ParentTSConfigRequest, TSConfig, find_parent_ts_config
from pants.build_graph.address import Address
from pants.core.util_rules.unowned_dependency_behavior import (
    UnownedDependencyError,
    UnownedDependencyUsage,
)
from pants.engine.addresses import Addresses
from pants.engine.fs import PathGlobs
from pants.engine.internals.graph import (
    OwnersRequest,
    find_owners,
    hydrate_sources,
    resolve_targets,
)
from pants.engine.internals.native_dep_inference import ParsedJavascriptDependencyCandidate
from pants.engine.internals.native_engine import InferenceMetadata, NativeDependenciesRequest
from pants.engine.internals.selectors import concurrently
from pants.engine.intrinsics import parse_javascript_deps, path_globs_to_paths
from pants.engine.rules import Rule, collect_rules, implicitly, rule
from pants.engine.target import (
    FieldSet,
    HydrateSourcesRequest,
    InferDependenciesRequest,
    InferredDependencies,
)
from pants.engine.unions import UnionRule
from pants.util.docutil import doc_url
from pants.util.frozendict import FrozenDict
from pants.util.logging import LogLevel
from pants.util.ordered_set import FrozenOrderedSet
from pants.util.strutil import bullet_list, softwrap
from pants.core.util_rules.source_files import SourceFilesRequest, determine_source_files

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NodePackageInferenceFieldSet(FieldSet):
    required_fields = (PackageJsonSourceField, NodePackageDependenciesField)

    source: PackageJsonSourceField
    dependencies: NodePackageDependenciesField


class InferNodePackageDependenciesRequest(InferDependenciesRequest):
    infer_from = NodePackageInferenceFieldSet


@dataclass(frozen=True)
class JSSourceInferenceFieldSet(FieldSet):
    required_fields = (JSRuntimeSourceField, JSRuntimeDependenciesField)

    source: JSRuntimeSourceField
    dependencies: JSRuntimeDependenciesField


class InferJSDependenciesRequest(InferDependenciesRequest):
    infer_from = JSSourceInferenceFieldSet


@rule(desc="infer node package dependencies", level=LogLevel.INFO)
async def infer_node_package_dependencies(
    request: InferNodePackageDependenciesRequest,
    nodejs_infer: NodeJSInfer,
) -> InferredDependencies:
    if not nodejs_infer.package_json_entry_points:
        return InferredDependencies(())
    entry_points = await script_entrypoints_for_source(request.field_set.source)
    candidate_js_files = await find_owners(
        OwnersRequest(tuple(entry_points.globs_from_root())), **implicitly()
    )
    js_targets = await resolve_targets(**implicitly(Addresses(candidate_js_files)))
    return InferredDependencies(
        tgt.address for tgt in js_targets if tgt.has_field(JSRuntimeSourceField)
    )


@dataclass(frozen=True)
class AllJavaScriptTargets:
    """A collection of all JavaScript targets in the project, separated into first-party and third-party."""
    first_party: tuple[Target, ...]  # JS/TS source targets
    third_party: tuple[Target, ...]  # node_third_party_package targets


@dataclass(frozen=True)
class FirstPartyJavaScriptMapping:
    """A mapping of file paths to their owning addresses."""

    paths_to_addresses: FrozenDict[str, tuple[Address, ...]]

    @classmethod
    def create(
        cls,
        paths_to_addresses: Mapping[str, Iterable[Address]],
    ) -> FirstPartyJavaScriptMapping:
        return FirstPartyJavaScriptMapping(
            FrozenDict(
                (path, tuple(sorted(addresses))) 
                for path, addresses in sorted(paths_to_addresses.items())
            )
        )

    def get_addresses(self, file_paths: Iterable[str]) -> Addresses:
        """Find all addresses that provide the given file paths."""
        result: list[Address] = []
        for file_path in file_paths:
            # Get just the filename without extension
            normalized_path = PurePath(file_path).name.rpartition('.')[0]
            logger.info(f"Looking up normalized path: {normalized_path}")
            # Find any stored paths that end with this filename
            for stored_path, addresses in self.paths_to_addresses.items():
                if stored_path.endswith(f"/{normalized_path}") or stored_path == normalized_path:
                    result.extend(addresses)
        logger.info(f"Found addresses: {result}")
        return Addresses(result)


def normalize_file_path(path: PurePath) -> str:
    """Normalize a file path by removing extension and handling index files.
    
    Uses the full repository path to avoid conflicts between files with the same name
    in different directories.
    
    For example:
    - project1/src/components/Button.tsx -> project1/src/components/Button
    - project2/src/components/index.ts -> project2/src/components
    """
    if path.name == "index.ts" or path.name == "index.js" or path.name == "index.tsx" or path.name == "index.jsx":
        return path.parent.as_posix()
    return path.with_suffix("").as_posix()


class NodePackageCandidateMap(FrozenDict[str, Address]):
    pass


@dataclass(frozen=True)
class RequestNodePackagesCandidateMap:
    address: Address


@rule
async def map_candidate_node_packages(
    req: RequestNodePackagesCandidateMap, first_party: FirstPartyNodePackageTargets
) -> NodePackageCandidateMap:
    owning_pkg = await find_owning_package_get(OwningNodePackageRequest(req.address))
    candidate_tgts = itertools.chain(
        first_party, owning_pkg.third_party if owning_pkg != OwningNodePackage.no_owner() else ()
    )
    return NodePackageCandidateMap(
        (tgt[NodePackageNameField].value, tgt.address) for tgt in candidate_tgts
    )


def _create_inference_metadata(
    imports: PackageJsonImports, config: TSConfig | None
) -> InferenceMetadata:
    return InferenceMetadata.javascript(
        imports.root_dir,
        dict(imports.imports),
        config.resolution_root_dir if config else None,
        dict(config.paths or {}) if config else {},
    )


async def _prepare_inference_metadata(address: Address, file_path: str) -> InferenceMetadata:
    owning_pkg, maybe_config = await concurrently(
        find_owning_package(OwningNodePackageRequest(address)),
        find_parent_ts_config(ParentTSConfigRequest(file_path), **implicitly()),
    )
    if not owning_pkg.target:
        return InferenceMetadata.javascript(
            (
                os.path.dirname(maybe_config.ts_config.path)
                if maybe_config.ts_config
                else address.spec_path
            ),
            {},
            maybe_config.ts_config.resolution_root_dir if maybe_config.ts_config else None,
            dict(maybe_config.ts_config.paths or {}) if maybe_config.ts_config else {},
        )
    return _create_inference_metadata(
        await subpath_imports_for_source(owning_pkg.target[PackageJsonSourceField]),
        maybe_config.ts_config,
    )


def _add_extensions(file_imports: frozenset[str], file_extensions: tuple[str, ...]) -> PathGlobs:
    extensions = file_extensions + tuple(f"/index{ext}" for ext in file_extensions)
    valid_file_extensions = set(file_extensions)
    return PathGlobs(
        string
        for file_import in file_imports
        for string in (
            [file_import]
            if PurePath(file_import).suffix in valid_file_extensions
            else [f"{file_import}{ext}" for ext in extensions]
        )
    )


async def _determine_import_from_candidates(
    candidates: ParsedJavascriptDependencyCandidate,
    package_candidate_map: NodePackageCandidateMap,
    first_party_mapping: FirstPartyJavaScriptMapping
) -> Addresses:

    logger.info(f"Determining import from candidates: {candidates.file_imports} and {candidates.package_imports}")

    # First check first-party mapping for file imports
    addresses = first_party_mapping.get_addresses(candidates.file_imports)
    if addresses:
        return addresses

    # If no first-party matches found, check third-party packages
    non_path_string_bases = FrozenOrderedSet(
        (
            # Handle scoped packages like "@foo/bar"
            # Ref: https://docs.npmjs.com/cli/v11/using-npm/scope
            "/".join(non_path_string.split("/")[:2])
            if non_path_string.startswith("@")
            # Handle regular packages like "foo"
            else non_path_string.partition("/")[0]
        )
        for non_path_string in candidates.package_imports
    )
    return Addresses(
        package_candidate_map[pkg_name]
        for pkg_name in non_path_string_bases
        if pkg_name in package_candidate_map
    )


def _handle_unowned_imports(
    address: Address,
    unowned_dependency_behavior: UnownedDependencyUsage,
    unowned_imports: frozenset[str],
) -> None:
    if not unowned_imports or unowned_dependency_behavior is UnownedDependencyUsage.DoNothing:
        return

    url = doc_url(
        "docs/using-pants/troubleshooting-common-issues#import-errors-and-missing-dependencies"
    )
    msg = softwrap(
        f"""
        Pants cannot infer owners for the following imports in the target {address}:

        {bullet_list(sorted(unowned_imports))}

        If you do not expect an import to be inferable, add `// pants: no-infer-dep` to the
        import line. Otherwise, see {url} for common problems.
        """
    )
    if unowned_dependency_behavior is UnownedDependencyUsage.LogWarning:
        logger.warning(msg)
    else:
        raise UnownedDependencyError(msg)


def _is_node_builtin_module(import_string: str) -> bool:
    """https://nodejs.org/api/modules.html#built-in-modules."""
    return import_string.startswith("node:")


@rule(desc="infer js source dependencies", level=LogLevel.INFO)
async def infer_js_source_dependencies(
    request: InferJSDependenciesRequest,
    nodejs_infer: NodeJSInfer,
    first_party_mapping: FirstPartyJavaScriptMapping,
) -> InferredDependencies:
    source: JSRuntimeSourceField = request.field_set.source
    if not nodejs_infer.imports:
        return InferredDependencies(())

    sources = await hydrate_sources(
        HydrateSourcesRequest(source, for_sources_types=[JSRuntimeSourceField]), **implicitly()
    )
    metadata = await _prepare_inference_metadata(request.field_set.address, source.file_path)

    import_strings, candidate_pkgs = await concurrently(
        parse_javascript_deps(NativeDependenciesRequest(sources.snapshot.digest, metadata)),
        map_candidate_node_packages(
            RequestNodePackagesCandidateMap(request.field_set.address), **implicitly()
        ),
    )
    imports = dict(
        zip(
            import_strings.imports,
            await concurrently(
                _determine_import_from_candidates(
                    candidates,
                    candidate_pkgs,
                    first_party_mapping,
                )
                for string, candidates in import_strings.imports.items()
            ),
        )
    )
    _handle_unowned_imports(
        request.field_set.address,
        nodejs_infer.unowned_dependency_behavior,
        frozenset(
            string
            for string, addresses in imports.items()
            if not addresses and not _is_node_builtin_module(string)
        ),
    )

    return InferredDependencies(itertools.chain.from_iterable(imports.values()))


@rule(desc="Find all JavaScript targets in project", level=LogLevel.DEBUG)
async def find_all_javascript_targets(all_targets: AllTargets) -> AllJavaScriptTargets:
    first_party = []
    third_party = []
    for tgt in all_targets:
        if tgt.has_field(JSRuntimeSourceField):
            first_party.append(tgt)
        if tgt.has_field(NodeThirdPartyPackageNameField):
            third_party.append(tgt)
    return AllJavaScriptTargets(
        tuple(sorted(first_party)), 
        tuple(sorted(third_party))
    )


@rule(desc="Creating map of first party JavaScript targets to file paths", level=LogLevel.DEBUG)
async def map_first_party_javascript_files(
    all_js_targets: AllJavaScriptTargets,
) -> FirstPartyJavaScriptMapping:
    """Map first-party JavaScript targets to their file paths."""
    logger.info(f"All JavaScript targets: {all_js_targets}")

    # Get stripped file paths for all first-party targets
    stripped_files = await concurrently(
        strip_file_name(StrippedFileNameRequest(tgt[JSRuntimeSourceField].file_path))
        for tgt in all_js_targets.first_party
    )

    logger.info(f"Stripped files: {stripped_files}")

    # Map each file to its owning address
    paths_to_addresses: DefaultDict[str, list[Address]] = defaultdict(list)

    for tgt, stripped_file in zip(all_js_targets.first_party, stripped_files):
        normalized_path = normalize_file_path(PurePath(stripped_file.value))
        paths_to_addresses[normalized_path].append(tgt.address)

    logger.info(f"Paths to addresses: {paths_to_addresses}")
    return FirstPartyJavaScriptMapping.create(paths_to_addresses)


def rules() -> Iterable[Rule | UnionRule]:
    return [
        *collect_rules(),
        *package_json.rules(),
        *tsconfig.rules(),
        UnionRule(InferDependenciesRequest, InferNodePackageDependenciesRequest),
        UnionRule(InferDependenciesRequest, InferJSDependenciesRequest),
    ]
