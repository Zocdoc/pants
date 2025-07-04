# Copyright 2014 Pants project contributors (see CONTRIBUTORS.md).
# Licensed under the Apache License, Version 2.0 (see LICENSE).

python_sources(sources=["*.py", "!pants_loader.py"])

python_sources(
    name="pants_loader",
    sources=["pants_loader.py"],
    dependencies=[":plugins"],
)

target(
    name="plugins",
    dependencies=[
        "src/python/pants/backend/awslambda/python",
        "src/python/pants/backend/build_files/fix/deprecations",
        "src/python/pants/backend/build_files/fmt/black",
        "src/python/pants/backend/build_files/fmt/buildifier",
        "src/python/pants/backend/build_files/fmt/ruff",
        "src/python/pants/backend/build_files/fmt/yapf",
        "src/python/pants/backend/codegen/protobuf/lint/buf",
        "src/python/pants/backend/codegen/protobuf/python",
        "src/python/pants/backend/codegen/thrift/apache/python",
        "src/python/pants/backend/docker",
        "src/python/pants/backend/docker/lint/hadolint",
        "src/python/pants/backend/experimental/adhoc",
        "src/python/pants/backend/experimental/bsp",
        "src/python/pants/backend/experimental/cc",
        "src/python/pants/backend/experimental/cc/lint/clangformat",
        "src/python/pants/backend/experimental/codegen/avro/java",
        "src/python/pants/backend/experimental/codegen/protobuf/go",
        "src/python/pants/backend/experimental/codegen/protobuf/java",
        "src/python/pants/backend/experimental/codegen/protobuf/scala",
        "src/python/pants/backend/experimental/codegen/python_format_string",
        "src/python/pants/backend/experimental/codegen/python_format_string/k8s",
        "src/python/pants/backend/experimental/codegen/thrift/apache/java",
        "src/python/pants/backend/experimental/codegen/thrift/scrooge/java",
        "src/python/pants/backend/experimental/codegen/thrift/scrooge/scala",
        "src/python/pants/backend/experimental/cue",
        "src/python/pants/backend/experimental/debian",
        "src/python/pants/backend/experimental/docker/lint/trivy",
        "src/python/pants/backend/experimental/docker/podman",
        "src/python/pants/backend/experimental/go",
        "src/python/pants/backend/experimental/go/debug_goals",
        "src/python/pants/backend/experimental/go/lint/golangci_lint",
        "src/python/pants/backend/experimental/go/lint/vet",
        "src/python/pants/backend/experimental/helm",
        "src/python/pants/backend/experimental/helm/check/kubeconform",
        "src/python/pants/backend/experimental/helm/lint/trivy",
        "src/python/pants/backend/experimental/java",
        "src/python/pants/backend/experimental/java/bsp",
        "src/python/pants/backend/experimental/java/debug_goals",
        "src/python/pants/backend/experimental/java/lint/google_java_format",
        "src/python/pants/backend/experimental/javascript",
        "src/python/pants/backend/experimental/javascript/lint/prettier",
        "src/python/pants/backend/experimental/k8s",
        "src/python/pants/backend/experimental/kotlin",
        "src/python/pants/backend/experimental/kotlin/debug_goals",
        "src/python/pants/backend/experimental/kotlin/lint/ktlint",
        "src/python/pants/backend/experimental/makeself",
        "src/python/pants/backend/experimental/nfpm",
        "src/python/pants/backend/experimental/openapi",
        "src/python/pants/backend/experimental/openapi/codegen/java",
        "src/python/pants/backend/experimental/openapi/codegen/python",
        "src/python/pants/backend/experimental/openapi/lint/openapi_format",
        "src/python/pants/backend/experimental/openapi/lint/spectral",
        "src/python/pants/backend/experimental/python",
        "src/python/pants/backend/experimental/python/framework/django",
        "src/python/pants/backend/experimental/python/framework/stevedore",
        "src/python/pants/backend/experimental/python/lint/add_trailing_comma",
        "src/python/pants/backend/experimental/python/lint/autoflake",
        "src/python/pants/backend/experimental/python/lint/pyupgrade",
        "src/python/pants/backend/experimental/python/lint/ruff",
        "src/python/pants/backend/experimental/python/lint/ruff/check",
        "src/python/pants/backend/experimental/python/lint/ruff/format",
        "src/python/pants/backend/experimental/python/packaging/pyoxidizer",
        "src/python/pants/backend/experimental/python/typecheck/pyright",
        "src/python/pants/backend/experimental/python/typecheck/pytype",
        "src/python/pants/backend/experimental/rust",
        "src/python/pants/backend/experimental/scala",
        "src/python/pants/backend/experimental/scala/bsp",
        "src/python/pants/backend/experimental/scala/debug_goals",
        "src/python/pants/backend/experimental/scala/lint/scalafix",
        "src/python/pants/backend/experimental/scala/lint/scalafmt",
        "src/python/pants/backend/experimental/sql",
        "src/python/pants/backend/experimental/sql/lint/sqlfluff",
        "src/python/pants/backend/experimental/swift",
        "src/python/pants/backend/experimental/terraform",
        "src/python/pants/backend/experimental/terraform/lint/tfsec",
        "src/python/pants/backend/experimental/terraform/lint/trivy",
        "src/python/pants/backend/experimental/tools/semgrep",
        "src/python/pants/backend/experimental/tools/trufflehog",
        "src/python/pants/backend/experimental/tools/workunit_logger",
        "src/python/pants/backend/experimental/tools/yamllint",
        "src/python/pants/backend/experimental/typescript",
        "src/python/pants/backend/experimental/visibility",
        "src/python/pants/backend/google_cloud_function/python",
        "src/python/pants/backend/plugin_development",
        "src/python/pants/backend/project_info",
        "src/python/pants/backend/python",
        "src/python/pants/backend/python/lint/autoflake",
        "src/python/pants/backend/python/lint/bandit",
        "src/python/pants/backend/python/lint/black",
        "src/python/pants/backend/python/lint/docformatter",
        "src/python/pants/backend/python/lint/flake8",
        "src/python/pants/backend/python/lint/isort",
        "src/python/pants/backend/python/lint/pydocstyle",
        "src/python/pants/backend/python/lint/pylint",
        "src/python/pants/backend/python/lint/pyupgrade",
        "src/python/pants/backend/python/lint/yapf",
        "src/python/pants/backend/python/mixed_interpreter_constraints",
        "src/python/pants/backend/python/providers/experimental/pyenv",
        "src/python/pants/backend/python/providers/experimental/pyenv/custom_install",
        "src/python/pants/backend/python/providers/experimental/python_build_standalone",
        "src/python/pants/backend/python/typecheck/mypy",
        "src/python/pants/backend/shell",
        "src/python/pants/backend/shell/lint/shellcheck",
        "src/python/pants/backend/shell/lint/shfmt",
        "src/python/pants/backend/tools/preamble",
        "src/python/pants/backend/tools/taplo",
        "src/python/pants/backend/url_handlers/s3",
        "src/python/pants/core",
    ],
)

resources(
    name="native_client",
    sources=["native_client"],
)

resources(
    name="sandboxer",
    sources=["sandboxer"],
)

python_tests(name="tests")
