# Copyright 2015 Pants project contributors (see CONTRIBUTORS.md).
# Licensed under the Apache License, Version 2.0 (see LICENSE).
from __future__ import annotations

from dataclasses import replace

from pants.help.help_formatter import HelpFormatter
from pants.help.help_info_extracter import HelpInfoExtracter, OptionHelpInfo
from pants.option.global_options import GlobalOptions
from pants.option.native_options import NativeOptionParser
from pants.option.ranked_value import Rank, RankedValue
from pants.option.registrar import OptionRegistrar, OptionValueHistory


class TestOptionHelpFormatter:
    @staticmethod
    def _format_for_single_option(**kwargs):
        ohi = OptionHelpInfo(
            display_args=("--foo",),
            comma_separated_display_args="--foo",
            scoped_cmd_line_args=("--foo",),
            unscoped_cmd_line_args=("--foo",),
            env_var="PANTS_FOO",
            config_key="foo",
            target_field_name=None,
            typ=bool,
            default=None,
            help="help for foo",
            deprecation_active=False,
            deprecated_message=None,
            removal_version=None,
            removal_hint=None,
            choices=None,
            comma_separated_choices=None,
            value_history=OptionValueHistory((RankedValue(Rank.HARDCODED, None),)),
            fromfile=False,
        )
        ohi = replace(ohi, **kwargs)
        lines = HelpFormatter(
            show_advanced=False, show_deprecated=False, color=False
        ).format_option(ohi)
        choices = kwargs.get("choices")
        assert len(lines) == 7 if choices else 6
        if choices:
            assert f"one of: [{', '.join(choices)}]" == lines[3].strip()
        assert "help for foo" in lines[6 if choices else 5]
        return lines[4] if choices else lines[3]

    def test_format_help(self) -> None:
        default_line = self._format_for_single_option(default="MYDEFAULT")
        assert default_line.lstrip() == "default: MYDEFAULT"

    def test_format_help_choices(self) -> None:
        default_line = self._format_for_single_option(
            typ=str, default="kiwi", choices=["apple", "banana", "kiwi"]
        )
        assert default_line.lstrip() == "default: kiwi"

    @classmethod
    def _get_registrar_and_parser(cls) -> tuple[OptionRegistrar, NativeOptionParser]:
        return OptionRegistrar(
            scope=GlobalOptions.options_scope,
        ), NativeOptionParser(
            args=[],
            env={},
            config_sources=[],
            allow_pantsrc=False,
            include_derivation=True,
            known_scopes_to_flags={},
            known_goals=[],
        )

    @classmethod
    def _format_for_global_scope(
        cls, show_advanced: bool, show_deprecated: bool, args: list[str], kwargs
    ) -> list[str]:
        registrar, native_parser = cls._get_registrar_and_parser()
        registrar.register(*args, **kwargs)
        return cls._format_for_global_scope_with_parser(
            registrar, native_parser, show_advanced=show_advanced, show_deprecated=show_deprecated
        )

    @classmethod
    def _format_for_global_scope_with_parser(
        cls,
        registrar: OptionRegistrar,
        native_parser: NativeOptionParser,
        show_advanced: bool,
        show_deprecated: bool,
    ) -> list[str]:
        oshi = HelpInfoExtracter("").get_option_scope_help_info(
            "", registrar, native_parser, False, "help.test"
        )
        return HelpFormatter(
            show_advanced=show_advanced, show_deprecated=show_deprecated, color=False
        ).format_options(oshi)

    def test_suppress_advanced(self) -> None:
        registrar, native_parser = self._get_registrar_and_parser()
        registrar.register("--foo", advanced=True)
        # must have a non advanced option to be able to supress showing advanced options.
        registrar.register("--jerry", advanced=False)
        lines = self._format_for_global_scope_with_parser(registrar, native_parser, False, False)
        assert len(lines) == 15
        assert not any("--foo" in line for line in lines)
        lines = self._format_for_global_scope_with_parser(registrar, native_parser, True, False)
        assert len(lines) == 24

    def test_suppress_deprecated(self) -> None:
        args = ["--foo"]
        kwargs = {"removal_version": "33.44.55.dev0"}
        lines = self._format_for_global_scope(False, False, args, kwargs)
        assert len(lines) == 8
        assert not any("--foo" in line for line in lines)
        lines = self._format_for_global_scope(True, True, args, kwargs)
        assert len(lines) == 23

    def test_provider_info(self) -> None:
        lines = self._format_for_global_scope(False, False, ["--foo"], {})
        assert len(lines) == 14
        assert "Activated by help.test" in lines
