#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `kg_recurit_restaurant_visitor_forecasting` package."""


import unittest
from click.testing import CliRunner

from kg_recurit_restaurant_visitor_forecasting import kg_recurit_restaurant_visitor_forecasting
from kg_recurit_restaurant_visitor_forecasting import cli


class TestKg_recurit_restaurant_visitor_forecasting(unittest.TestCase):
    """Tests for `kg_recurit_restaurant_visitor_forecasting` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_000_something(self):
        """Test something."""

    def test_command_line_interface(self):
        """Test the CLI."""
        runner = CliRunner()
        result = runner.invoke(cli.main)
        assert result.exit_code == 0
        assert 'kg_recurit_restaurant_visitor_forecasting.cli.main' in result.output
        help_result = runner.invoke(cli.main, ['--help'])
        assert help_result.exit_code == 0
        assert '--help  Show this message and exit.' in help_result.output
