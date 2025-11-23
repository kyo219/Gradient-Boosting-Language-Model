"""
Tests for scripts (import verification).
"""

import pytest
import sys
from pathlib import Path

# Add scripts directory to path for import testing
scripts_dir = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))


def test_script_imports():
    """Test that all scripts can be imported without errors."""

    # List of scripts to test (without .py extension)
    scripts = [
        'build_vocab',
        'make_dataset',
        'train_gblm',
        'sample_gblm',
        'analyze_vocab_coverage',
        'quick_test_train'
    ]

    for script_name in scripts:
        # Try to import the script
        # This will fail if there are syntax errors or missing imports
        try:
            __import__(script_name)
            # Clean up the import
            if script_name in sys.modules:
                del sys.modules[script_name]
        except SystemExit:
            # Scripts might call sys.exit() or have argparse that exits
            # This is fine for import testing
            pass
        except Exception as e:
            # Check if it's just an argument parsing error (which is fine)
            if "arguments" not in str(e).lower() and "argparse" not in str(e).lower():
                pytest.fail(f"Failed to import {script_name}: {e}")


def test_script_structure():
    """Test that scripts have proper structure."""

    scripts = [
        'build_vocab.py',
        'make_dataset.py',
        'train_gblm.py',
        'sample_gblm.py',
        'analyze_vocab_coverage.py'
    ]

    for script_name in scripts:
        script_path = scripts_dir / script_name

        # Check that script exists
        assert script_path.exists(), f"Script {script_name} not found"

        # Read script content
        with open(script_path, 'r') as f:
            content = f.read()

        # Check for main guard
        assert 'if __name__ == "__main__"' in content, \
            f"Script {script_name} should have main guard"

        # Check for proper imports
        assert 'from src.' in content or 'import src.' in content, \
            f"Script {script_name} should use src.* imports"