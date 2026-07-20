"""Vendored Lingbot Map MLX namespace.

Modified by CloudPoint to live under a private package namespace.  The alias is
installed only so the otherwise unchanged Apache-2.0 source snapshot can retain
its original absolute imports.  See PROVENANCE.md and LICENSE.txt here.
"""

from __future__ import annotations

import sys


_alias = "lingbot_map_mlx"
_this_module = sys.modules[__name__]
_existing = sys.modules.get(_alias)
if _existing is not None and _existing is not _this_module:
    raise ImportError("an external lingbot_map_mlx package is already imported")
sys.modules[_alias] = _this_module
