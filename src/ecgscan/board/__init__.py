"""Capture board with ArUco markers (SD-065).

The ECG page lies inside the board's zone; the markers around it are found in the
photo and give a homography from the photo to board millimetres. That replaces the
grid-texture rectification when a board is present:

* the scale is exact and independent of the paper grid (faint, grey or no grid);
* the photo's own resolution (px per board mm) is known everywhere in the zone, so
  the 8 px/mm floor is checked on the worst part of the page, not on an average;
* the scan sees only the zone: markers, rulers and the table never reach the reader.

No board in the photo: the grid path is unchanged.
"""

from ecgscan.board.compose import Composite, compose_zone
from ecgscan.board.detect import BoardFit, detect_board, warp_zone
from ecgscan.board.render import render_board, render_sheets
from ecgscan.board.spec import BoardSpec, load_board, load_boards

__all__ = ["BoardFit", "BoardSpec", "Composite", "compose_zone", "detect_board", "load_board", "load_boards", "render_board", "render_sheets", "warp_zone"]
