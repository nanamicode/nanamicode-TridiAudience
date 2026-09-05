#!/usr/bin/env python3
from pathlib import Path
import sys

root = Path(sys.argv[1])
vr = root / "app/src/main/java/com/tridi/audience/VisionResult.java"
svc = root / "app/src/main/java/com/tridi/audience/AudienceService.java"

s = vr.read_text()
s = s.replace("static final int STRIDE = 24;", "static final int STRIDE = 28;")
if "static final int BODY_X = 24;" not in s:
    s = s.replace(
        "    static final int ATTENTION_GEOMETRY_VALID = 23;\n",
        "    static final int ATTENTION_GEOMETRY_VALID = 23;\n"
        "    static final int BODY_X = 24;\n"
        "    static final int BODY_Y = 25;\n"
        "    static final int BODY_WIDTH = 26;\n"
        "    static final int BODY_HEIGHT = 27;\n"
    )
vr.write_text(s)

s = svc.read_text()
old = """                    eventStore.enqueueReach(reach, trackId, capturedAtMs, eventJpeg,
                            frameWidth, frameHeight,
                            result[p + VisionResult.X], result[p + VisionResult.Y],
                            result[p + VisionResult.WIDTH], result[p + VisionResult.HEIGHT]);"""
new = """                    // Reach evidence must preserve the exact whole-body box used by
                    // the native tracker/preview, never the fresh SCRFD face box.
                    eventStore.enqueueReach(reach, trackId, capturedAtMs, eventJpeg,
                            frameWidth, frameHeight,
                            result[p + VisionResult.BODY_X], result[p + VisionResult.BODY_Y],
                            result[p + VisionResult.BODY_WIDTH], result[p + VisionResult.BODY_HEIGHT]);"""
if old in s:
    s = s.replace(old, new, 1)
elif "VisionResult.BODY_X" not in s:
    raise SystemExit("AudienceService reach enqueue anchor not found")
svc.write_text(s)

print("exact body protocol applied")
