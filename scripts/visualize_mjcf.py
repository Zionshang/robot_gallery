#!/usr/bin/env python3

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize an MJCF XML with MuJoCo.")
    parser.add_argument("xml_path", help="Path to the MJCF XML file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    xml_path = Path(args.xml_path).expanduser().resolve()
    if not xml_path.is_file():
        raise FileNotFoundError(f"MJCF XML not found: {xml_path}")

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
