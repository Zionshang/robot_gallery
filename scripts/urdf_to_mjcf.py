#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import os
import shutil
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

VISUAL_GROUP = "2"
COLLISION_GROUP = "3"


@dataclass(frozen=True)
class ConvertedMesh:
    staged_obj: Path
    final_obj: Path
    staged_texture: Path | None
    final_texture: Path | None
    rgba: tuple[float, float, float, float]


@dataclass(frozen=True)
class MimicJoint:
    joint: str
    source: str
    multiplier: float
    offset: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a URDF model to MJCF and generate a preview scene with "
            "MuJoCo's built-in compiler."
        )
    )
    parser.add_argument("urdf_path", help="Path to the input URDF file.")
    parser.add_argument(
        "mjcf_path",
        nargs="?",
        help="Output MJCF path (default: replace the URDF suffix with .xml).",
    )
    parser.add_argument(
        "--keep-visuals",
        action="store_true",
        help=(
            "Keep URDF <visual> elements and convert DAE meshes to textured OBJ "
            "assets automatically."
        ),
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Overwrite the model and generated scene if they already exist.",
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Do not open the generated scene in the MuJoCo viewer.",
    )
    parser.add_argument(
        "--root-pos",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="Translate the generated model's fixed root body by X Y Z meters.",
    )
    parser.add_argument(
        "--disable-self-collision",
        action="store_true",
        help=(
            "Disable contacts between all bodies in the converted model while "
            "preserving contacts with the scene and external models."
        ),
    )
    return parser.parse_args()


def load_urdf(path: Path) -> ET.ElementTree:
    try:
        tree = ET.parse(path)
    except ET.ParseError as exc:
        raise ValueError(f"Invalid URDF XML: {path}: {exc}") from exc

    if tree.getroot().tag != "robot":
        raise ValueError(f"Expected <robot> as the root element: {path}")
    return tree


def read_mimic_joints(tree: ET.ElementTree) -> list[MimicJoint]:
    """Read URDF mimic relationships before MuJoCo discards them."""
    joint_names = {joint.get("name", "") for joint in tree.getroot().findall("joint")}
    mimic_joints: list[MimicJoint] = []

    for joint in tree.getroot().findall("joint"):
        mimic = joint.find("mimic")
        if mimic is None:
            continue

        joint_name = joint.get("name", "")
        source_name = mimic.get("joint", "")
        if not joint_name:
            raise ValueError("A URDF joint with <mimic> is missing its name")
        if not source_name:
            raise ValueError(f"Mimic joint {joint_name!r} is missing its source joint")
        if source_name not in joint_names:
            raise ValueError(
                f"Mimic joint {joint_name!r} references unknown joint {source_name!r}"
            )
        if source_name == joint_name:
            raise ValueError(f"Mimic joint {joint_name!r} cannot reference itself")

        try:
            multiplier = float(mimic.get("multiplier", "1"))
            offset = float(mimic.get("offset", "0"))
        except ValueError as exc:
            raise ValueError(
                f"Mimic joint {joint_name!r} has an invalid multiplier or offset"
            ) from exc

        mimic_joints.append(
            MimicJoint(
                joint=joint_name,
                source=source_name,
                multiplier=multiplier,
                offset=offset,
            )
        )

    return mimic_joints


def enable_visuals(tree: ET.ElementTree) -> None:
    root = tree.getroot()
    mujoco_element = root.find("mujoco")
    if mujoco_element is None:
        mujoco_element = ET.SubElement(root, "mujoco")

    compiler = mujoco_element.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(mujoco_element, "compiler")
    compiler.set("discardvisual", "false")


def compiler_mesh_dir(tree: ET.ElementTree, urdf_dir: Path) -> Path | None:
    compiler = tree.getroot().find("./mujoco/compiler")
    if compiler is None or not compiler.get("meshdir"):
        return None
    return (urdf_dir / compiler.get("meshdir", "")).resolve()


def rewrite_asset_paths(
    mjcf_path: Path,
    urdf_dir: Path,
    mesh_dir: Path | None,
    path_remap: dict[Path, Path],
) -> None:
    """Make paths emitted relative to the URDF work from the MJCF directory."""
    tree = ET.parse(mjcf_path)
    changed = False

    for element in tree.getroot().findall("./asset/*[@file]"):
        file_value = element.get("file")
        if not file_value:
            continue

        file_path = Path(file_value)
        if file_path.is_absolute():
            source_path = path_remap.get(file_path.resolve(), file_path.resolve())
            if not source_path.is_file():
                source_path = None
        else:
            candidates = [urdf_dir / file_path]
            if mesh_dir is not None and element.tag in {"mesh", "skin"}:
                candidates.append(mesh_dir / file_path)
            source_path = next(
                (
                    candidate.resolve()
                    for candidate in candidates
                    if candidate.is_file()
                ),
                None,
            )
            if source_path is not None:
                source_path = path_remap.get(source_path, source_path)

        if source_path is None:
            continue

        relative_path = os.path.relpath(source_path, start=mjcf_path.parent)
        normalized_path = Path(relative_path).as_posix()
        if normalized_path != file_value:
            element.set("file", normalized_path)
            changed = True

    if changed:
        ET.indent(tree, space="  ")
        tree.write(mjcf_path, encoding="utf-8", xml_declaration=False)


def resolve_mesh_path(
    filename: str,
    urdf_dir: Path,
    mesh_dir: Path | None,
) -> Path:
    mesh_path = Path(filename)
    candidates = [mesh_path] if mesh_path.is_absolute() else [urdf_dir / mesh_path]
    if not mesh_path.is_absolute() and mesh_dir is not None:
        candidates.append(mesh_dir / mesh_path)

    resolved_path = next(
        (candidate.resolve() for candidate in candidates if candidate.is_file()),
        None,
    )
    if resolved_path is None:
        raise FileNotFoundError(f"Mesh referenced by URDF was not found: {filename}")
    return resolved_path


def material_rgba(mesh: object) -> tuple[float, float, float, float]:
    material = getattr(getattr(mesh, "visual", None), "material", None)
    color = getattr(material, "main_color", None)
    if color is None or len(color) < 3:
        return (1.0, 1.0, 1.0, 1.0)

    values = [float(channel) for channel in color[:4]]
    if len(values) == 3:
        values.append(255.0)
    if max(values) > 1.0:
        values = [channel / 255.0 for channel in values]
    return (values[0], values[1], values[2], values[3])


def has_diffuse_texture(mesh: object) -> bool:
    material = getattr(getattr(mesh, "visual", None), "material", None)
    return (
        getattr(material, "baseColorTexture", None) is not None
        or getattr(material, "image", None) is not None
    )


def merge_with_color_texture(trimesh: object, meshes: list[object]) -> object:
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Color-preserving DAE conversion requires numpy and Pillow. Install "
            "them with: pip install 'trimesh[easy]'"
        ) from exc

    colors = [
        tuple(round(channel * 255) for channel in material_rgba(mesh))
        for mesh in meshes
    ]
    texture = Image.new("RGBA", (len(colors), 1))
    texture.putdata(colors)

    uv_parts = []
    for index, mesh in enumerate(meshes):
        u = (index + 0.5) / len(meshes)
        uv_parts.append(np.tile((u, 0.5), (len(mesh.vertices), 1)))

    combined = trimesh.util.concatenate(meshes)
    combined.visual = trimesh.visual.TextureVisuals(
        uv=np.concatenate(uv_parts),
        image=texture,
    )
    return combined


def write_obj_component(
    mesh: object,
    output_dir: Path,
    obj_name: str,
) -> tuple[Path, Path | None]:
    try:
        from trimesh.exchange.obj import export_obj
    except ImportError as exc:
        raise RuntimeError(
            "DAE conversion requires trimesh. Install it with: "
            "pip install 'trimesh[easy]'"
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    obj_data, auxiliary_files = export_obj(
        mesh,
        include_texture=True,
        return_texture=True,
    )
    obj_path = output_dir / obj_name
    obj_path.write_text(obj_data, encoding="utf-8")

    texture_path: Path | None = None
    diffuse_texture_name: str | None = None
    for filename, contents in auxiliary_files.items():
        destination = output_dir / Path(filename).name
        if isinstance(contents, bytes):
            destination.write_bytes(contents)
        else:
            destination.write_text(contents, encoding="utf-8")

        if destination.suffix.lower() == ".mtl":
            mtl_text = (
                contents.decode("utf-8") if isinstance(contents, bytes) else contents
            )
            for line in mtl_text.splitlines():
                if line.lstrip().lower().startswith("map_kd "):
                    diffuse_texture_name = line.split(maxsplit=1)[1].strip()
                    break

    if diffuse_texture_name:
        candidate = output_dir / Path(diffuse_texture_name).name
        if candidate.is_file():
            texture_path = candidate
    return obj_path, texture_path


def convert_dae_mesh(
    dae_path: Path,
    staging_assets: Path,
    final_assets: Path,
) -> list[ConvertedMesh]:
    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError(
            "DAE conversion requires trimesh. Install it with: "
            "pip install 'trimesh[easy]'"
        ) from exc

    try:
        scene = trimesh.load(dae_path, force="scene")
    except ImportError as exc:
        raise RuntimeError(
            "DAE conversion requires trimesh's COLLADA dependencies. Install "
            "them with: pip install 'trimesh[easy]'"
        ) from exc
    if not isinstance(scene, trimesh.Scene) or not scene.geometry:
        raise ValueError(f"DAE contains no mesh geometry: {dae_path}")

    digest = hashlib.sha256(dae_path.read_bytes()).hexdigest()[:8]
    source_dir_name = f"{dae_path.stem}_{digest}"
    converted: list[ConvertedMesh] = []

    nodes = list(scene.graph.nodes_geometry)
    if not nodes:
        nodes = list(scene.geometry)

    component_meshes = []
    for node_name in nodes:
        if node_name in scene.graph.nodes_geometry:
            transform, geometry_name = scene.graph[node_name]
        else:
            transform = None
            geometry_name = node_name

        mesh = scene.geometry[geometry_name].copy()
        if transform is not None:
            mesh.apply_transform(transform)
        component_meshes.append(mesh)

    preserve_textures = any(has_diffuse_texture(mesh) for mesh in component_meshes)
    if not preserve_textures:
        component_meshes = [merge_with_color_texture(trimesh, component_meshes)]

    for index, mesh in enumerate(component_meshes):
        relative_dir = Path(source_dir_name) / f"part_{index:03d}"
        staged_dir = staging_assets / relative_dir
        final_dir = final_assets / relative_dir
        staged_obj, staged_texture = write_obj_component(
            mesh,
            staged_dir,
            obj_name=f"{source_dir_name}_part_{index:03d}.obj",
        )
        converted.append(
            ConvertedMesh(
                staged_obj=staged_obj.resolve(),
                final_obj=(final_dir / staged_obj.name).resolve(),
                staged_texture=(
                    staged_texture.resolve() if staged_texture is not None else None
                ),
                final_texture=(
                    (final_dir / staged_texture.name).resolve()
                    if staged_texture is not None
                    else None
                ),
                rgba=material_rgba(mesh),
            )
        )

    return converted


def convert_dae_visuals(
    tree: ET.ElementTree,
    urdf_dir: Path,
    mesh_dir: Path | None,
    staging_assets: Path,
    final_assets: Path,
) -> list[ConvertedMesh]:
    converted_by_source: dict[Path, list[ConvertedMesh]] = {}
    all_converted: list[ConvertedMesh] = []

    for link in tree.getroot().findall("link"):
        link_name = link.get("name", "link")
        for collision_index, collision in enumerate(link.findall("collision")):
            if not collision.get("name"):
                collision.set("name", f"{link_name}_collision_{collision_index:03d}")

        for visual_index, visual in enumerate(list(link.findall("visual"))):
            mesh_element = visual.find("./geometry/mesh")
            if mesh_element is None or not mesh_element.get("filename"):
                if not visual.get("name"):
                    visual.set("name", f"{link_name}_visual_{visual_index:03d}")
                continue

            filename = mesh_element.get("filename", "")
            if Path(filename).suffix.lower() != ".dae":
                if not visual.get("name"):
                    visual.set("name", f"{link_name}_visual_{visual_index:03d}")
                continue

            dae_path = resolve_mesh_path(
                filename,
                urdf_dir=urdf_dir,
                mesh_dir=mesh_dir,
            )

            components = converted_by_source.get(dae_path)
            if components is None:
                print(f"Converting DAE: {dae_path}")
                components = convert_dae_mesh(
                    dae_path,
                    staging_assets=staging_assets,
                    final_assets=final_assets,
                )
                converted_by_source[dae_path] = components
                all_converted.extend(components)

            insertion_index = list(link).index(visual)
            link.remove(visual)
            for component_index, component in enumerate(components):
                component_visual = copy.deepcopy(visual)
                component_visual.set(
                    "name",
                    (f"{link_name}_visual_{visual_index:03d}_{component_index:03d}"),
                )
                component_mesh = component_visual.find("./geometry/mesh")
                if component_mesh is None:
                    raise RuntimeError("Converted visual is missing its mesh element")
                component_mesh.set("filename", str(component.staged_obj))
                link.insert(insertion_index + component_index, component_visual)

    return all_converted


def conversion_source(
    tree: ET.ElementTree,
    urdf_path: Path,
    keep_visuals: bool,
    mesh_dir: Path | None,
    staging_assets: Path | None,
    final_assets: Path,
) -> tuple[Path, Path | None, list[ConvertedMesh]]:
    if not keep_visuals:
        return urdf_path, None, []
    if staging_assets is None:
        raise RuntimeError("A staging directory is required to convert DAE assets")

    enable_visuals(tree)
    converted = convert_dae_visuals(
        tree,
        urdf_dir=urdf_path.parent,
        mesh_dir=mesh_dir,
        staging_assets=staging_assets,
        final_assets=final_assets,
    )
    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{urdf_path.stem}_",
        suffix=".urdf",
        dir=urdf_path.parent,
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
        tree.write(temporary_file, encoding="utf-8", xml_declaration=True)
    return temporary_path, temporary_path, converted


def add_converted_materials(
    mjcf_path: Path,
    converted_meshes: list[ConvertedMesh],
) -> None:
    if not converted_meshes:
        return

    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    asset = root.find("asset")
    if asset is None:
        asset = ET.Element("asset")
        root.insert(0, asset)

    mesh_by_file: dict[Path, ET.Element] = {}
    for mesh_element in asset.findall("mesh[@file]"):
        file_value = mesh_element.get("file")
        if file_value:
            mesh_by_file[Path(file_value).resolve()] = mesh_element

    for index, converted in enumerate(converted_meshes):
        mesh_element = mesh_by_file.get(converted.staged_obj)
        if mesh_element is None or not mesh_element.get("name"):
            raise RuntimeError(
                f"Converted OBJ was not emitted in the MJCF: {converted.staged_obj}"
            )

        mesh_name = mesh_element.get("name", "")
        geoms = root.findall(f".//geom[@mesh='{mesh_name}']")
        rgba = " ".join(f"{value:.8g}" for value in converted.rgba)
        if converted.final_texture is None:
            for geom in geoms:
                geom.set("rgba", rgba)
            continue

        texture_name = f"dae_texture_{index:03d}"
        material_name = f"dae_material_{index:03d}"
        asset.insert(
            0,
            ET.Element(
                "texture",
                {
                    "name": texture_name,
                    "type": "2d",
                    "file": str(converted.final_texture),
                },
            ),
        )
        asset.insert(
            1,
            ET.Element(
                "material",
                {
                    "name": material_name,
                    "texture": texture_name,
                    "rgba": rgba,
                },
            ),
        )
        for geom in geoms:
            geom.attrib.pop("rgba", None)
            geom.set("material", material_name)

    ET.indent(tree, space="  ")
    tree.write(mjcf_path, encoding="utf-8", xml_declaration=False)


def assign_geom_groups(mjcf_path: Path) -> None:
    """Put URDF visual and collision geoms into explicit MuJoCo groups."""
    tree = ET.parse(mjcf_path)
    for geom in tree.getroot().findall(".//geom"):
        # MuJoCo's URDF importer marks visual geoms as group 1 and disables
        # their contact bits. Collision geoms use group 0 by default.
        is_visual = (
            geom.get("group") == "1"
            and geom.get("contype") == "0"
            and geom.get("conaffinity") == "0"
        )
        geom.set("group", VISUAL_GROUP if is_visual else COLLISION_GROUP)

    ET.indent(tree, space="  ")
    tree.write(mjcf_path, encoding="utf-8", xml_declaration=False)


def add_mimic_constraints(
    mjcf_path: Path,
    mimic_joints: list[MimicJoint],
) -> None:
    """Convert URDF mimic joints to MuJoCo joint equality constraints."""
    if not mimic_joints:
        return

    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    mjcf_joint_names = {joint.get("name", "") for joint in root.findall(".//joint")}

    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")

    constrained_joints = {
        constraint.get("joint1", "") for constraint in equality.findall("joint")
    }
    for mimic in mimic_joints:
        missing = {
            name for name in (mimic.joint, mimic.source) if name not in mjcf_joint_names
        }
        if missing:
            missing_names = ", ".join(sorted(repr(name) for name in missing))
            raise RuntimeError(
                f"Cannot create mimic constraint for {mimic.joint!r}; generated "
                f"MJCF is missing joint(s): {missing_names}"
            )
        if mimic.joint in constrained_joints:
            raise RuntimeError(
                f"Generated MJCF already constrains mimic joint {mimic.joint!r}"
            )

        ET.SubElement(
            equality,
            "joint",
            {
                "joint1": mimic.joint,
                "joint2": mimic.source,
                "polycoef": (f"{mimic.offset:.8g} {mimic.multiplier:.8g} 0 0 0"),
            },
        )
        constrained_joints.add(mimic.joint)

    ET.indent(tree, space="  ")
    tree.write(mjcf_path, encoding="utf-8", xml_declaration=False)


def configure_root_body(
    mjcf_path: Path,
    root_pos: tuple[float, float, float] | None,
    disable_self_collision: bool,
) -> None:
    if root_pos is None and not disable_self_collision:
        return

    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("Generated MJCF has no <worldbody> element")

    existing_body_names = {
        body.get("name") for body in worldbody.findall(".//body") if body.get("name")
    }
    root_name = "converted_root"
    suffix = 1
    while root_name in existing_body_names:
        root_name = f"converted_root_{suffix}"
        suffix += 1

    position = root_pos or (0.0, 0.0, 0.0)
    root_body = ET.Element(
        "body",
        {
            "name": root_name,
            "pos": " ".join(f"{value:.8g}" for value in position),
        },
    )
    for element in list(worldbody):
        worldbody.remove(element)
        root_body.append(element)
    worldbody.append(root_body)

    if disable_self_collision:
        bodies = [root_body, *root_body.findall(".//body")]
        body_names: list[str] = []
        used_names = set(existing_body_names)
        for index, body in enumerate(bodies):
            body_name = body.get("name")
            if not body_name:
                body_name = f"converted_body_{index:03d}"
                name_suffix = 1
                while body_name in used_names:
                    body_name = f"converted_body_{index:03d}_{name_suffix}"
                    name_suffix += 1
                body.set("name", body_name)
            used_names.add(body_name)
            body_names.append(body_name)

        contact = root.find("contact")
        if contact is None:
            contact = ET.SubElement(root, "contact")
        existing_excludes = {
            frozenset((exclude.get("body1", ""), exclude.get("body2", "")))
            for exclude in contact.findall("exclude")
        }
        for body1, body2 in itertools.combinations(body_names, 2):
            pair = frozenset((body1, body2))
            if pair in existing_excludes:
                continue
            ET.SubElement(contact, "exclude", {"body1": body1, "body2": body2})
            existing_excludes.add(pair)

    ET.indent(tree, space="  ")
    tree.write(mjcf_path, encoding="utf-8", xml_declaration=False)


def write_scene(
    scene_path: Path,
    model_file: str,
    model_name: str,
    center: tuple[float, float, float],
    extent: float,
) -> None:
    root = ET.Element("mujoco", {"model": f"{model_name} scene"})
    ET.SubElement(root, "include", {"file": model_file})
    ET.SubElement(
        root,
        "statistic",
        {
            "center": " ".join(f"{value:.8g}" for value in center),
            "extent": f"{max(extent, 0.5):.8g}",
        },
    )

    visual = ET.SubElement(root, "visual")
    ET.SubElement(
        visual,
        "headlight",
        {
            "diffuse": "0.6 0.6 0.6",
            "ambient": "0.3 0.3 0.3",
            "specular": "0 0 0",
        },
    )
    ET.SubElement(visual, "rgba", {"haze": "0.15 0.25 0.35 1"})
    ET.SubElement(visual, "global", {"azimuth": "140", "elevation": "-20"})

    asset = ET.SubElement(root, "asset")
    ET.SubElement(
        asset,
        "texture",
        {
            "type": "skybox",
            "builtin": "gradient",
            "rgb1": "0.3 0.5 0.7",
            "rgb2": "0 0 0",
            "width": "512",
            "height": "3072",
        },
    )
    ET.SubElement(
        asset,
        "texture",
        {
            "type": "2d",
            "name": "scene_groundplane_texture",
            "builtin": "checker",
            "mark": "edge",
            "rgb1": "0.2 0.3 0.4",
            "rgb2": "0.1 0.2 0.3",
            "markrgb": "0.8 0.8 0.8",
            "width": "300",
            "height": "300",
        },
    )
    ET.SubElement(
        asset,
        "material",
        {
            "name": "scene_groundplane_material",
            "texture": "scene_groundplane_texture",
            "texuniform": "true",
            "texrepeat": "5 5",
            "reflectance": "0.2",
        },
    )

    worldbody = ET.SubElement(root, "worldbody")
    ET.SubElement(
        worldbody,
        "light",
        {
            "pos": "0 0 1.5",
            "dir": "0 0 -1",
            "directional": "true",
        },
    )
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "scene_floor",
            "size": "0 0 0.05",
            "type": "plane",
            "group": "0",
            "material": "scene_groundplane_material",
        },
    )

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(scene_path, encoding="utf-8", xml_declaration=False)


def open_in_viewer(xml_path: Path) -> None:
    import time

    import mujoco
    from mujoco import viewer

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    print(f"Opening {xml_path} in the MuJoCo viewer...")
    with viewer.launch_passive(model, data) as active_viewer:
        while active_viewer.is_running():
            mujoco.mj_step(model, data)
            active_viewer.sync()
            time.sleep(model.opt.timestep)


def main() -> None:
    args = parse_args()

    urdf_path = Path(args.urdf_path).expanduser().resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    mjcf_path = (
        Path(args.mjcf_path).expanduser().resolve()
        if args.mjcf_path
        else urdf_path.with_suffix(".xml")
    )
    if mjcf_path == urdf_path:
        raise ValueError("The MJCF output path must differ from the input URDF path.")
    scene_path = mjcf_path.with_name("scene.xml")
    if scene_path == mjcf_path:
        raise ValueError(
            "The robot MJCF cannot be named scene.xml because that name is reserved "
            "for the generated scene."
        )
    if mjcf_path.exists() and not args.force:
        raise FileExistsError(
            f"Output already exists: {mjcf_path} (use --force to overwrite it)"
        )
    if scene_path.exists() and not args.force:
        raise FileExistsError(
            f"Scene already exists: {scene_path} (use --force to overwrite it)"
        )

    try:
        import mujoco
    except ImportError as exc:
        raise SystemExit(
            "The 'mujoco' package is required. Install it with: pip install mujoco"
        ) from exc

    mjcf_path.parent.mkdir(parents=True, exist_ok=True)
    tree = load_urdf(urdf_path)
    mimic_joints = read_mimic_joints(tree)
    mesh_dir = compiler_mesh_dir(tree, urdf_path.parent)
    final_assets = mjcf_path.parent / f"{mjcf_path.stem}_assets"
    temporary_assets = (
        tempfile.TemporaryDirectory(
            prefix=f".{mjcf_path.stem}_assets_",
            dir=mjcf_path.parent,
        )
        if args.keep_visuals
        else None
    )
    staging_assets = (
        Path(temporary_assets.name) if temporary_assets is not None else None
    )

    temporary_source: Path | None = None
    temporary_output: Path | None = None
    temporary_scene: Path | None = None
    try:
        source_path, temporary_source, converted_meshes = conversion_source(
            tree,
            urdf_path,
            args.keep_visuals,
            mesh_dir=mesh_dir,
            staging_assets=staging_assets,
            final_assets=final_assets,
        )
        with tempfile.NamedTemporaryFile(
            prefix=f".{mjcf_path.stem}_",
            suffix=".xml",
            dir=mjcf_path.parent,
            delete=False,
        ) as temporary_file:
            temporary_output = Path(temporary_file.name)

        model = mujoco.MjModel.from_xml_path(str(source_path))
        mujoco.mj_saveLastXML(str(temporary_output), model)
        if converted_meshes and staging_assets is not None:
            shutil.copytree(staging_assets, final_assets, dirs_exist_ok=True)
        add_converted_materials(temporary_output, converted_meshes)
        assign_geom_groups(temporary_output)
        add_mimic_constraints(temporary_output, mimic_joints)
        configure_root_body(
            temporary_output,
            root_pos=(tuple(args.root_pos) if args.root_pos is not None else None),
            disable_self_collision=args.disable_self_collision,
        )
        path_remap = {
            converted.staged_obj: converted.final_obj for converted in converted_meshes
        }
        path_remap.update(
            {
                converted.staged_texture: converted.final_texture
                for converted in converted_meshes
                if converted.staged_texture is not None
                and converted.final_texture is not None
            }
        )
        rewrite_asset_paths(
            temporary_output,
            urdf_dir=urdf_path.parent,
            mesh_dir=mesh_dir,
            path_remap=path_remap,
        )
        converted_model = mujoco.MjModel.from_xml_path(str(temporary_output))

        with tempfile.NamedTemporaryFile(
            prefix=".scene_",
            suffix=".xml",
            dir=scene_path.parent,
            delete=False,
        ) as temporary_file:
            temporary_scene = Path(temporary_file.name)

        center = tuple(float(value) for value in converted_model.stat.center)
        write_scene(
            temporary_scene,
            model_file=temporary_output.name,
            model_name=mjcf_path.stem,
            center=center,
            extent=float(converted_model.stat.extent),
        )
        mujoco.MjModel.from_xml_path(str(temporary_scene))
        write_scene(
            temporary_scene,
            model_file=mjcf_path.name,
            model_name=mjcf_path.stem,
            center=center,
            extent=float(converted_model.stat.extent),
        )

        temporary_output.replace(mjcf_path)
        temporary_scene.replace(scene_path)
    except Exception as exc:
        message = f"Failed to convert {urdf_path}: {exc}"
        raise RuntimeError(message) from exc
    finally:
        if temporary_source is not None:
            temporary_source.unlink(missing_ok=True)
        if temporary_output is not None:
            temporary_output.unlink(missing_ok=True)
        if temporary_scene is not None:
            temporary_scene.unlink(missing_ok=True)
        if temporary_assets is not None:
            temporary_assets.cleanup()

    print(f"Wrote {mjcf_path}")
    print(f"Wrote {scene_path}")
    if not args.no_viewer:
        open_in_viewer(scene_path)


if __name__ == "__main__":
    main()
