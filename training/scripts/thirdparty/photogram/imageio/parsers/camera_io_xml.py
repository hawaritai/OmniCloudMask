from __future__ import annotations
import xml.etree.ElementTree as ET
from ...entities.camera_io import CameraIO, RadialTangentialDistortion

def parse_camera_io_xml(path: str) -> CameraIO:
    """
    Expects XML like:

    <CAMERA>
        <MODEL>UltraCam Eagle 4.1</MODEL>
        <SERIAL>436S12536X813111-f120</SERIAL>
        <IMAGE_SIZE WIDTH="9370" HEIGHT="6020" />
        <CHIP_SIZE WIDTH="105.694" HEIGHT="67.906" UNIT="mm" />
        <PIXEL_SIZE X="4.0" Y="4.0" UNIT="µm" />
        <FOCAL_LENGTH VALUE="122.700" UNIT="mm" PRECISION="0.002" />
        <PRINCIPAL_POINT_OFFSET X="0.0752" Y="0.0000" UNIT="mm" PRECISION="0.002" />
        <DISTORTION_MODEL type="Residual">
            <MAX_REMAINING_DISTORTION VALUE="0.002" UNIT="mm" />
        </DISTORTION_MODEL>
    </CAMERA>
    """

    root = ET.parse(path).getroot()

    # Identification
    model_name = root.findtext("MODEL", default="Camera")
    serial = root.findtext("SERIAL", default="")

    # Image size in pixels
    image_size_el = root.find("IMAGE_SIZE")
    width_px = int(image_size_el.get("WIDTH"))
    height_px = int(image_size_el.get("HEIGHT"))

    # Chip size in mm
    chip_size_el = root.find("CHIP_SIZE")
    chip_w_mm = float(chip_size_el.get("WIDTH"))
    chip_h_mm = float(chip_size_el.get("HEIGHT"))

    # Pixel size in mm (convert to µm)
    pixel_size_el = root.find("PIXEL_SIZE")
    if pixel_size_el is not None:
        sx_um = float(pixel_size_el.get("X"))  
        sy_um = float(pixel_size_el.get("Y"))
    else:
        # fallback: compute from chip / image size
        sx_um = (chip_w_mm / width_px) * 1000.0
        sy_um = (chip_h_mm / height_px) * 1000.0

    # Focal length
    fl_el = root.find("FOCAL_LENGTH")
    focal_length_mm = float(fl_el.get("VALUE"))

    # Principal point (in mm, need to convert to pixels)
    pp_offset_el = root.find("PRINCIPAL_POINT_OFFSET")
    pp_offset_x_mm = float(pp_offset_el.get("X"))
    pp_offset_y_mm = float(pp_offset_el.get("Y"))
    pp_x_px = pp_offset_x_mm / (sx_um / 1000.0)  # µm->mm
    pp_y_px = pp_offset_y_mm / (sy_um / 1000.0)
    
    pp_x_px = (width_px - 1) / 2.0 + pp_offset_x_mm / (sx_um / 1000.0)  # µm->mm
    pp_y_px = (height_px - 1) / 2.0 - pp_offset_y_mm / (sy_um / 1000.0)  # note: minus because image y points down

    # Distortion (dummy for now, since only residual <0.002 mm is given)
    dist_el = root.find("DISTORTION_MODEL/MAX_REMAINING_DISTORTION")
    max_resid = float(dist_el.get("VALUE")) if dist_el is not None else 0.0
    # distortion = RadialTangentialDistortion(max_residual_mm=max_resid)

    return CameraIO.from_any(
        model_name=model_name,
        serial=serial,
        focal_length_mm=focal_length_mm,
        pixel_size_um=(sx_um, sy_um),
        principal_point_px=(pp_x_px, pp_y_px),
        image_size_px=(width_px, height_px),
        # distortion=distortion
    )
