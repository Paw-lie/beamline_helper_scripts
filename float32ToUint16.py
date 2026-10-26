"""Convert a raw float32 volume to uint16 out-of-memory, slice-chunk by slice-chunk.

Values in [vmin, vmax] are linearly mapped to [0, 65535]; values outside are clipped.
"""
import os
import numpy as np


def float32_to_uint16(in_path, out_path, shape, vmin, vmax, chunk_slices=16):
    """shape = (z, y, x); the raw file is read via np.memmap and written via a
    second memmap, so only `chunk_slices` slices are held in RAM at a time."""
    if os.path.exists(out_path):
        print(f"16-bit output already exists; skipping conversion: {out_path}")
        return

    if vmax <= vmin:
        raise ValueError("vmax must be larger than vmin")
    src = np.memmap(in_path, dtype=np.float32, mode="r", shape=tuple(shape))
    dst = np.memmap(out_path, dtype=np.uint16, mode="w+", shape=tuple(shape))
    scale = 65535.0 / (vmax - vmin)
    for z in range(0, shape[0], chunk_slices):
        chunk = src[z:z + chunk_slices].astype(np.float32)
        chunk -= vmin
        chunk *= scale
        np.rint(chunk, out=chunk)
        np.clip(chunk, 0, 65535, out=chunk)  # also handles values outside range
        dst[z:z + chunk_slices] = chunk.astype(np.uint16)
    dst.flush()
    del src, dst


if __name__ == "__main__":
    nx, ny, nz = 4358, 4358, 1184
    vmin = -5.0
    vmax = 5.0
    float32_to_uint16(
        "/mnt/M1/data3/2026_10_BM18_Gore/PROCESSED_DATA/X001_Dorner_Cab2_9p212um_132keV_13ms_HA2150px_5000p_0005/X001_Dorner_Cab2_9p212um_132keV_13ms_HA2150px_5000p_0005_0001/reconstructed_volumes/X001_Dorner_Cab2_9p212um_132keV_13ms_HA2150px_5000p_0005_0001_1_1_0000pag_db0100_vol.vol",
        f"/mnt/XNAS/data2/59_GORE/2026_10_BM18/02_cropped/tmp_vmin_{vmin}_vmax_{vmax}_uint16_{nx}x{ny}x{nz}.raw",
        shape=(nx, ny, nz),
        vmin=vmin,
        vmax=vmax,
    )
