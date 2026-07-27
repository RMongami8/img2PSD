import struct


def _ascii_name(name: str) -> bytes:
    return "".join(c if ord(c) < 128 else '_' for c in name).encode('ascii')


def _pascal_name(name: str) -> bytes:
    b = _ascii_name(name)
    length = len(b)
    total = 1 + length
    pad = (-total) % 4
    return bytes([length]) + b + b"\x00" * pad


def _unicode_name_block(name: str) -> bytes:
    utf16 = name.encode('utf-16-be')
    content = struct.pack('>I', len(name)) + utf16
    if len(content) % 2 != 0:
        content += b"\x00"
    return b"8BIM" + b"luni" + struct.pack('>I', len(content)) + content


def _layer_record_bytes(name: str, height: int, width: int) -> bytes:
    parts = []
    parts.append(struct.pack('>iiii', 0, 0, height, width))
    parts.append(struct.pack('>H', 4))
    ch_len = 2 + height * width
    for cid in (-1, 0, 1, 2):
        parts.append(struct.pack('>hI', cid, ch_len))
    parts.append(b"8BIM")
    parts.append(b"norm")
    parts.append(struct.pack('>BBBB', 255, 0, 0, 0))

    pascal = _pascal_name(name)
    unicode_block = _unicode_name_block(name)
    extra_body = struct.pack('>II', 0, 0) + pascal + unicode_block
    if len(extra_body) % 2 != 0:
        extra_body += b"\x00"

    parts.append(struct.pack('>I', len(extra_body)))
    parts.append(extra_body)

    return b"".join(parts)


def write_psd(path: str, width: int, height: int, layers: list, composite_rgb) -> None:
    n = len(layers)

    layer_records = [_layer_record_bytes(l["name"], height, width) for l in layers]
    layer_records_total = sum(len(r) for r in layer_records)

    ch_size = 2 + height * width
    channel_data_total = n * 4 * ch_size

    layer_info_inner = 2 + layer_records_total + channel_data_total
    pad = layer_info_inner % 2
    layer_info_declared_len = layer_info_inner + pad

    lm_section_content_len = 4 + layer_info_declared_len + 4

    with open(path, "wb") as f:
        # (A) File Header
        f.write(b"8BPS")
        f.write(struct.pack('>H', 1))
        f.write(b"\x00" * 6)
        f.write(struct.pack('>H', 3))
        f.write(struct.pack('>I', height))
        f.write(struct.pack('>I', width))
        f.write(struct.pack('>H', 8))
        f.write(struct.pack('>H', 3))

        # (B) Color Mode Data Section
        f.write(struct.pack('>I', 0))

        # (C) Image Resources Section
        f.write(struct.pack('>I', 0))

        # (D) Layer and Mask Information Section
        f.write(struct.pack('>I', lm_section_content_len))
        f.write(struct.pack('>I', layer_info_declared_len))
        f.write(struct.pack('>h', n))

        for r in layer_records:
            f.write(r)

        for layer in layers:
            alpha = layer["alpha"].astype('uint8', copy=False)
            rgb = layer["rgb"].astype('uint8', copy=False)
            r_ch = rgb[:, :, 0]
            g_ch = rgb[:, :, 1]
            b_ch = rgb[:, :, 2]
            for plane in (alpha, r_ch, g_ch, b_ch):
                f.write(struct.pack('>H', 0))
                f.write(plane.tobytes())

        if pad:
            f.write(b"\x00")

        f.write(struct.pack('>I', 0))  # Global Layer Mask Info length

        # (E) Image Data Section
        f.write(struct.pack('>H', 0))
        f.write(composite_rgb[:, :, 0].tobytes())
        f.write(composite_rgb[:, :, 1].tobytes())
        f.write(composite_rgb[:, :, 2].tobytes())
