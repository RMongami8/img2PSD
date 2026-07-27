import cv2


def ink_to_svg(ink_hi, out_size: int = 2048, ss: int = 2,
                smooth: int = 1, color: str = "#000000") -> str:
    contours, _ = cv2.findContours(ink_hi, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    min_area = 3.0 * ss * ss
    eps = (0.30 + 0.35 * smooth) * ss

    subpaths = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        approx = cv2.approxPolyDP(c, eps, True)
        if len(approx) < 3:
            continue
        pts = approx.reshape(-1, 2)
        coords = []
        for x, y in pts:
            fx = round(float(x) / ss, 2)
            fy = round(float(y) / ss, 2)
            coords.append(f"{fx} {fy}")
        d = "M " + " L ".join(coords) + " Z"
        subpaths.append(d)

    path_d = " ".join(subpaths)

    svg = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{out_size}" height="{out_size}" '
        f'viewBox="0 0 {out_size} {out_size}" shape-rendering="geometricPrecision">\n'
        f'<path fill="{color}" fill-rule="evenodd" d="{path_d}"/>\n'
        '</svg>\n'
    )
    return svg
