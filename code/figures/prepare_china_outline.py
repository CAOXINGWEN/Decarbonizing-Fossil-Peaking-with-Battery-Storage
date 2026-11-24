"""Dissolve the existing provincial geometry into exterior-only China outlines.

Run with the layout environment (Shapely installed). No boundary vertices are
invented, buffered or simplified. The figure itself does not require Shapely.
"""
import hashlib
import json
from pathlib import Path

from shapely.geometry import Point, box, shape
from shapely.ops import unary_union
from shapely.validation import explain_validity

HERE = Path(__file__).resolve().parent
V86 = HERE.parents[1]
ROOT = V86.parents[1]
SOURCE = ROOT / 'dataset/中国_省.geojson'
OUT = V86 / 'figures/assets/china_outer_outline.geojson'


def main():
    raw = SOURCE.read_bytes()
    source = json.loads(raw)
    features = [f for f in source['features']
                if f['geometry']['type'] in ('Polygon', 'MultiPolygon')]
    polygons = [shape(f['geometry']) for f in features]
    merged = unary_union(polygons)
    assert merged.is_valid and merged.geom_type == 'MultiPolygon'
    exteriors = [list(p.exterior.coords) for p in merged.geoms]
    result = dict(type='FeatureCollection', crs=source.get('crs'), features=[dict(
        type='Feature', properties=dict(
            source=str(SOURCE.relative_to(ROOT)),
            source_sha256=hashlib.sha256(raw).hexdigest(),
            derivation='unary_union of province polygons; exterior rings only; no simplification',
        ), geometry=dict(type='MultiLineString', coordinates=exteriors))])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, separators=(',', ':')))
    checks = {
        'province_features': len(polygons), 'exterior_rings': len(exteriors),
        'input_invalid_rings': [dict(name=f['properties']['name'], issue=explain_validity(p))
                                for f, p in zip(features, polygons) if not p.is_valid],
        'union_valid': merged.is_valid,
        'sum_input_area_deg2': sum(p.area for p in polygons),
        'union_area_deg2': merged.area,
        'southern_tibet_points_in_outline': {
            str(p): merged.covers(Point(p))
            for p in [(91.75, 27.75), (94, 28), (93.6, 27.1), (95.3, 28.1)]},
        'aksai_chin_check_in_outline': merged.covers(Point(79.3, 35.2)),
        'island_geometry_in_diaoyu_region': merged.intersects(box(123, 25, 125, 26.5)),
        'source_sha256': hashlib.sha256(raw).hexdigest(),
    }
    qa = V86 / 'figures/qa/main_figure1_china_geometry.json'
    qa.write_text(json.dumps(checks, ensure_ascii=False, indent=2))
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
