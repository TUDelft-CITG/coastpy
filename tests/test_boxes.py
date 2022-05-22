from coastpy.geometries.boxes import make_boxes


def test_make_boxes():

    boxes = make_boxes(zoom_level=7, output_crs="epsg:4326")
    actual = boxes.shape
    expected = (16384, 2)
    assert actual == expected, "Shape of boxes dataframe is correct. "
