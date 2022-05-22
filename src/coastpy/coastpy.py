from coastpy.geometries.boxes import make_boxes

if __name__ == "__main__":

    boxes = make_boxes(zoom_level=7, output_crs="epsg:4326")
    print(boxes.shape)
