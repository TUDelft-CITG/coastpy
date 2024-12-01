import datetime

import msgspec
from shapely import LineString

from coastpy.schema import (
    Transect,
    TypologyInferenceSample,
    TypologyTestSample,
    TypologyTrainSample,
    custom_schema_hook,
)

if __name__ == "__main__":
    linestring = LineString([[45, 55], [34, 57]])
    bounds = linestring.bounds
    bbox = {"minx": bounds[0], "miny": bounds[1], "maxx": bounds[2], "maxy": bounds[3]}
    transect = Transect(
        transect_id="a",
        geometry=linestring,
        bearing=40.0,
        bbox=bbox,
    )

    # NOTE: continue with TypologyTrainSample.null().to_frame()
    train_sample = TypologyTrainSample(
        transect=transect,
        user="floris-calkoen",
        uuid="3b984582ecd6",
        datetime_created=datetime.datetime.now(),
        datetime_updated=datetime.datetime.now(),
        shore_type="sandy_gravel_or_small_boulder_sediments",
        coastal_type="cliffed_or_steep",
        landform_type="mainland_coast",
        is_built_environment="false",
        has_defense="true",
        confidence="high",
        is_validated=True,
        is_challenging=False,
        comment="This is a test comment",
        link="https://example.com",
    )

    # fr = transect.to_frame2()
    fr2 = train_sample.to_frame(geometry="transect.geometry", bbox="transect.bbox")
    d1 = train_sample.to_dict()
    d2 = train_sample.to_dict(flatten=True)
    d3 = TypologyTrainSample.from_dict(d1)
    d4 = TypologyTrainSample.from_dict(d2)
    d5 = TypologyTrainSample.from_frame(fr2)
    empty_frame = TypologyTrainSample.null().empty_frame(geometry="transect.geometry")

    test_sample = TypologyTestSample.example()
    fr3 = test_sample.to_frame()

    transect_nulls = Transect.null()
    train_nulls = TypologyTrainSample.null()
    train_sample.to_dict()
    train_sample.to_meta()
    TypologyTrainSample.example()
    schema = msgspec.json.schema(Transect, schema_hook=custom_schema_hook)
    import json
    import pathlib

    with (pathlib.Path.cwd() / "transect_schema.json").open("w") as f:
        f.write(json.dumps(schema, indent=2))

    # Transect.null()
    TypologyTrainSample.null()
    test_sample = TypologyTestSample(
        train_sample=train_sample,
        pred_shore_type="sandy_gravel_or_small_boulder_sediments",
        pred_coastal_type="bedrock_plain",
        pred_has_defense="true",
        pred_is_built_environment="false",
    )

    inference_sample = TypologyInferenceSample(
        transect=transect,
        pred_shore_type="sandy_gravel_or_small_boulder_sediments",
        pred_coastal_type="bedrock_plain",
        pred_has_defense="true",
        pred_is_built_environment="false",
    )
    print("done")
