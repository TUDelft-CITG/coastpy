from typing import Annotated, Literal

from msgspec import Meta

# Type literals for structured data
ShoreType = Literal[
    "sandy_gravel_or_small_boulder_sediments",
    "muddy_sediments",
    "rocky_shore_platform_or_large_boulders",
    "no_sediment_or_shore_platform",
    "ice_or_tundra",
]
CoastalType = Literal[
    "cliffed_or_steep",
    "moderately_sloped",
    "bedrock_plain",
    "sediment_plain",
    "dune",
    "wetland",
    "coral",
    "inlet",
    "engineered_structures",
]
LandformType = Literal[
    "barrier_island",
    "barrier_system",
    "barrier",
    "bay",
    "coral_island",
    "delta",
    "estuary",
    "headland",
    "inlet",
    "lagoon",
    "mainland_coast",
    "pocket_beach",
    "spit",
    "tombolo",
    "N/A",
]
IsBuiltEnvironment = Literal["true", "false", "N/A"]
HasDefense = Literal["true", "false", "N/A"]


DeterminationMethod = Annotated[
    Literal[
        "ShorelineMonitor",
        "CoastSat",
        "Cassie",
        "ManualDigitization",
        "AIModel",
        "Thresholding",
    ],
    Meta(
        description="Methods or tools used to determine waterlines, including algorithms and software"
    ),
]

EOInstrument = Annotated[
    Literal[
        "Sentinel-1 SAR",
        "Sentinel-2 MSI",
        "Landsat-8 OLI",
        "WorldView-2",
        "WorldView-3",
        "PlanetScope",
        "SkySat",
    ],
    Meta(
        description="Earth Observation instruments or sensors used in coastal monitoring"
    ),
]

Provider = Annotated[
    Literal["Deltares", "USGS", "CoastSat", "Planetary Computer", "Maxar", "Airbus"],
    Meta(description="Organizations or entities providing coastal monitoring data"),
]

WaterLineType = Annotated[
    Literal["instantaneous", "composite"],
    Meta(
        description=(
            "Types of waterlines: 'instantaneous' represents single moment waterlines (e.g., derived from one image), "
            "while 'composite' refers to aggregated waterlines from multiple images."
        )
    ),
]
