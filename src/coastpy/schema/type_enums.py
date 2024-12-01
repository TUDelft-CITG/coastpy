from typing import Literal

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
