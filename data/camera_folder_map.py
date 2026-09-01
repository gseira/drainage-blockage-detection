"""
camera_folder_map.py
----------------------
Shared campath -> images/<folder> naming, used by both
data/export_reviewed_images.py (batch export) and app/main.py's
/results/reclassify endpoint (live, per-correction export) — kept in one
place so the two never drift into using different folder names for the
same camera.

Best-known folder names for cameras that already have labeled data in
images/ on hex (confirmed during the 2026-07-07 fine-tuning work) — new
exports for these land in the SAME folder as the existing data, rather
than creating a second, differently-named one for the same camera.
Cameras NOT listed here fall back to a sanitized campath (region_Location)
— this is NOT guaranteed to match any pre-existing folder naming
convention on hex; check manually before assuming it lines up.
"""

CAMPATH_TO_FOLDER = {
    "cornwall/WadebridgePolmorla/cam1":  "Cornwall_WadebridgePolmorla",
    "cornwall/PlymptonChaddlewood/cam1": "Cornwall_PlymptonChaddlewood_MainScree",
    "cornwall/BodminPetrocsWell/cam1":   "Cornwall_BodminPetrocsWell_Scree",
    "cornwall/BudeCedarGrove/cam1":      "Cornwall_BudeCedarGrove",
    "cornwall/KingsandCP/cam1":          "Cornwall_KingsandCP",
    "cornwall/LostwithielUP/cam1":       "Cornwall_LostwithielUP_Scree",
    "cornwall/Mevagissey/cam1":          "Cornwall_Mevagissey_PreScree",
    "cornwall/Penryn/cam1":              "Cornwall_PenrynTP",
    "cornwall/PenzanceCC/cam1":          "Cornwall_PenzanceCC",
    "cornwall/PlymptonForSt/cam1":       "Cornwall_PlymptonForSt",
    "cornwall/PorthlevenScreen/cam1":    "Cornwall_PorthlevenScree",
    "cornwall/Portreath/cam4":           "Cornwall_Portreath",
    "cornwall/TamertonFoliot/cam2":      "Cornwall_TamertonFoliot",
    "cornwall/StIvesConsols/cam1":       "Cornwall_StIvesConsols_Scree",
    "BarnstapleBradiford":               "Devon_BarnstapleBradiford",
    "BarnstapleConeyGut/Screen":         "Devon_BarnstapleConeyGut_Scree",
    "BarnstaplePortmarshLane":           "Devon_BarnstaplePortmarshLane",
    "Buckfastleigh":                     "Devon_Buckfastleigh",
    "KenwithValleyChannelScreen/Screen": "Devon_KenwithValleyChannelScree",
    "LympstoneScreen":                   "Devon_LympstoneScree",
    "SwimbridgeScreen":                  "Devon_SwimbridgeScree",

    # The 5 cameras confirmed (2026-07-08) to have ZERO historical data
    # anywhere — not in images/, not in the 80k raw dataset, under any
    # naming variant. These entries exist purely so live corrections land
    # in a consistent Region_LocationName folder from the very first one,
    # instead of falling back to a differently-styled sanitized campath.
    "cornwall/Porthallow/cam1":      "Cornwall_Porthallow",
    "cornwall/LauncestonWooda/cam1": "Cornwall_LauncestonWooda",
    "AshburtonLower/Screen":         "Devon_AshburtonLower",
    "KingsbridgeDuncombe":           "Devon_KingsbridgeDuncombe",
    "NewtonAbbotBakersPark":         "Devon_NewtonAbbotBakersPark",
}


def folder_for(campath: str) -> str:
    if campath in CAMPATH_TO_FOLDER:
        return CAMPATH_TO_FOLDER[campath]
    # Fallback for cameras with no known hex folder yet — sanitized campath.
    return campath.replace("/", "_").replace(" ", "_")
