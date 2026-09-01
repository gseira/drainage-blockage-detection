/**
 * dashboard-data.ts
 * -----------------
 * Shared types for the DrainWatch dashboard.
 * STATIC_ASSETS is a fallback for when the backend is offline.
 * Live data comes from useAssets() → /webcams + /results/latest.
 */

import { getCoords } from "./camera-coords";
import { campathToId } from "./api";
import type { WebcamDef, ResultRow } from "./api";

export type Risk        = "high" | "medium" | "low";
export type AssetStatus = "blocked" | "monitoring" | "clear" | "offline";

export interface Asset {
  id: string;
  name: string;
  region: "Cornwall" | "Devon";
  campath: string;
  /** SVG map x coordinate */
  x: number;
  /** SVG map y coordinate */
  y: number;
  /** Real GPS latitude */
  lat: number;
  /** Real GPS longitude */
  lng: number;
  risk: Risk;
  status: AssetStatus;
  /** Blockage percentage 0-100 */
  blockage: number;
  /** Model confidence 0-100 */
  confidence: number;
  cameraId: string;
  updatedMinsAgo: number;
  /** Raw p_blocked from model (0-1) */
  pBlocked: number | null;
  /** Heatmap coverage fraction (0-1) */
  heatmapCoverage: number | null;
  /** LLM explanation text */
  explanation: string | null;
  /** base64 Grad-CAM overlay JPEG */
  overlayB64: string | null;
  /** base64 heatmap JPEG */
  heatmapB64: string | null;
  /** ISO string of last scan */
  lastScanAt: string | null;
}

/** Convert backend risk string → frontend Risk */
function toRisk(r: string): Risk {
  if (r === "HIGH")    return "high";
  if (r === "FLAGGED") return "medium";
  return "low";
}

/** Convert backend risk string → frontend AssetStatus */
function toStatus(r: string): AssetStatus {
  if (r === "HIGH")                        return "blocked";
  if (r === "OFFLINE" || r === "UNKNOWN")  return "offline";
  if (r === "FLAGGED")                     return "monitoring";
  return "clear";
}

/** Merge a WebcamDef + optional ResultRow into an Asset */
export function toAsset(cam: WebcamDef, result: ResultRow | null): Asset {
  const coords      = getCoords(cam.campath);
  const risk        = result ? toRisk(result.risk) : "low";
  const status      = result ? toStatus(result.risk) : "clear";
  const pBlocked    = result?.p_blocked ?? null;
  const blockage    = pBlocked !== null ? Math.round(pBlocked * 100) : 0;
  const confidence  = pBlocked !== null ? Math.round(Math.max(pBlocked, 1 - pBlocked) * 100) : 0;

  // Minutes since last scan
  let updatedMinsAgo = 0;
  if (result?.created_at) {
    const diffMs = Date.now() - new Date(result.created_at).getTime();
    updatedMinsAgo = Math.max(0, Math.round(diffMs / 60_000));
  }

  return {
    id:              campathToId(cam.campath),
    name:            cam.name,
    region:          cam.region,
    campath:         cam.campath,
    x:               coords.x,
    y:               coords.y,
    lat:             coords.lat,
    lng:             coords.lng,
    risk,
    status,
    blockage,
    confidence,
    cameraId:        cam.campath.split("/").slice(-2).join("/").toUpperCase(),
    updatedMinsAgo,
    pBlocked,
    heatmapCoverage: result?.heatmap_coverage ?? null,
    explanation:     result?.explanation ?? null,
    overlayB64:      result?.overlay_b64 ?? null,
    heatmapB64:      result?.heatmap_b64 ?? null,
    lastScanAt:      result?.created_at ?? null,
  };
}

// ── Static fallback (used when backend is offline) ────────────────────────────

function cam(
  name: string, region: "Cornwall" | "Devon", campath: string,
  risk: Risk, status: AssetStatus, blockage: number, confidence: number, updatedMinsAgo: number,
): Asset {
  const coords = getCoords(campath);
  return {
    id: campathToId(campath),
    name, region, campath,
    x: coords.x, y: coords.y,
    lat: coords.lat, lng: coords.lng,
    risk, status, blockage, confidence,
    cameraId: campath.split("/").slice(-2).join("/").toUpperCase(),
    updatedMinsAgo,
    pBlocked: blockage / 100,
    heatmapCoverage: null,
    explanation: null, overlayB64: null, heatmapB64: null,
    lastScanAt: null,
  };
}

export const STATIC_ASSETS: Asset[] = [
  // Cornwall
  cam("Angarrack",              "Cornwall", "cornwall/Angarrack/cam1",          "high", "blocked",    78, 94,  2),
  cam("Bodmin Flaxmoor Ter.",   "Cornwall", "cornwall/BodminFlaxmoor/cam1",     "low",  "clear",       8, 90,  5),
  cam("Bodmin Petrocs Well",    "Cornwall", "cornwall/BodminPetrocsWell/cam1",  "low",  "clear",       8, 90,  3),
  cam("Boscundle",              "Cornwall", "cornwall/Boscundle/cam1",          "low",  "clear",      10, 78, 12),
  cam("Bude Berries Avenue",    "Cornwall", "cornwall/BudeBerries/cam1",        "low",  "clear",      11, 88,  7),
  cam("Bude Cedar Grove",       "Cornwall", "cornwall/BudeCedarGrove/cam1",     "high", "blocked",    72, 92,  1),
  cam("Cawsand Car Park",       "Cornwall", "cornwall/CawsandCP/cam1",          "low",  "clear",       6, 95,  4),
  cam("Gorran Haven",           "Cornwall", "cornwall/GorranHaven/cam1",        "low",  "clear",      15, 76,  9),
  cam("Hayle Mellanear",        "Cornwall", "cornwall/HayleMellanear/cam1",     "high", "blocked",    81, 96,  2),
  cam("Horrabridge Fillace Pk", "Cornwall", "cornwall/HorrabridgeFP/cam1",      "low",  "clear",      14, 87,  6),
  cam("Horrabridge Springflds", "Cornwall", "cornwall/HorrabridgeSF/cam1",      "low",  "clear",      20, 83, 18),
  cam("Kingsand Car Park",      "Cornwall", "cornwall/KingsandCP/cam1",         "low",  "clear",       5, 91,  8),
  cam("Launceston Wooda Lane",  "Cornwall", "cornwall/LauncestonWooda/cam1",    "low",  "clear",      12, 79, 22),
  cam("Loe Bar",                "Cornwall", "cornwall/LoeBar/cam1",             "high", "blocked",    69, 88,  3),
  cam("Lostwithiel Uzella Pk",  "Cornwall", "cornwall/LostwithielUP/cam1",      "low",  "clear",      12, 93,  5),
  cam("Mevagissey Fire Stn",   "Cornwall", "cornwall/Mevagissey/cam1",         "low",  "clear",      20, 80, 14),
  cam("Penryn",                "Cornwall", "cornwall/Penryn/cam1",             "low",  "clear",       9, 89,  2),
  cam("Pentewan Screen",        "Cornwall", "cornwall/PentewanScreen/cam1",     "high", "blocked",    74, 91,  4),
  cam("Penzance Chyander Sq",   "Cornwall", "cornwall/PenzanceCS/cam1",         "low",  "clear",       7, 92,  6),
  cam("Penzance Coombe Cott.",  "Cornwall", "cornwall/PenzanceCC/cam1",         "low",  "clear",      15, 77, 11),
  cam("Perrancoombe",           "Cornwall", "cornwall/Perrancoombe/cam1",       "low",  "clear",      10, 86,  3),
  cam("Plympton Chaddlewood",   "Cornwall", "cornwall/PlymptonChaddlewood/cam1","low",  "clear",      15, 82,  8),
  cam("Plympton Fore St",       "Cornwall", "cornwall/PlymptonForSt/cam1",      "low",  "offline",     0,  0, 54),
  cam("Polperro Top Screen",    "Cornwall", "cornwall/PolperroTS/cam1",         "high", "blocked",    67, 89,  1),
  cam("Port Isaac",             "Cornwall", "cornwall/PortIsaac/cam1",          "low",  "clear",      13, 90,  7),
  cam("Porthallow",             "Cornwall", "cornwall/Porthallow/cam1",         "low",  "clear",      20, 78, 16),
  cam("Porthleven Screen",      "Cornwall", "cornwall/PorthlevenScreen/cam1",   "low",  "clear",       3, 94,  2),
  cam("Portreath",              "Cornwall", "cornwall/Portreath/cam4",          "low",  "clear",      20, 81,  9),
  cam("St Ives Consols",        "Cornwall", "cornwall/StIvesConsols/cam1",      "low",  "clear",       8, 88,  4),
  cam("Tamerton Foliot",        "Cornwall", "cornwall/TamertonFoliot/cam2",     "high", "blocked",    76, 93,  2),
  cam("Wadebridge Polmorla",    "Cornwall", "cornwall/WadebridgePolmorla/cam1", "low",  "clear",      15, 87,  6),
  cam("Walkhampton Screen",     "Cornwall", "cornwall/WalkhamptonScreen/cam1",  "low",  "clear",      20, 75, 20),
  // Devon
  cam("Ashburton Lower Screen",   "Devon", "AshburtonLower/Screen",             "low",  "clear",      20, 80,  7),
  cam("Barnstaple Bradiford",     "Devon", "BarnstapleBradiford",               "high", "blocked",    82, 95,  1),
  cam("Barnstaple Coney Gut",     "Devon", "BarnstapleConeyGut/Screen",         "low",  "clear",       6, 91,  3),
  cam("Barnstaple Newport Rd",    "Devon", "BarnstapleNewportRoad",             "low",  "clear",      20, 83, 10),
  cam("Barnstaple Portmarsh Ln", "Devon", "BarnstaplePortmarshLane",           "low",  "clear",      11, 88,  5),
  cam("Bideford Elliots Garage", "Devon", "BidefordElliotsGarageScreen",       "high", "blocked",    70, 90,  2),
  cam("Braunton Hordens Bridge", "Devon", "BrauntonHordens",                   "low",  "clear",       9, 89,  4),
  cam("Buckfastleigh",            "Devon", "Buckfastleigh",                     "low",  "clear",      20, 79, 13),
  cam("Cullompton Langlands",     "Devon", "CullomptonLanglands",               "low",  "clear",       7, 92,  6),
  cam("Dawlish Warren",           "Devon", "DawlishWarren",                     "high", "blocked",    75, 93,  2),
  cam("Dulverton",                "Devon", "Dulverton",                         "low",  "clear",      12, 87,  8),
  cam("East Budleigh",            "Devon", "EastBudleigh",                      "low",  "clear",      20, 82, 15),
  cam("Harbertonford Screen",     "Devon", "HarbertonfordScreen",               "low",  "clear",       5, 93,  3),
  cam("Ilfracombe Screen",        "Devon", "IlfracombeScreen",                  "high", "blocked",    68, 88,  1),
  cam("Kenwith Valley Screen",    "Devon", "KenwithValleyChannelScreen/Screen", "low",  "offline",     0,  0, 48),
  cam("Kingsbridge Duncombe",     "Devon", "KingsbridgeDuncombe",               "low",  "clear",      20, 84,  9),
  cam("Lympstone Screen",         "Devon", "LympstoneScreen",                   "low",  "clear",      10, 90,  5),
  cam("Newton Abbot Bakers Pk",  "Devon", "NewtonAbbotBakersPark",             "low",  "clear",      20, 81, 11),
  cam("Occombe Valley Screen",    "Devon", "OccombeValley",                     "low",  "clear",       8, 88,  4),
  cam("Ottery SM Chapel Lane",    "Devon", "OtterySMChapelLane",                "high", "blocked",    79, 94,  2),
  cam("Ottery SM Kennaway Rd",   "Devon", "OtterySMKenawayRoad",               "low",  "clear",      20, 76, 19),
  cam("Swimbridge Screen",        "Devon", "SwimbridgeScreen",                  "low",  "clear",      13, 91,  6),
  cam("Teignmouth First Ave",     "Devon", "TeignmouthFirstAvenue/cam2",        "low",  "clear",      20, 83,  8),
  cam("Yalberton",                "Devon", "Yalberton",                         "low",  "clear",       4, 95,  3),
];

// Keep legacy name `assets` pointing to static fallback for any code that still references it
export const assets = STATIC_ASSETS;

export const aiFindings = [
  { label: "Organic debris (branches, foliage)", value: 62 },
  { label: "Sediment build-up",                  value: 18 },
  { label: "Plastic / anthropogenic waste",      value: 11 },
  { label: "Unobstructed flow area",             value:  9 },
];

export const inspectionHistory = [
  { ts: "29 Jun 14:02", actor: "ResNet-50", note: "Blockage reclassified HIGH — p_blocked rose to 0.94." },
  { ts: "29 Jun 09:18", actor: "Grad-CAM",  note: "Heatmap coverage 68% — debris concentrated at screen centre." },
  { ts: "28 Jun 17:44", actor: "ResNet-50", note: "Confidence raised to 96% after lighting normalisation." },
  { ts: "28 Jun 06:30", actor: "LLM",       note: "Dense organic material visible across screen grid." },
  { ts: "26 Jun 11:05", actor: "ResNet-50", note: "Risk LOW → HIGH after sustained rainfall event." },
];
