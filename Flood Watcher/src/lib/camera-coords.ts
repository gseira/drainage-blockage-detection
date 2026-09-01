/**
 * camera-coords.ts
 * ----------------
 * SVG map coordinates AND real GPS lat/lng for every EA screen camera.
 *
 * SVG ViewBox: 0..1000 × 0..640  (Devon right, Cornwall left)
 * GPS: WGS-84 decimal degrees (approx. town-level accuracy)
 */

export interface CameraCoords {
  /** SVG x coordinate for the stylised map */
  x: number;
  /** SVG y coordinate for the stylised map */
  y: number;
  /** Real latitude (WGS-84) */
  lat: number;
  /** Real longitude (WGS-84, negative = West) */
  lng: number;
}

export const CAMERA_COORDS: Record<string, CameraCoords> = {
  // ── Cornwall ───────────────────────────────────────────────────────────────
  "cornwall/Angarrack/cam1":          { x: 198, y: 463, lat: 50.178, lng: -5.311 },
  "cornwall/BodminFlaxmoor/cam1":     { x: 394, y: 388, lat: 50.471, lng: -4.717 },
  "cornwall/BodminPetrocsWell/cam1":  { x: 400, y: 392, lat: 50.465, lng: -4.722 },
  "cornwall/Boscundle/cam1":          { x: 374, y: 414, lat: 50.341, lng: -4.793 },
  "cornwall/BudeBerries/cam1":        { x: 436, y: 294, lat: 50.828, lng: -4.551 },
  "cornwall/BudeCedarGrove/cam1":     { x: 432, y: 298, lat: 50.831, lng: -4.549 },
  "cornwall/CawsandCP/cam1":          { x: 553, y: 425, lat: 50.334, lng: -4.208 },
  "cornwall/GorranHaven/cam1":        { x: 371, y: 450, lat: 50.241, lng: -4.792 },
  "cornwall/HayleMellanear/cam1":     { x: 170, y: 463, lat: 50.186, lng: -5.420 },
  "cornwall/HorrabridgeFP/cam1":      { x: 568, y: 382, lat: 50.506, lng: -4.117 },
  "cornwall/HorrabridgeSF/cam1":      { x: 564, y: 386, lat: 50.508, lng: -4.120 },
  "cornwall/KingsandCP/cam1":         { x: 557, y: 428, lat: 50.337, lng: -4.212 },
  "cornwall/LauncestonWooda/cam1":    { x: 496, y: 354, lat: 50.636, lng: -4.362 },
  "cornwall/LoeBar/cam1":             { x: 200, y: 487, lat: 50.059, lng: -5.244 },
  "cornwall/LostwithielUP/cam1":      { x: 410, y: 408, lat: 50.404, lng: -4.668 },
  "cornwall/Mevagissey/cam1":         { x: 374, y: 444, lat: 50.267, lng: -4.786 },
  "cornwall/Penryn/cam1":             { x: 272, y: 468, lat: 50.167, lng: -5.103 },
  "cornwall/PentewanScreen/cam1":     { x: 378, y: 442, lat: 50.275, lng: -4.769 },
  "cornwall/PenzanceCS/cam1":         { x: 138, y: 480, lat: 50.118, lng: -5.537 },
  "cornwall/PenzanceCC/cam1":         { x: 134, y: 484, lat: 50.115, lng: -5.541 },
  "cornwall/Perrancoombe/cam1":       { x: 253, y: 422, lat: 50.345, lng: -5.141 },
  "cornwall/PlymptonChaddlewood/cam1":{ x: 584, y: 414, lat: 50.390, lng: -4.050 },
  "cornwall/PlymptonForSt/cam1":      { x: 580, y: 418, lat: 50.387, lng: -4.054 },
  "cornwall/PolperroTS/cam1":         { x: 438, y: 426, lat: 50.332, lng: -4.519 },
  "cornwall/PortIsaac/cam1":          { x: 358, y: 366, lat: 50.594, lng: -4.832 },
  "cornwall/Porthallow/cam1":         { x: 280, y: 493, lat: 50.067, lng: -5.069 },
  "cornwall/PorthlevenScreen/cam1":   { x: 202, y: 490, lat: 50.083, lng: -5.318 },
  "cornwall/Portreath/cam4":          { x: 208, y: 444, lat: 50.261, lng: -5.289 },
  "cornwall/StIvesConsols/cam1":      { x: 150, y: 461, lat: 50.211, lng: -5.474 },
  "cornwall/TamertonFoliot/cam2":     { x: 565, y: 396, lat: 50.432, lng: -4.129 },
  "cornwall/WadebridgePolmorla/cam1": { x: 355, y: 378, lat: 50.518, lng: -4.840 },
  "cornwall/WalkhamptonScreen/cam1":  { x: 566, y: 380, lat: 50.519, lng: -4.096 },

  // ── Devon ──────────────────────────────────────────────────────────────────
  "AshburtonLower/Screen":            { x: 701, y: 377, lat: 50.514, lng: -3.753 },
  "BarnstapleBradiford":              { x: 605, y: 225, lat: 51.082, lng: -4.058 },
  "BarnstapleConeyGut/Screen":        { x: 609, y: 229, lat: 51.079, lng: -4.055 },
  "BarnstapleNewportRoad":            { x: 603, y: 233, lat: 51.076, lng: -4.062 },
  "BarnstaplePortmarshLane":          { x: 601, y: 228, lat: 51.080, lng: -4.065 },
  "BidefordElliotsGarageScreen":      { x: 557, y: 244, lat: 51.011, lng: -4.208 },
  "BrauntonHordens":                  { x: 573, y: 218, lat: 51.106, lng: -4.163 },
  "Buckfastleigh":                    { x: 695, y: 385, lat: 50.481, lng: -3.775 },
  "CullomptonLanglands":              { x: 816, y: 287, lat: 50.849, lng: -3.391 },
  "DawlishWarren":                    { x: 803, y: 365, lat: 50.595, lng: -3.449 },
  "Dulverton":                        { x: 771, y: 245, lat: 51.015, lng: -3.556 },
  "EastBudleigh":                     { x: 819, y: 350, lat: 50.643, lng: -3.390 },
  "HarbertonfordScreen":              { x: 704, y: 412, lat: 50.357, lng: -3.716 },
  "IlfracombeScreen":                 { x: 581, y: 193, lat: 51.210, lng: -4.117 },
  "KenwithValleyChannelScreen/Screen":{ x: 554, y: 234, lat: 51.016, lng: -4.231 },
  "KingsbridgeDuncombe":              { x: 695, y: 442, lat: 50.283, lng: -3.776 },
  "LympstoneScreen":                  { x: 806, y: 352, lat: 50.640, lng: -3.429 },
  "NewtonAbbotBakersPark":            { x: 758, y: 372, lat: 50.528, lng: -3.611 },
  "OccombeValley":                    { x: 763, y: 400, lat: 50.437, lng: -3.581 },
  "OtterySMChapelLane":               { x: 851, y: 325, lat: 50.748, lng: -3.278 },
  "OtterySMKenawayRoad":              { x: 847, y: 329, lat: 50.745, lng: -3.281 },
  "SwimbridgeScreen":                 { x: 667, y: 234, lat: 51.033, lng: -3.953 },
  "TeignmouthFirstAvenue/cam2":       { x: 782, y: 367, lat: 50.546, lng: -3.494 },
  "Yalberton":                        { x: 755, y: 403, lat: 50.430, lng: -3.601 },
};

/** Look up full coordinates for a campath; fall back to a plausible region position. */
export function getCoords(campath: string): CameraCoords {
  if (CAMERA_COORDS[campath]) return CAMERA_COORDS[campath];
  const isDevon = !campath.startsWith("cornwall/");
  return isDevon
    ? { x: 680 + Math.floor(Math.random() * 120), y: 300 + Math.floor(Math.random() * 120), lat: 50.7 + Math.random() * 0.4, lng: -3.7 - Math.random() * 0.5 }
    : { x: 240 + Math.floor(Math.random() * 180), y: 380 + Math.floor(Math.random() * 120), lat: 50.3 + Math.random() * 0.4, lng: -4.8 - Math.random() * 0.8 };
}
