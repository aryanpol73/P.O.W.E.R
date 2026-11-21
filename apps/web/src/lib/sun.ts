// Sunrise / sunset for a place and date (NOAA solar-position equations, accurate to about a minute),
// used to switch the city map between its day and night styles on the simulation clock.

const RAD = Math.PI / 180;

export type SunTimes = { sunrise: Date; sunset: Date };

/** Sunrise and sunset (as UTC instants) on the UTC calendar day of `when`, at latitude/longitude in degrees. */
export function sunTimes(when: Date, lat: number, lon: number): SunTimes {
  const day = Date.UTC(when.getUTCFullYear(), when.getUTCMonth(), when.getUTCDate());
  const n = (day - Date.UTC(when.getUTCFullYear(), 0, 1)) / 86_400_000; // day of year, 0-based
  const g = ((2 * Math.PI) / 365) * (n + 0.5); // fractional year at noon (radians)
  const eqTime = 229.18 * (0.000075 + 0.001868 * Math.cos(g) - 0.032077 * Math.sin(g) - 0.014615 * Math.cos(2 * g) - 0.040849 * Math.sin(2 * g));
  const decl =
    0.006918 - 0.399912 * Math.cos(g) + 0.070257 * Math.sin(g) - 0.006758 * Math.cos(2 * g) + 0.000907 * Math.sin(2 * g) -
    0.002697 * Math.cos(3 * g) + 0.00148 * Math.sin(3 * g);
  // hour angle at the moment the sun's upper limb touches the horizon, with refraction (90.833 deg zenith)
  const cosH = Math.cos(90.833 * RAD) / (Math.cos(lat * RAD) * Math.cos(decl)) - Math.tan(lat * RAD) * Math.tan(decl);
  const ha = Math.acos(Math.min(1, Math.max(-1, cosH))) / RAD;
  const noon = 720 - 4 * lon - eqTime; // solar noon, minutes after 00:00 UTC
  return {
    sunrise: new Date(day + (noon - 4 * ha) * 60_000),
    sunset: new Date(day + (noon + 4 * ha) * 60_000),
  };
}

/** True when the sun is down at `when` at this place. */
export function isNight(when: Date, lat: number, lon: number): boolean {
  // the local day can straddle two UTC dates (IST is UTC+5:30), so check the neighbouring days too
  const t = when.getTime();
  for (const offset of [-1, 0, 1]) {
    const { sunrise, sunset } = sunTimes(new Date(t + offset * 86_400_000), lat, lon);
    if (t >= sunrise.getTime() && t < sunset.getTime()) return false;
  }
  return true;
}
