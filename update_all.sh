#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"

GFS_BUCKET="https://noaa-gfs-bdp-pds.s3.amazonaws.com"
AI_BUCKET="https://noaa-nws-graphcastgfs-pds.s3.amazonaws.com"

has_key() {
  local url="$1"
  curl --fail --silent --show-error --max-time 30 "$url" | grep -q '<Key>'
}

# Select the newest cycle for which both model lines have their long forecast.
DATE=""
CYCLE=""
for age in 0 1 2; do
  day="$(date -u -d "-${age} day" +%Y%m%d)"
  for cc in 18 12 06 00; do
    gfs_idx="${GFS_BUCKET}/?list-type=2&prefix=gfs.${day}/${cc}/atmos/gfs.t${cc}z.pgrb2.0p25.f240.idx&max-keys=2"
    ai_file="${AI_BUCKET}/?list-type=2&prefix=aigfs.${day}/${cc}/model/atmos/grib2/aigfs.t${cc}z.sfc.f384.grib2&max-keys=2"
    if has_key "$gfs_idx" && has_key "$ai_file"; then
      DATE="$day"; CYCLE="$cc"; break 2
    fi
  done
done

if [[ -z "$DATE" ]]; then
  echo "No common complete GFS/AI-GFS cycle found in the last 72 hours." >&2
  exit 1
fi

echo "Selected complete cycle: ${DATE}/${CYCLE}Z"
python3 update_all.py "$DATE" "$CYCLE"

python3 build_3comp_24h.py "$DATE" "$CYCLE"
python3 build_3comp_scenario.py "$DATE" "$CYCLE"
python3 build_3comp_ai.py "$DATE" "$CYCLE"

python3 hadramout_v17_total_24h_map.py "$DATE" "$CYCLE"
python3 hadramout_v17_3comp_panel_24h.py
python3 hadramout_v17_3comp_panel.py
python3 hadramout_ai_total_24h_map.py "$DATE" "$CYCLE"
python3 hadramout_ai_3comp_panel_24h.py
python3 hadramout_ai_3comp_panel.py

# eccodes may abort after successfully closing the ocean plot; retain the files
# and continue, but fail if the expected image was not produced.
set +e
python3 arabian_sea_tropical_14d.py "$DATE" "$CYCLE"
sea_gfs_status=$?
python3 arabian_sea_tropical_ai.py "$DATE" "$CYCLE"
sea_ai_status=$?
set -e
[[ -s "hadramout_v17_arabian_sea_14d_${DATE}_${CYCLE}Z.png" ]] || exit 1
[[ -s "hadramout_ai_arabian_sea_14d_${DATE}_${CYCLE}Z.png" ]] || exit 1
printf 'Ocean renderer exit codes after files were written: GFS=%s AI=%s\n' "$sea_gfs_status" "$sea_ai_status"

python3 verify_mukalla.py || true
python3 make_gallery.py
