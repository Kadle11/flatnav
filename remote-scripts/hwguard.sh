#!/usr/bin/env bash
# Shared hardware-state guards for the measurement scripts in this directory. Source it:
#
#   . "$(dirname "$0")/hwguard.sh"
#
# It exists because three silent failures were found in already-archived sweep data, all of the
# same kind: the hardware quietly did something other than what the run labelled it.
#
#   1. An uncore ratio below the package's floor is CLAMPED, not rejected. RATIOS="24 16 8 4" on
#      a part whose floor is 800 MHz (= ratio 8) makes r4 a duplicate of r8 -- four labelled
#      points, three distinct ones, and a "trend" with a fabricated last step. set_ratio only
#      warned, to a console that was never archived. hwguard_require_ratios refuses to start.
#
#   2. A killed run leaks its throttle. min_freq_khz stays pinned where that run left it; the
#      next run reads the leak as "original", restores it faithfully on exit, and it becomes
#      permanent. Anything measured before the first set_ratio -- the near baselines -- then ran
#      at whatever clock the last crash happened to leave behind. hwguard_uncore_init compares
#      against initial_*_freq_khz and resets, so each script captures clean values.
#
#   3. AutoNUMA was on for every run so far. It cannot move the index (numa_alloc_onnode binds
#      MPOL_BIND, and bound pages are never "misplaced"), but it can migrate the PQ codes and
#      query arrays, and it can migrate THREADS across sockets in any run pinned to more than
#      one node -- which is exactly the dual-socket control. Off for the duration, restored.
#
# Every mutation here is restored by hwguard_restore_all, which callers put in their EXIT trap.

# --- AutoNUMA -------------------------------------------------------------------------------
HWG_NB_ORIG=""
hwguard_numa_balancing_off() {
  [ -r /proc/sys/kernel/numa_balancing ] || { echo "[hwguard] no numa_balancing knob on this kernel"; return 0; }
  local now; now=$(cat /proc/sys/kernel/numa_balancing)
  [ "$now" = "0" ] && { echo "[hwguard] numa_balancing already 0"; return 0; }
  echo 0 | sudo tee /proc/sys/kernel/numa_balancing >/dev/null
  HWG_NB_ORIG=$now
  echo "[hwguard] numa_balancing $now -> 0 (restored on exit)"
}
hwguard_numa_balancing_restore() {
  [ -n "$HWG_NB_ORIG" ] || return 0
  echo "$HWG_NB_ORIG" | sudo tee /proc/sys/kernel/numa_balancing >/dev/null
  echo "[hwguard] numa_balancing restored to $HWG_NB_ORIG"
  HWG_NB_ORIG=""
}

# --- uncore range, leftover state -----------------------------------------------------------
HWG_UNCORE_DOMAIN=""
HWG_UNCORE_HW_MIN=""
HWG_UNCORE_HW_MAX=""

# $1 = physical package id. Must be called BEFORE the caller captures its own ORIG_* values,
# so that what it captures is the hardware default rather than a previous run's leak.
hwguard_uncore_init() {
  local pkg=$1 d cur_min cur_max
  d=/sys/devices/system/cpu/intel_uncore_frequency/$(printf 'package_%02d_die_00' "$pkg")
  [ -d "$d" ] || {
    echo "[hwguard] no intel_uncore_frequency domain for package $pkg -- MSR path in use,"
    echo "[hwguard] so neither the floor nor leftover state can be checked. Ratios unverified."
    return 0; }
  HWG_UNCORE_DOMAIN=$d
  HWG_UNCORE_HW_MIN=$( [ -r "$d/initial_min_freq_khz" ] && cat "$d/initial_min_freq_khz" || cat "$d/min_freq_khz" )
  HWG_UNCORE_HW_MAX=$( [ -r "$d/initial_max_freq_khz" ] && cat "$d/initial_max_freq_khz" || cat "$d/max_freq_khz" )
  cur_min=$(< "$d/min_freq_khz"); cur_max=$(< "$d/max_freq_khz")
  echo "[hwguard] uncore pkg$pkg hardware range ${HWG_UNCORE_HW_MIN}-${HWG_UNCORE_HW_MAX} kHz" \
       "(ratio $((HWG_UNCORE_HW_MIN / 100000))..$((HWG_UNCORE_HW_MAX / 100000)))"
  if [ "$cur_min" != "$HWG_UNCORE_HW_MIN" ] || [ "$cur_max" != "$HWG_UNCORE_HW_MAX" ]; then
    echo "[hwguard] LEFTOVER STATE: found ${cur_min}-${cur_max} kHz, not the hardware default."
    echo "[hwguard] A previous run did not restore. Resetting before anything is measured."
    echo "$HWG_UNCORE_HW_MIN" | sudo tee "$d/min_freq_khz" >/dev/null
    echo "$HWG_UNCORE_HW_MAX" | sudo tee "$d/max_freq_khz" >/dev/null
    echo "[hwguard] uncore reset to ${HWG_UNCORE_HW_MIN}-${HWG_UNCORE_HW_MAX} kHz"
  fi
}

# $1 = the whole ratio list. Checked up front so a bad point fails in the first second rather
# than after an hour of sweeping, and so it can never reach the archive under a false label.
hwguard_require_ratios() {
  [ -n "$HWG_UNCORE_DOMAIN" ] || return 0
  local lo=$((HWG_UNCORE_HW_MIN / 100000)) hi=$((HWG_UNCORE_HW_MAX / 100000)) r bad=""
  for r in $1; do
    [ "$r" -ge "$lo" ] && [ "$r" -le "$hi" ] || bad+=" $r"
  done
  [ -z "$bad" ] && { echo "[hwguard] ratios '$1' all within [$lo,$hi]"; return 0; }
  echo "[hwguard] REFUSING TO RUN: ratio(s)$bad fall outside this package's [$lo,$hi]."
  echo "[hwguard] The driver clamps out-of-range requests silently, so each one would duplicate"
  echo "[hwguard] the nearest in-range point and be archived under a clock the hardware never ran."
  return 1
}

# $1 = the ratio just applied. Hard verification, for use right after a set_ratio.
hwguard_uncore_verify() {
  [ -n "$HWG_UNCORE_DOMAIN" ] || return 0
  local khz=$(( $1 * 100000 )) gmin gmax
  gmin=$(< "$HWG_UNCORE_DOMAIN/min_freq_khz"); gmax=$(< "$HWG_UNCORE_DOMAIN/max_freq_khz")
  [ "$gmin" = "$khz" ] && [ "$gmax" = "$khz" ] && return 0
  echo "[hwguard] ASSERT FAILED: ratio $1 asked for $khz kHz, hardware reports min=$gmin max=$gmax"
  return 1
}

# --- what every caller puts in its EXIT trap -------------------------------------------------
hwguard_restore_all() { hwguard_numa_balancing_restore; }

# --- provenance: what the numbers were actually taken on --------------------------------------
# $1 = file to append to. The archived run directory should carry proof of its own config.
hwguard_record_config() {
  local f=$1
  {
    echo "=== hwguard config record ==="
    echo "host=$(hostname)  date=$(date -Is)  kernel=$(uname -r)"
    echo "numa_balancing=$(cat /proc/sys/kernel/numa_balancing 2>/dev/null || echo n/a)"
    if [ -n "$HWG_UNCORE_DOMAIN" ]; then
      echo "uncore_domain=$HWG_UNCORE_DOMAIN"
      echo "uncore_hw_range=${HWG_UNCORE_HW_MIN}-${HWG_UNCORE_HW_MAX} kHz"
      echo "uncore_now=$(< "$HWG_UNCORE_DOMAIN/min_freq_khz")-$(< "$HWG_UNCORE_DOMAIN/max_freq_khz") kHz"
    else
      echo "uncore_domain=none (MSR path)"
    fi
    echo "thp=$(cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || echo n/a)"
    numactl -H 2>/dev/null | grep -E "^node [0-9]+ (size|free)|^node distances|^ +[0-9]+:"
  } >> "$f"
}
