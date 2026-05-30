#!/usr/bin/env bash
# Utilities for allocating a truly-free TCP port for `accelerate launch`.
#
# Problem this solves
# -------------------
# The previous inference scripts used
#   --main_process_port $((30000 + RANDOM % 10000))
# which:
#   1. Falls ENTIRELY inside the kernel ephemeral port range (32768-60999
#      on most Linux boxes) — so the OS is free to hand that port to some
#      random outgoing TCP connection between the time we pick it and
#      the time accelerate binds it. Race -> EADDRINUSE on a "clean" box.
#   2. Has no retry. A single collision kills the run.
#   3. Bash $RANDOM has poor entropy when multiple shells start close in
#      time (seeded from PID+time) -> reruns can pick the SAME "random"
#      port they just failed on.
#
# Fix
# ---
# * Pick from 20000-29999 (OUTSIDE typical ephemeral range, rarely used by
#   system services).
# * Pre-filter candidates against `ss -tln` (drop ports currently LISTEN-ing
#   in our pool) + user-supplied PORT_DENYLIST — O(1) reject before any
#   bind test.
# * Actually verify the port is bindable via python3 (kernel-level check),
#   not just a blind `ss -tln | grep` which races the same way.
# * Retry up to MAX_PORT_TRIES times.
# * Sweep current user's own orphan/zombie accelerate launchers before
#   launching (cleanup_my_orphans). Only ppid=1 or stat=Z processes that
#   belong to this user AND look like accelerate/torchrun cmdlines — never
#   kills anyone else's process, never kills based on port ownership.
# * Expose `run_with_port_retry` that re-runs the launch command with a
#   fresh port if accelerate still hits EADDRINUSE (rare, but possible
#   between our bind test and the actual launch).
#
# Why we do NOT "kill whoever holds the busy port"
# ------------------------------------------------
# EADDRINUSE in this environment is most often caused by (a) the kernel
# handing out an ephemeral port to some outbound TCP connection, (b) other
# users' or system services' long-running sockets (no permission to kill),
# (c) our own zombie torchrun workers. Only (c) is safely killable, and we
# target it by process identity (user+cmdline+orphan), NOT by "whatever PID
# happens to hold port N". Killing by port would risk wrecking other users'
# jobs or system services. Picking a different port is always faster and
# safer than trying to free the busy one.
#
# Usage (in inference_*.sh):
#   source "$(dirname "${BASH_SOURCE[0]}")/../../scripts/_port_utils.sh"
#   run_with_port_retry accelerate launch --num_processes=8 -m lmms_eval ...
#
# You can still force a port explicitly:
#   MAIN_PROCESS_PORT=34567 bash examples/v_cast/inference_qwen3vl_v_cast_32.sh
#
# Opt-out switches:
#   PORT_UTILS_AUTO_CLEANUP=0   disable orphan cleanup
#   PORT_DENYLIST="P1 P2 ..."   force-skip extra ports

# Port pool: 20000-29999 is outside the default Linux ephemeral range
# (32768-60999) and far from registered service ports. Keep this config
# here so all inference scripts use the same range.
PORT_POOL_MIN="${PORT_POOL_MIN:-20000}"
PORT_POOL_MAX="${PORT_POOL_MAX:-29999}"
MAX_PORT_TRIES="${MAX_PORT_TRIES:-30}"
MAX_LAUNCH_RETRIES="${MAX_LAUNCH_RETRIES:-5}"

# Extra ports to avoid (space-separated). Caller may append known-bad ports
# here. Example:  PORT_DENYLIST="23456 24680" bash ...
PORT_DENYLIST="${PORT_DENYLIST:-}"

# Whether to auto-sweep user-owned orphan/zombie accelerate/torchrun
# processes at the start of run_with_port_retry. On by default because
# leaked launchers are the most common cause of "can't bind port" on
# reruns. Set to 0 to disable (useful in CI where touching processes is
# undesired). Independent of port allocation — no port is ever "killed".
PORT_UTILS_AUTO_CLEANUP="${PORT_UTILS_AUTO_CLEANUP:-1}"

# --- Low-level: ask the kernel if a port is bindable right now --------------
# Returns 0 if bindable, non-zero otherwise. Uses python3 because `nc -l` /
# `ss` are either unreliable or race-prone.
_port_is_free() {
  local port="$1"
  python3 - "$port" <<'PY' 2>/dev/null
import socket, sys
p = int(sys.argv[1])
# SO_REUSEADDR=1 mirrors what torch.distributed.TCPStore uses; if even that
# bind fails, the port is genuinely unusable.
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    s.bind(("", p))
except OSError:
    sys.exit(1)
finally:
    s.close()
sys.exit(0)
PY
}

# --- Snapshot of currently LISTEN-ing ports (in our pool) -------------------
# Uses `ss -tln` (no -p, so no root needed). Cheap, but can be stale by the
# time we actually bind — that's why _port_is_free still does a real bind.
# We use it only as an O(1) pre-filter to skip obviously-busy ports.
_collect_listen_ports() {
  local min="$1" max="$2"
  # Output format from `ss -Htln`:
  #   LISTEN 0 4096 0.0.0.0:8823 0.0.0.0:*
  # We want the port after the last ':' in col 4.
  ss -Htln 2>/dev/null \
    | awk -v mn="$min" -v mx="$max" '{
        n = split($4, a, ":");
        p = a[n] + 0;
        if (p >= mn && p <= mx) print p;
      }' | sort -u
}

# --- Find a free port in our pool -------------------------------------------
# Strategy:
#   1. Build a denylist = LISTEN-ing ports in our pool + user-specified
#      PORT_DENYLIST. Treat these as "don't even bother".
#   2. Try up to MAX_PORT_TRIES random ports, skipping the denylist.
#   3. Verify each candidate with _port_is_free (real kernel bind test).
#   4. Fallback: bind(0) — kernel-chosen free port, guaranteed at the
#      moment of return (tiny race window until accelerate binds it, hence
#      run_with_port_retry wraps the whole launch).
find_free_port() {
  local span=$((PORT_POOL_MAX - PORT_POOL_MIN + 1))

  # Build denylist once per call. Cheap (~1 ms).
  local -A deny=()
  local p
  for p in $(_collect_listen_ports "${PORT_POOL_MIN}" "${PORT_POOL_MAX}"); do
    deny[$p]=1
  done
  for p in ${PORT_DENYLIST}; do
    deny[$p]=1
  done

  local tries=0
  while (( tries < MAX_PORT_TRIES )); do
    local candidate=$(( PORT_POOL_MIN + RANDOM % span ))
    if [[ -n "${deny[$candidate]:-}" ]]; then
      tries=$((tries + 1))
      continue
    fi
    if _port_is_free "${candidate}"; then
      echo "${candidate}"
      return 0
    fi
    # Remember this one failed the real bind test too.
    deny[$candidate]=1
    tries=$((tries + 1))
  done

  echo "[_port_utils] WARN: ${MAX_PORT_TRIES} random picks in " \
       "${PORT_POOL_MIN}-${PORT_POOL_MAX} all busy; falling back to kernel bind(0)" >&2
  python3 - <<'PY'
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(("", 0))
print(s.getsockname()[1])
s.close()
PY
}

# --- Safe orphan cleanup for user-owned dead launchers ----------------------
# "Safe" means:
#   * ONLY inspects processes owned by the current user (EUID match).
#   * ONLY targets cmdlines that look like our launchers:
#       accelerate launch / torchrun / torch.distributed.run
#   * ONLY kills processes whose parent PID == 1 (true orphan — its launcher
#     shell already exited), or whose state is Z (defunct / zombie).
#   * Skips processes that belong to the current shell's own process group,
#     so we never suicide the batch script mid-run.
#   * First sends SIGTERM, waits 2 s, then SIGKILL survivors.
#
# This is NOT "kill whoever is holding a port". That would be unsafe
# (other users, system services, torch worker siblings, transient kernel
# ephemeral sockets). We only reap our OWN leaked launchers.
cleanup_my_orphans() {
  local my_uid
  my_uid="$(id -u)"
  local my_pgid
  my_pgid="$(ps -o pgid= -p $$ | tr -d ' ')"

  # ps columns: pid ppid pgid state user cmd
  # -e: all; -o: custom fields; --no-headers to simplify.
  local candidates
  candidates="$(
    ps -e -o pid=,ppid=,pgid=,stat=,euid=,cmd= 2>/dev/null \
      | awk -v u="${my_uid}" -v mypg="${my_pgid}" '
          {
            pid=$1; ppid=$2; pgid=$3; stat=$4; euid=$5;
            cmd="";
            for (i=6; i<=NF; i++) { cmd = cmd (i==6 ? "" : " ") $i }

            if (euid != u) next;                  # not mine
            if (pgid == mypg) next;               # same process group as me
            if (pid == PROCINFO["ppid"]) next;    # my own parent

            # match our launcher cmdlines
            if (cmd !~ /(accelerate[[:space:]]+launch|torchrun|torch\.distributed\.run|torch\.distributed\.launch)/) next;

            # only orphans (ppid==1) or zombies (stat starts with Z)
            is_orphan = (ppid == 1);
            is_zombie = (substr(stat,1,1) == "Z");
            if (!is_orphan && !is_zombie) next;

            print pid, ppid, stat, cmd;
          }
        '
  )"

  if [[ -z "${candidates}" ]]; then
    return 0
  fi

  echo "[_port_utils] found user-owned orphan/zombie launchers, cleaning up:" >&2
  while IFS= read -r line; do
    [[ -z "${line}" ]] && continue
    local pid
    pid="$(echo "${line}" | awk '{print $1}')"
    echo "  - killing pid=${pid}: ${line}" >&2
    kill -TERM "${pid}" 2>/dev/null || true
  done <<< "${candidates}"

  # Give them 2 s to exit gracefully, then SIGKILL any survivors.
  sleep 2
  while IFS= read -r line; do
    [[ -z "${line}" ]] && continue
    local pid
    pid="$(echo "${line}" | awk '{print $1}')"
    if kill -0 "${pid}" 2>/dev/null; then
      echo "  - SIGKILL pid=${pid} (did not exit on SIGTERM)" >&2
      kill -KILL "${pid}" 2>/dev/null || true
    fi
  done <<< "${candidates}"
}

# --- Run a command with auto-retry on EADDRINUSE ----------------------------
# Injects --main_process_port <free port> into the accelerate command.
# If accelerate exits with a port-related error, we pick a new port and
# retry up to MAX_LAUNCH_RETRIES times.
#
# Respects an externally-set MAIN_PROCESS_PORT (skips auto-allocation and
# retry in that case — if you pin the port, you own the consequences).
run_with_port_retry() {
  if [[ "${PORT_UTILS_AUTO_CLEANUP}" == "1" ]]; then
    cleanup_my_orphans
  fi

  if [[ -n "${MAIN_PROCESS_PORT:-}" ]]; then
    echo "[_port_utils] Using externally pinned MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT}"
    _inject_port_and_run "${MAIN_PROCESS_PORT}" "$@"
    return $?
  fi

  local attempt=1
  while (( attempt <= MAX_LAUNCH_RETRIES )); do
    local port
    port=$(find_free_port) || {
      echo "[_port_utils] ERROR: could not find a free port" >&2
      return 1
    }
    echo "[_port_utils] attempt ${attempt}/${MAX_LAUNCH_RETRIES}: main_process_port=${port}"

    local log
    log=$(mktemp)
    if _inject_port_and_run "${port}" "$@" 2> >(tee "${log}" >&2); then
      rm -f "${log}"
      return 0
    fi
    local rc=$?

    if grep -qE 'EADDRINUSE|address already in use|port.*already.*in use' "${log}"; then
      echo "[_port_utils] port ${port} collided (EADDRINUSE), retrying..." >&2
      rm -f "${log}"
      attempt=$((attempt + 1))
      continue
    fi
    rm -f "${log}"
    return "${rc}"
  done

  echo "[_port_utils] ERROR: exhausted ${MAX_LAUNCH_RETRIES} launch retries" >&2
  return 1
}

# Internal: replace/inject --main_process_port <port> in the arg list and exec.
#
# If the caller already has --main_process_port (or =form), we REPLACE the
# value. Otherwise, we INSERT --main_process_port <port> right AFTER the
# `launch` subcommand (e.g. `accelerate launch`) — putting it before `launch`
# would make accelerate's CLI parser mistake the number for a subcommand.
# If we can't find `launch`, we fall back to appending at the end.
_inject_port_and_run() {
  local port="$1"; shift
  local -a new_args=()
  local skip_next=0
  local injected=0
  for arg in "$@"; do
    if (( skip_next )); then skip_next=0; continue; fi
    case "$arg" in
      --main_process_port)       skip_next=1;  new_args+=("--main_process_port" "${port}"); injected=1 ;;
      --main_process_port=*)     new_args+=("--main_process_port=${port}");                 injected=1 ;;
      *)                         new_args+=("$arg") ;;
    esac
  done

  if (( ! injected )); then
    local -a rebuilt=()
    local inserted=0
    for a in "${new_args[@]}"; do
      rebuilt+=("$a")
      if (( ! inserted )) && [[ "$a" == "launch" ]]; then
        rebuilt+=("--main_process_port" "${port}")
        inserted=1
      fi
    done
    if (( ! inserted )); then
      rebuilt+=("--main_process_port" "${port}")
    fi
    new_args=("${rebuilt[@]}")
  fi
  "${new_args[@]}"
}
