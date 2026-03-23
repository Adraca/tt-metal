#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
import logging
import os
import sys
import glob
from typing import List, Dict, Tuple, Optional, Union


@dataclass
class KernelZoneEvent:
    """POD for kernel zone trace event."""

    device_id: int
    core_x: int
    core_y: int
    risc_type: str
    zone_name: str
    start_cycles: int
    end_cycles: int


@dataclass
class TensixCore:
    """POD for Tensix core profiling data."""

    device_id: int
    core_x: int
    core_y: int
    duration_cycles: int
    duration_ns: float


@dataclass
class OpsPerfData:
    """POD for ops perf results data."""

    fpu_util: float
    device_kernel_duration_ns: float


@dataclass
class DeviceStats:
    """POD for device statistics."""

    cores: int
    min: float
    max: float
    avg: float
    p50: float
    p75: float
    p90: float
    p99: float
    compute_balance: float
    fpu_util: float
    device_kernel_duration_ns: float


def percentile(data: List[float], p: float) -> float:
    """Calculate the p-th percentile of data."""
    sorted_data = sorted(data)
    n = len(sorted_data)
    if n == 0:
        return 0
    k = (n - 1) * p / 100
    f = int(k)
    c = f + 1
    if c >= n:
        return sorted_data[-1]
    if f == k:
        return sorted_data[f]
    # Linear interpolation
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def validate_tracy_directory(tracy_dir: str) -> None:
    """Validate that the Tracy directory exists."""
    if not os.path.isdir(tracy_dir):
        logging.error(f"Tracy directory not found: {tracy_dir}")
        sys.exit(1)


def find_ops_perf_results(tracy_dir: str) -> str:
    """Find and return the ops_perf_results CSV file path."""
    ops_perf_files = glob.glob(os.path.join(tracy_dir, "ops_perf_results_*.csv"))
    if not ops_perf_files:
        logging.error(f"No ops_perf_results CSV file found in {tracy_dir}")
        sys.exit(1)
    return ops_perf_files[0]


def extract_report_date(ops_perf_file: str) -> str:
    """Extract report date from ops_perf_results filename."""
    basename = os.path.basename(ops_perf_file)
    report_timestamp = basename.replace("ops_perf_results_", "").replace(".csv", "")
    # Convert to readable format: YYYY-MM-DD HH:MM:SS
    return f"{report_timestamp[:4]}-{report_timestamp[5:7]}-{report_timestamp[8:10]} {report_timestamp[11:13]}:{report_timestamp[14:16]}:{report_timestamp[17:19]}"


def parse_ops_perf_data(ops_perf_file: str) -> Dict[int, OpsPerfData]:
    """Parse ops perf results data from ops_perf_results CSV."""
    ops_perf_data: Dict[int, OpsPerfData] = {}
    with open(ops_perf_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            device_id = int(row["DEVICE ID"].strip())

            # Parse FPU utilization
            fpu_util_str = row.get("PM FPU UTIL (%)", "").strip()
            if not fpu_util_str:
                fpu_util_str = row.get("Avg FPU util on full grid (%)", "").strip()
            fpu_util = float(fpu_util_str) if fpu_util_str else 0.0

            # Parse device kernel duration
            device_kernel_duration_str = row.get("DEVICE KERNEL DURATION [ns]", "").strip()
            device_kernel_duration_ns = float(device_kernel_duration_str) if device_kernel_duration_str else 0.0

            ops_perf_data[device_id] = OpsPerfData(
                fpu_util=fpu_util, device_kernel_duration_ns=device_kernel_duration_ns
            )

    return ops_perf_data


def parse_trace_events(tracy_dir: str) -> Tuple[List[KernelZoneEvent], int]:
    """Parse all trace events from profile_log_device.csv and return list of KernelZoneEvent objects and chip frequency."""
    ZONE_START_TAG = "ZONE_START"
    ZONE_END_TAG = "ZONE_END"

    profile_log_path = os.path.join(tracy_dir, "profile_log_device.csv")
    if not os.path.exists(profile_log_path):
        logging.error(f"profile_log_device.csv not found in {tracy_dir}")
        sys.exit(1)

    # Temporary storage for pairing start/end events
    event_pairs: Dict[Tuple[int, int, int, str, str], Dict[str, int]] = {}
    chip_freq_mhz: int = 0

    with open(profile_log_path, "r") as f:
        lines = f.readlines()

        # Parse metadata from first line
        metadata = lines[0].strip()
        for part in metadata.split(","):
            if "CHIP_FREQ[MHz]" in part:
                chip_freq_mhz = int(part.split(":")[1].strip())

        # Parse CSV data starting from line 2
        reader = csv.DictReader(lines[1:])

        for row in reader:
            # Strip whitespace from column names
            row = {k.strip(): v.strip() for k, v in row.items()}

            device = int(row["PCIe slot"])
            core_x = int(row["core_x"])
            core_y = int(row["core_y"])
            risc_type = row["RISC processor type"]
            zone_name = row["zone name"]
            zone_type = row["type"]
            time_cycles = int(row["time[cycles since reset]"])

            # Pair start/end events for all zones
            event_key = (device, core_x, core_y, risc_type, zone_name)
            if event_key not in event_pairs:
                event_pairs[event_key] = {}

            if zone_type == ZONE_START_TAG:
                event_pairs[event_key]["start"] = time_cycles
            elif zone_type == ZONE_END_TAG:
                event_pairs[event_key]["end"] = time_cycles

    # Convert paired events to KernelZoneEvent objects
    kernel_events: List[KernelZoneEvent] = []
    for (device_id, core_x, core_y, risc_type, zone_name), event_data in event_pairs.items():
        if "start" in event_data and "end" in event_data:
            kernel_events.append(
                KernelZoneEvent(
                    device_id=device_id,
                    core_x=core_x,
                    core_y=core_y,
                    risc_type=risc_type,
                    zone_name=zone_name,
                    start_cycles=event_data["start"],
                    end_cycles=event_data["end"],
                )
            )

    return kernel_events, chip_freq_mhz


def calculate_core_durations(kernel_events: List[KernelZoneEvent], chip_freq_mhz: int) -> List[TensixCore]:
    """Calculate kernel durations per core from KernelZoneEvent objects.

    Filters for BRISC-KERNEL events only and returns list of TensixCore objects.
    """
    KERNEL_ZONE_NAME = "BRISC-KERNEL"
    RISC_TYPE = "BRISC"

    def cycles_to_ns(cycles: int) -> float:
        return (cycles / chip_freq_mhz) * 1000

    tensix_cores: List[TensixCore] = []

    for event in kernel_events:
        # Filter for BRISC-KERNEL zones only
        if event.zone_name == KERNEL_ZONE_NAME and event.risc_type == RISC_TYPE:
            duration_cycles = event.end_cycles - event.start_cycles
            duration_ns = cycles_to_ns(duration_cycles)

            if duration_cycles > 0:
                tensix_cores.append(
                    TensixCore(
                        device_id=event.device_id,
                        core_x=event.core_x,
                        core_y=event.core_y,
                        duration_cycles=duration_cycles,
                        duration_ns=duration_ns,
                    )
                )

    return tensix_cores


def generate_device_report(
    device: int, cores: List[TensixCore], ops_perf_data: Dict[int, OpsPerfData]
) -> Tuple[List[str], Optional[DeviceStats]]:
    """Generate analysis report for a single device.

    Args:
        device: Device ID (int)
        cores: List of TensixCore objects for this device
        ops_perf_data: Dict mapping device_id to ops perf data
    """
    device_report: List[str] = []
    device_report.append(f"## Device {device}\n\n")

    if not cores:
        device_report.append("No kernel data found.\n\n")
        return device_report, None

    # Determine grid dimensions
    x_coords = sorted(set(c.core_x for c in cores))
    y_coords = sorted(set(c.core_y for c in cores))

    # Create grid lookup
    grid: Dict[Tuple[int, int], float] = {}
    for core in cores:
        grid[(core.core_x, core.core_y)] = core.duration_ns

    # Generate table
    device_report.append("### Kernel Duration Per Core (milliseconds)\n\n")

    # Header row
    header = "| Y\\X |"
    for x in x_coords:
        header += f" {x:2d} |"
    device_report.append(header + "\n")

    # Separator
    separator = "|-----|"
    for _ in x_coords:
        separator += "--------|"
    device_report.append(separator + "\n")

    # Data rows
    for y in y_coords:
        row = f"| {y:2d}  |"
        for x in x_coords:
            if (x, y) in grid:
                duration_ms = grid[(x, y)] / 1_000_000  # Convert ns to ms
                row += f" {duration_ms:6.2f} |"
            else:
                row += "      - |"
        device_report.append(row + "\n")

    # Statistics
    durations_ms: List[float] = [c.duration_ns / 1_000_000 for c in cores]  # Convert ns to ms

    # Calculate compute balance: sum(duration) / (#cores * max(duration))
    num_cores: int = len(durations_ms)
    max_duration: float = max(durations_ms)
    sum_duration: float = sum(durations_ms)
    compute_balance: float = (sum_duration / (num_cores * max_duration)) * 100  # as percentage

    # Get ops perf data from ops_perf_results
    ops_perf = ops_perf_data.get(device, OpsPerfData(fpu_util=0.0, device_kernel_duration_ns=0.0))

    # Create statistics object
    stats = DeviceStats(
        cores=num_cores,
        min=min(durations_ms),
        max=max_duration,
        avg=sum(durations_ms) / len(durations_ms),
        p50=percentile(durations_ms, 50),
        p75=percentile(durations_ms, 75),
        p90=percentile(durations_ms, 90),
        p99=percentile(durations_ms, 99),
        compute_balance=compute_balance,
        fpu_util=ops_perf.fpu_util,
        device_kernel_duration_ns=ops_perf.device_kernel_duration_ns,
    )

    device_report.append(f"\n**Statistics:**\n")
    device_report.append(f"- Total Cores: {stats.cores}\n")
    device_report.append(f"- Min Duration: {stats.min:.2f} ms\n")
    device_report.append(f"- Max Duration: {stats.max:.2f} ms\n")
    device_report.append(f"- Avg Duration: {stats.avg:.2f} ms\n")
    device_report.append(f"- P50 (Median): {stats.p50:.2f} ms\n")
    device_report.append(f"- P75: {stats.p75:.2f} ms\n")
    device_report.append(f"- P90: {stats.p90:.2f} ms\n")
    device_report.append(f"- P99: {stats.p99:.2f} ms\n")
    device_report.append(f"- Compute Balance: {stats.compute_balance:.2f}%\n")
    device_report.append(f"- FPU Utilization: {stats.fpu_util:.2f}%\n")
    device_report.append(f"- Device Kernel Duration: {stats.device_kernel_duration_ns / 1_000_000:.2f} ms\n")
    device_report.append("\n")

    return device_report, stats


def generate_summary_table(device_stats: Dict[int, DeviceStats]) -> List[str]:
    """Generate cross-device comparison summary table."""
    summary: List[str] = []
    summary.append("## Summary: Cross-Device Comparison\n\n")
    summary.append("| Metric | Device 0 | Device 1 | Device 2 | Device 3 |\n")
    summary.append("|--------|----------|----------|----------|----------|\n")

    devices_sorted = sorted(device_stats.keys())

    # Total Cores
    row = "| Total Cores |"
    for dev in devices_sorted:
        row += f" {device_stats[dev].cores} |"
    summary.append(row + "\n")

    # Min Duration
    row = "| Min (ms) |"
    for dev in devices_sorted:
        row += f" {device_stats[dev].min:.2f} |"
    summary.append(row + "\n")

    # Max Duration
    row = "| Max (ms) |"
    for dev in devices_sorted:
        row += f" {device_stats[dev].max:.2f} |"
    summary.append(row + "\n")

    # Avg Duration
    row = "| Avg (ms) |"
    for dev in devices_sorted:
        row += f" {device_stats[dev].avg:.2f} |"
    summary.append(row + "\n")

    # P50
    row = "| P50 (ms) |"
    for dev in devices_sorted:
        row += f" {device_stats[dev].p50:.2f} |"
    summary.append(row + "\n")

    # P75
    row = "| P75 (ms) |"
    for dev in devices_sorted:
        row += f" {device_stats[dev].p75:.2f} |"
    summary.append(row + "\n")

    # P90
    row = "| P90 (ms) |"
    for dev in devices_sorted:
        row += f" {device_stats[dev].p90:.2f} |"
    summary.append(row + "\n")

    # P99
    row = "| P99 (ms) |"
    for dev in devices_sorted:
        row += f" {device_stats[dev].p99:.2f} |"
    summary.append(row + "\n")

    # Compute Balance
    row = "| Compute Balance (%) |"
    for dev in devices_sorted:
        row += f" {device_stats[dev].compute_balance:.2f} |"
    summary.append(row + "\n")

    # FPU Utilization
    row = "| FPU Util (%) |"
    for dev in devices_sorted:
        row += f" {device_stats[dev].fpu_util:.2f} |"
    summary.append(row + "\n")

    # Device Kernel Duration
    row = "| Device Kernel Duration (ms) |"
    for dev in devices_sorted:
        row += f" {device_stats[dev].device_kernel_duration_ns / 1_000_000:.2f} |"
    summary.append(row + "\n\n")

    return summary


def write_report(
    output_path: str,
    report_date: str,
    chip_freq_mhz: int,
    device_stats: Dict[int, DeviceStats],
    device_reports: List[List[str]],
) -> None:
    """Write the complete analysis report to file."""
    report: List[str] = []
    report.append("# Tracy Profiling Analysis\n")
    report.append(f"**Report Date:** {report_date}\n")
    report.append(f"**Architecture:** Blackhole\n")
    report.append(f"**Chip Frequency:** {chip_freq_mhz} MHz\n")
    report.append(f"**Operation:** RingJointSDPADeviceOperation\n\n")

    # Add summary table
    report.extend(generate_summary_table(device_stats))

    # Add device detail reports
    for device_report in device_reports:
        report.extend(device_report)

    # Write to file
    with open(output_path, "w") as f:
        f.writelines(report)

    logging.info(f"Analysis complete! Report written to {output_path}")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Parse Tracy profiling logs and generate analysis report")
    parser.add_argument(
        "tracy_dir", help="Directory containing Tracy profiling logs (profile_log_device.csv, ops_perf_results_*.csv)"
    )
    parser.add_argument("-o", "--output", help="Output path for analysis report (default: <tracy_dir>/analysis.md)")
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Validate and setup paths
    tracy_dir = os.path.abspath(args.tracy_dir)
    validate_tracy_directory(tracy_dir)
    output_path = os.path.abspath(args.output) if args.output else os.path.join(tracy_dir, "analysis.md")

    # Parse data
    ops_perf_file = find_ops_perf_results(tracy_dir)
    report_date = extract_report_date(ops_perf_file)
    ops_perf_data = parse_ops_perf_data(ops_perf_file)
    kernel_events, chip_freq_mhz = parse_trace_events(tracy_dir)
    tensix_cores = calculate_core_durations(kernel_events, chip_freq_mhz)

    # Group cores by device
    cores_by_device = defaultdict(list)
    for core in tensix_cores:
        cores_by_device[core.device_id].append(core)

    # Generate reports
    device_stats: Dict[int, DeviceStats] = {}
    device_reports: List[List[str]] = []
    for device in sorted(cores_by_device.keys()):
        report, stats = generate_device_report(device, cores_by_device[device], ops_perf_data)
        if stats:
            device_stats[device] = stats
            device_reports.append(report)

    # Write output
    write_report(output_path, report_date, chip_freq_mhz, device_stats, device_reports)
