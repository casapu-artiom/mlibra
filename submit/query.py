import datetime
import time
from collections import defaultdict
import requests
import numpy as np
# --- Configuration ---
RUNAI_BASE_URL = "https://sdsc.run.ai"
CLIENT_ID = "41e26704d2b040fb"
CLIENT_SECRET = "AwXK9la6kIlIPoxnWDAgTjsDH7ygkfh9"  # <-- Paste your secret key here

# Target prefix for filtering job names
JOB_NAME_PREFIX = "lap-260605"

TARGET_ENV_VARS = {
    "GRAPHBANDWIDTH",
    "KNN_METHOD",
#    "KNN_K",
#    "NORMALIZATION",
#    "THRESHOLD",
    "CROSS_REGION_INFLATION",
}


def get_access_token():
    """Exchanges Client ID and Secret for a temporary OAuth2 Bearer token."""
    url = f"{RUNAI_BASE_URL}/api/v1/token"
    payload = {
        "grantType": "client_credentials",
        "clientId": CLIENT_ID,
        "clientSecret": CLIENT_SECRET,
    }
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()["accessToken"]


def fetch_all_workloads(token):
    """Fetches ALL workloads across all pages using limit and offset parameters."""
    url = f"{RUNAI_BASE_URL}/api/v1/workloads"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }

    all_workloads = []
    limit = 100
    offset = 0

    print("Beginning paginated data retrieval...")
    while True:
        params = {"limit": limit, "offset": offset}
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()

        data = response.json()
        page_workloads = data.get("workloads", [])

        if not page_workloads:
            break

        all_workloads.extend(page_workloads)
        print(f"  Fetched records {offset} to {offset + len(page_workloads)}...")

        if "next" in data and data["next"] is not None:
            offset = data["next"]
        else:
            offset += limit

    print(f"Successfully downloaded a total of {len(all_workloads)} system records.")
    return all_workloads


def format_seconds(seconds):
    """Converts raw seconds into a human-readable HH:MM:SS format."""
    if seconds is None or np.isnan(seconds):
        return "N/A"
    return str(datetime.timedelta(seconds=int(seconds)))


def calculate_distribution_stats(durations_list):
    """Calculates Avg, P10, Median, and P90 for a list of durations."""
    if not durations_list:
        return None
    arr = np.array(durations_list)
    return {
        "avg": np.mean(arr),
        "p10": np.percentile(arr, 10),
        "median": np.median(arr),
        "p90": np.percentile(arr, 90),
        "count": len(durations_list),
    }


def print_metrics_table(title, metrics_dict):
    """Helper to cleanly print percentile blocks."""
    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)
    for config, stats in metrics_dict.items():
        if stats:
            print(f"Config: [{config}] (Sample Size: {stats['count']})")
            print(f"  -> Avg:    {format_seconds(stats['avg'])}")
            print(f"  -> p10:    {format_seconds(stats['p10'])}")
            print(f"  -> Median: {format_seconds(stats['median'])}")
            print(f"  -> p90:    {format_seconds(stats['p90'])}")
            print("-" * 50)
        else:
            print(f"Config: [{config}]\n  -> No active/applicable workloads found.")
            print("-" * 50)
    print("=" * 110)


def calculate_statistics(workloads_list):
    status_counts = defaultdict(lambda: defaultdict(int))
    
    # Raw duration collectors grouped by configuration profile
    running_durations = defaultdict(list)
    completed_durations = defaultdict(list)

    filtered_count = 0

    for wl in workloads_list:
        job_name = wl.get("name", "")
        if not job_name.startswith(JOB_NAME_PREFIX):
            continue

        filtered_count += 1
        status = wl.get("phase", "UNKNOWN")

        # Extract environment variables
        env_dict = wl.get("environmentVariables", {}) or {}
        matched_configs = []
        for var_name in TARGET_ENV_VARS:
            if var_name in env_dict:
                matched_configs.append(f"{var_name}={env_dict[var_name]}")

        config_key = ", ".join(sorted(matched_configs)) if matched_configs else "NONE"
        status_counts[config_key][status] += 1

        # Collect execution values based on status
        if status.upper() == "RUNNING":
            r_time = wl.get("totalRunningTimeSeconds")
            if r_time is not None:
                running_durations[config_key].append(r_time)

        elif status.upper() == "COMPLETED":
            r_time = wl.get("totalRunningTimeSeconds")
            if r_time is not None:
                completed_durations[config_key].append(r_time)

    # Process distributions via numpy
    running_stats = {}
    completed_stats = {}
    
    all_configs = set(status_counts.keys())
    for config in all_configs:
        running_stats[config] = calculate_distribution_stats(running_durations[config])
        completed_stats[config] = calculate_distribution_stats(completed_durations[config])

    print(f"Total workloads matching prefix '{JOB_NAME_PREFIX}': {filtered_count}")
    return status_counts, running_stats, completed_stats


def main():
    try:
        print("Authenticating with Run:ai API...")
        token = get_access_token()

        payload = fetch_all_workloads(token)
        status_counts, running_stats, completed_stats = calculate_statistics(payload)

        if not status_counts:
            print(f"\nNo jobs found matching the prefix: {JOB_NAME_PREFIX}")
            return

        # 1. Print Status Counts
        print("\n" + "=" * 110)
        print(f"WORKLOAD STATUS COUNTS (Filtered: {JOB_NAME_PREFIX}*)")
        print("=" * 110)
        for config, statuses in status_counts.items():
            status_str = ", ".join(f"{k}: {v}" for k, v in statuses.items())
            print(f"Config: [{config}]")
            print(f"  -> {status_str}")
        print("=" * 110)

        # 2. Print Running Telemetry
        print_metrics_table(
            f"RUNNING WORKLOADS: CURRENT ACTIVE DURATION DISTRIBUTION (Filtered: {JOB_NAME_PREFIX}*)",
            running_stats
        )

        # 3. Print Completed Telemetry
        print_metrics_table(
            f"COMPLETED WORKLOADS: HISTORICAL RUNTIME DISTRIBUTION (Filtered: {JOB_NAME_PREFIX}*)",
            completed_stats
        )

    except Exception as e:
        print(f"\nAn error occurred: {e}")


if __name__ == "__main__":
    main()
