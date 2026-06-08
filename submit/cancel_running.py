import requests

RUNAI_BASE_URL = "https://sdsc.run.ai"
CLIENT_ID = "41e26704d2b040fb"
CLIENT_SECRET = "AwXK9la6kIlIPoxnWDAgTjsDH7ygkfh9"  # <-- Paste your secret key here

JOB_NAME_PREFIX = "lap-260605"

# Safety toggle: True will only show what WOULD be deleted. Change to False to actually execute.
DRY_RUN = False

# Target criteria
FILTER_CRITERIA = {
    "CROSS_REGION_INFLATION": "100",
    "GRAPHBANDWIDTH": "0.05"
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

    while True:
        params = {"limit": limit, "offset": offset}
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()

        data = response.json()
        page_workloads = data.get("workloads", [])

        if not page_workloads:
            break

        all_workloads.extend(page_workloads)

        if "next" in data and data["next"] is not None:
            offset = data["next"]
        else:
            offset += limit

    return all_workloads


def delete_workload(token, workload_id, workload_name):
    """Sends a DELETE request to cancel a specific workload."""
    url = f"{RUNAI_BASE_URL}/api/v1/workloads/{workload_id}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }
    
    if DRY_RUN:
        print(f"[DRY-RUN] Would delete workload: {workload_name} (ID: {workload_id})")
        return True
        
    try:
        response = requests.delete(url, headers=headers)
        if response.status_code in [200, 202, 204]:
            print(f"Successfully requested deletion for: {workload_name}")
            return True
        else:
            print(f"Failed to delete {workload_name}: API returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"Error communicating with API for {workload_name}: {e}")
        return False


def main():
    try:
        print("Authenticating with Run:ai API...")
        token = get_access_token()

        print("Scanning workload manifest...")
        workloads = fetch_all_workloads(token)

        matching_workloads = []

        for wl in workloads:
            # Match the target name prefix
            job_name = wl.get("name", "")
            if not job_name.startswith(JOB_NAME_PREFIX):
                continue

            # We only care about active/cancellable jobs (Pending, Running)
            phase = wl.get("phase", "").upper()
            if phase in ["COMPLETED", "FAILED", "STOPPED"]:
                continue

            # Validate environment parameters
            env_dict = wl.get("environmentVariables", {}) or {}
            
            match = True
            for key, expected_value in FILTER_CRITERIA.items():
                if env_dict.get(key) != expected_value:
                    match = False
                    break
            
            if match:
                matching_workloads.append(wl)

        print(f"\nFound {len(matching_workloads)} active/pending workloads matching target configurations.")
        if DRY_RUN:
            print("!!! Script is currently in DRY-RUN mode. No actual deletions will occur. !!!\n")

        # Execute cancellation loop
        success_count = 0
        for target in matching_workloads:
            wl_id = target.get("id")
            wl_name = target.get("name")
            
            if delete_workload(token, wl_id, wl_name):
                success_count += 1

        if not DRY_RUN:
            print(f"\nAction complete. Successfully targeted {success_count} workloads for deletion.")

    except Exception as e:
        print(f"\nAn error occurred during workflow: {e}")


if __name__ == "__main__":
    main()
