import json
import subprocess
import requests

# --- Configuration ---
RUNAI_BASE_URL = "https://sdsc.run.ai"
CLIENT_ID = "41e26704d2b040fb"
CLIENT_SECRET = "AwXK9la6kIlIPoxnWDAgTjsDH7ygkfh9"  # <-- Paste your secret key here

# Safety toggle: True will only print out what it plans to do. 
# Change to False to actually execute the restarts via your terminal.
DRY_RUN = False

JOB_NAME_PREFIX = "gp-manifold-artiom-260608"


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
    """Fetches ALL workloads across all pages using the REST API."""
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


def main():
    try:
        print("Authenticating with Run:ai API...")
        token = get_access_token()

        print("Scanning comprehensive API workload catalog...")
        workloads = fetch_all_workloads(token)

        failed_jobs = []
        for wl in workloads:
            # Filter specifically for failed jobs matching your targeted prefix
            if wl.get("name", "").startswith(JOB_NAME_PREFIX) and wl.get("phase", "").upper() == "FAILED":
                failed_jobs.append(wl)

        print(f"\nFound {len(failed_jobs)} failed workloads matching prefix '{JOB_NAME_PREFIX}'.")
        if DRY_RUN:
            print("!!! Script is running in DRY-RUN mode. No jobs will be submitted. !!!\n")

        restart_count = 0

        for job in failed_jobs:
            job_name = job.get("name")
            orig_cmd = job.get("additionalFields", {}).get("cliCommand")

            if not orig_cmd:
                print(f"[-] Missing original submit command details for {job_name}. Skipping.")
                continue

            # Modify the string to add '-retry' right after the original job name
            parts = orig_cmd.split(" ")
            try:
                submit_idx = parts.index("submit")
                name_idx = submit_idx + 1
                parts[name_idx] = f"{parts[name_idx]}-retry"
                retry_cmd_str = " ".join(parts)
            except ValueError:
                print(f"[-] Unexpected command layout for {job_name}. Skipping.")
                continue

            if DRY_RUN:
                print(f"[DRY-RUN] Would restart: {job_name}")
                print(f"          Command: {retry_cmd_str}\n")
                restart_count += 1
            else:
                print(f"[+] Restarting {job_name} as {job_name}-retry...")
                try:
                    # Execute using your local shell session privileges
                    result = subprocess.run(retry_cmd_str, shell=True, capture_output=True, text=True, check=True)
                    print(f"    Success: {result.stdout.strip()}")
                    restart_count += 1
                except subprocess.CalledProcessError as err:
                    print(f"    [ERROR] Failed to run command for {job_name}: {err.stderr.strip()}")

        print("\n" + "="*60)
        action_word = "Identified" if DRY_RUN else "Successfully restarted"
        print(f"Execution Complete. {action_word} {restart_count} workloads.")
        print("="*60)

    except Exception as e:
        print(f"\nAn error occurred during execution: {e}")


if __name__ == "__main__":
    main()