#!/usr/bin/env python3
import os
import re
import json
import logging
import requests
import sys
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Add the parent directory to the path so we can import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DOWNLOAD_ROOT, DOWNLOAD_FILES_FOLDER, DOWNLOAD_METADATA_FOLDER, PARENT_FOLDER_NAME, CLIENT_NAME_FIELD

def get_token(base_url, user, pwd, vault):
    resp = session.post(
        f"{base_url}/REST/server/authenticationtokens",
        json={"Username": user, "Password": pwd, "VaultGuid": vault},
        headers={"Accept": "application/json"},
        timeout=30
    )
    resp.raise_for_status()
    return resp.json()["Value"]

def load_property_definitions(base_url, token):
    headers = {"X-Authentication": token, "Accept": "application/json"}
    resp = session.get(f"{base_url}/REST/structure/properties", headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    items = data if isinstance(data, list) else data.get("Items", [])
    return {item["ID"]: item["Name"] for item in items}

def map_properties(properties, prop_defs):
    flat = {}
    for p in properties:
        pd = p["PropertyDef"]
        name = prop_defs.get(pd, str(pd))
        tv = p.get("TypedValue", {})
        if "Lookups" in tv and isinstance(tv["Lookups"], list):
            raw = tv["Lookups"]
        elif "Lookup" in tv and isinstance(tv["Lookup"], dict):
            raw = tv["Lookup"]
        else:
            raw = tv.get("Value")
        flat[name] = {
            "id": pd,
            "dataType": tv.get("DataType"),
            "rawValue": raw,
            "displayValue": tv.get("DisplayValue")
        }
    return flat

def sanitize_filename(name: str, maxlen: int = 100):
    safe = "".join(c if c.isalnum() or c in " -_." else "_" for c in name).strip()
    return safe[:maxlen] or "untitled"

def get_client_folder_name(flat_props):
    """Get the client folder name from metadata."""
    # Try to get client name from the configured field
    client_name = None
    if CLIENT_NAME_FIELD:
        client_prop = flat_props.get(CLIENT_NAME_FIELD, {})
        client_name = client_prop.get("displayValue") if client_prop else None
    
    # If no client name found or field not configured, use object ID as fallback
    if not client_name:
        client_name = f"client_{flat_props.get('__objectID', 'unknown')}"
    
    # Sanitize the client name for folder creation
    return sanitize_filename(client_name)

def create_client_folders(parent_folder, client_name):
    """Create and return file and metadata folders for a client."""
    # Create client folder
    client_folder = os.path.join(parent_folder, client_name)
    os.makedirs(client_folder, exist_ok=True)
    
    # Create files and metadata subfolders
    files_folder = os.path.join(client_folder, DOWNLOAD_FILES_FOLDER)
    metadata_folder = os.path.join(client_folder, DOWNLOAD_METADATA_FOLDER)
    
    os.makedirs(files_folder, exist_ok=True)
    os.makedirs(metadata_folder, exist_ok=True)
    
    return files_folder, metadata_folder
#params = {"limit": "40000", "p1440": "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21", "p1021": "2215, 3190"}
def search_documents(base_url, token):
    params = {"limit": "40000", "p1416": "1,2,3,4,5,6", "p1173": "2306, 2439"}
    headers = {"X-Authentication": token, "Accept": "application/json"}
    
    logging.info(f"Searching documents with params: {params}")
    logging.info(f"Using base URL: {base_url}")
    
    resp = session.get(f"{base_url}/REST/objects.aspx", params=params, headers=headers, timeout=30)
    logging.info(f"Search response status: {resp.status_code}")
    
    resp.raise_for_status()
    
    data = resp.json()
    items_count = len(data.get("Items", []))
    logging.info(f"Found {items_count} items in search response")
    
    # Try a simple search with just a limit if no results are found
    if items_count == 0:
        logging.info("No results found with specific parameters, trying a simple search")
        simple_params = {"limit": "100"}  # Just get the first 100 documents
        resp = session.get(f"{base_url}/REST/objects.aspx", params=simple_params, headers=headers, timeout=30)
        resp.raise_for_status()
        
        data = resp.json()
        simple_items_count = len(data.get("Items", []))
        logging.info(f"Simple search found {simple_items_count} items")
        
        if simple_items_count > 0:
            return data.get("Items", [])
    
    return data.get("Items", [])

def fetch_properties_and_files(base_url, token, t, i, v):
    url = f"{base_url}/REST/objects/{t}/{i}/{v}?include=properties,files"
    headers = {"X-Authentication": token, "Accept": "application/json"}
    resp = session.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data.get("Properties", []), data.get("Files", [])

def download_pdf(base_url, token, t, i, v, fid, out_folder, title):
    url = f"{base_url}/REST/objects/{t}/{i}/{v}/files/{fid}/content"
    headers = {"X-Authentication": token}
    resp = session.get(url, headers=headers, timeout=60)
    resp.raise_for_status()
    path = os.path.join(out_folder, f"{title}.pdf")
    with open(path, "wb") as f:
        f.write(resp.content)
    logging.info(f"PDF → {path}")

def load_done_set(parent_folder):
    done = set()
    
    # Check client folders for already processed files
    if not os.path.exists(parent_folder):
        return done
    
    # Look in all client folders
    for client_folder in os.listdir(parent_folder):
        client_path = os.path.join(parent_folder, client_folder)
        if not os.path.isdir(client_path):
            continue
            
        # Check for metadata folder in this client folder
        metadata_path = os.path.join(client_path, DOWNLOAD_METADATA_FOLDER)
        if not os.path.isdir(metadata_path):
            continue
            
        # Check metadata files in this client's metadata folder
        for fn in os.listdir(metadata_path):
            if not fn.endswith('.json'):
                continue
                
            try:
                with open(os.path.join(metadata_path, fn), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    t = data.get("__objectType")
                    i = data.get("__objectID")
                    v = data.get("__version")
                    
                    if t is not None and i is not None and v is not None:
                        done.add((int(t), int(i), int(v)))
            except Exception:
                # Skip files that can't be parsed
                pass
    
    return done

def process_document(item, idx, total, base_url, token, vault_guid, prop_defs, parent_folder):
    entry = item.get("ObjVer", item)
    t, i, v = entry.get("Type"), entry.get("ID"), entry.get("Version")
    logging.info(f"[{idx}/{total}] Doc Type={t}, ID={i}, Ver={v}")
    try:
        props, files = fetch_properties_and_files(base_url, token, t, i, v)
        flat = map_properties(props, prop_defs)
        flat.update({
            "__vaultGUID": vault_guid,
            "__objectType": t,
            "__objectID": i,
            "__version": v,
            "__fileIDs": [fitem["ID"] for fitem in files if fitem.get("Extension","").lower()=="pdf"]
        })

        raw_title = (
            flat.get("Name oder Titel", {}).get("displayValue")
            or flat.get("Titel Lohnlauf", {}).get("displayValue")
            or str(i)
        )
        exact_title = raw_title
        
        # Get client name and create folders
        client_folder_name = get_client_folder_name(flat)
        files_folder, metadata_folder = create_client_folders(parent_folder, client_folder_name)
        
        # For each PDF file, save both the PDF and its matching metadata
        for fitem in files:
            if fitem.get("Extension","").lower() == "pdf":
                # Download the PDF to client's files folder
                download_pdf(base_url, token, t, i, v, fitem["ID"], files_folder, exact_title)
                
                # Save metadata with exactly the same name as the PDF but in metadata folder
                meta_path = os.path.join(metadata_folder, f"{exact_title}.json")
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(flat, f, indent=2, ensure_ascii=False)
                logging.info(f"Metadata → {meta_path}")

    except Exception as e:
        logging.error(f"Error processing Doc {t}-{i}-v{v}: {e}", exc_info=True)

def main():
    load_dotenv()
    BASE = os.getenv("MFILES_BASE_URL")
    USER = os.getenv("MFILES_USERNAME")
    PWD  = os.getenv("MFILES_PASSWORD")
    VAULT= os.getenv("MFILES_VAULT_GUID")
    if not all([BASE, USER, PWD, VAULT]):
        raise SystemExit("Define MFILES_BASE_URL, MFILES_USERNAME, MFILES_PASSWORD & MFILES_VAULT_GUID")

    logging.info(f"Starting M-Files download with URL: {BASE}, User: {USER}, Vault: {VAULT}")
    
    # Create parent folder structure
    parent_folder = os.path.join(DOWNLOAD_ROOT, PARENT_FOLDER_NAME)
    os.makedirs(parent_folder, exist_ok=True)
    logging.info(f"Download folder: {parent_folder}")

    done = load_done_set(parent_folder)
    logging.info(f"Loaded {len(done)} already processed documents")

    try:
        logging.info("Authenticating with M-Files...")
        token = get_token(BASE, USER, PWD, VAULT)
        logging.info("Authentication successful")
        
        logging.info("Loading property definitions...")
        prop_defs = load_property_definitions(BASE, token)
        logging.info(f"Loaded {len(prop_defs)} property definitions")
    except Exception as e:
        logging.error(f"Error during startup: {e}", exc_info=True)
        raise SystemExit(f"Error connecting to M-Files: {e}")

    items_all = search_documents(BASE, token)
    to_do = []
    for item in items_all:
        ev = item.get("ObjVer", item)
        key = (ev["Type"], ev["ID"], ev["Version"])
        if key not in done:
            to_do.append(item)
    logging.info(f"{len(items_all)-len(to_do)} docs already done, {len(to_do)} to process")

    with ThreadPoolExecutor(max_workers=50) as exe:
        futures = {
            exe.submit(process_document, item, idx+1, len(to_do),
                       BASE, token, VAULT, prop_defs, parent_folder): idx
            for idx, item in enumerate(to_do)
        }
        for fut in as_completed(futures):
            # exceptions are already logged inside process_document
            pass

if __name__ == "__main__":
    # configure logging
    logging.basicConfig(
        filename="mfiles_harvest.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )

    # prepare a session with retries
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1,
                    status_forcelist=[500,502,503,504])
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    main()
