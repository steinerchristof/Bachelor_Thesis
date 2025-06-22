"""
Metadata utilities for processing and formatting document metadata
"""
from typing import Dict, Any, List

def to_str(v: Any) -> str:
    """
    Convert any value to a string representation, with special handling for lists and dicts
    """
    if isinstance(v, list):
        return "; ".join(
            (x.get("DisplayValue")
             if isinstance(x, dict) and "DisplayValue" in x else str(x))
            for x in v
        )
    return str(v)

def flatten_props(meta: Dict[str, Any]) -> Dict[str, str]:
    """
    Flatten M-Files-style metadata dictionaries to plain strings.
    
    Args:
        meta: Metadata dictionary with potentially nested properties
        
    Returns:
        Dictionary with flattened string values
    """
    if isinstance(meta.get("properties"), dict):
        props = meta["properties"]
    elif isinstance(meta.get("document_metadata", {}).get("properties"), dict):
        props = meta["document_metadata"]["properties"]
    elif isinstance(meta.get("file_metadata"), dict):
        props = meta["file_metadata"]
    elif any(isinstance(v, dict) and "displayValue" in v for v in meta.values()):
        props = meta
    else:
        props = {}
    
    flat: Dict[str, str] = {}
    for k, v in props.items():
        if isinstance(v, dict):
            flat[k] = v.get("displayValue") or to_str(v.get("rawValue"))
        else:
            flat[k] = to_str(v)
    return flat

def extract_payload(meta: Dict[str, Any]) -> Dict[str, str]:
    """
    Extract key fields from metadata for indexing
    
    Args:
        meta: Metadata dictionary
        
    Returns:
        Dictionary with extracted fields for indexing
    """
    props = flatten_props(meta)
    file_meta = meta.get("file_metadata") or meta.get("properties") or meta

    # -- Externer Zugriff → Item-IDs ----------------------------------------- #
    ex_items: List[str] = []
    ex_entry = file_meta.get("Externer Zugriff für", {})
    if isinstance(ex_entry, dict):
        rv = ex_entry.get("rawValue")
        if isinstance(rv, list):
            ex_items = [str(v.get("Item")) for v in rv if isinstance(v, dict)]
        elif isinstance(rv, dict):
            ex_items = [str(rv.get("Item"))]
    ex_items_str = "; ".join(ex_items)

    # -- Klasse (Nummer) ------------------------------------------------------ #
    klasse_num = ""
    klasse_entry = file_meta.get("Klasse", {})
    rv = klasse_entry.get("rawValue")
    if isinstance(rv, dict):
        klasse_num = rv.get("Item", "")
    elif isinstance(rv, list) and rv and isinstance(rv[0], dict):
        klasse_num = rv[0].get("Item", "")

    mfiles_id = (file_meta.get("__objectID")
                 or meta.get("__objectID")
                 or meta.get("id", ""))

    return {
        "Adressen":             props.get("Adressen", ""),
        "ExternerZugriff":      props.get("Externer Zugriff für", ""),
        "ExternerZugriffItems": ex_items_str,
        "Titel":                props.get("Name oder Titel", ""),
        "MfilesId":             str(mfiles_id),
        "PortalLink":           props.get("Portal Link", ""),
        "Jahr":                 props.get("Jahr", ""),
        "KlasseNummer":         str(klasse_num),
        "KlasseName":           props.get("Klasse", ""),
    } 