#!/usr/bin/env python3
"""
Script to convert PDFs to Markdown and JSON using Docling with OCR enabled.
Processes multiple PDFs in parallel with enhanced page rotation correction.

Configuration is defined in the DEFAULT_CONFIG dictionary at the top of this file.
You can modify these settings directly in the script.

Usage:
    python ocr_chunking.py [--input INPUT_PATH] [--output OUTPUT_DIR] [--force]

Examples:
    # Process with default settings
    python ocr_chunking.py
    
    # Process specific input/output
    python ocr_chunking.py --input data/pdfs --output data/chunks
    
    # Force reprocessing of all files
    python ocr_chunking.py --force
"""

import os
import json
import logging
import argparse
import uuid
import subprocess
from datetime import datetime, timezone
import multiprocessing
import shutil
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
import fnmatch

# Default configuration - modify these settings as needed
DEFAULT_CONFIG = {
    # OCR Configuration
    "ocr": {
        "engine": "paddle",  # Options: paddle, tesseract, easyocr
        "language": "deu",   # Language code (e.g., deu, eng, fra)
        "dpi": 300,
        "confidence": 0.5
    },
    
    # Input/Output Configuration
    "paths": {
        "input": "downloads/bbConcept/Lucid GmbH/files",      # Directory or file to process
        "output": "chunks/bbConcept/Lucid GmbH",   # Output directory
        "metadata_dir": "downloads/bbConcept/Lucid GmbH/metadata", # Metadata directory name
        "workspace_root": "."       # Workspace root for relative paths
    },
    
    # Processing Options
    "processing": {
        "fix_rotation": True,      # Enable page rotation correction
        "enforce_rotation": False, # Force rotation detection
        "rotate_pages": [],        # List of pages to rotate (e.g., ["2:90", "3:270"])
        "skip_processed": True,    # Skip files that are already processed
        "max_chars_per_chunk": 28000,
        "pages_per_split": 25
    },
    
    # File Patterns
    "patterns": {
        "include": ["*.pdf"],      # Files to include
        "exclude": [               # Files to exclude
            "*_temp.pdf",
            "*.rotated.pdf",
            "*.draft.pdf",
            "*.backup.pdf"
        ]
    }
}

import os
import json
import logging
import argparse
import uuid
import subprocess
from datetime import datetime, timezone
import multiprocessing
import shutil
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
import yaml
import fnmatch

# Import Docling components
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode, EasyOcrOptions
from docling.datamodel.base_models import InputFormat

# Try to import PyMuPDF for page dimension analysis
try:
    import fitz  # PyMuPDF
    from PIL import Image
    import numpy as np
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logging.warning("PyMuPDF/PIL not available. Some page rotation features will be limited.")

# Try to import pypdf for splitting
try:
    from pypdf import PdfReader, PdfWriter
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False
    logging.warning("pypdf library not found. Large PDF splitting feature will be disabled.")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(processName)s - %(message)s")

class OCRConfig:
    """Configuration for OCR processing."""
    def __init__(
        self,
        engine: str = "paddle",
        language: str = "en",
        dpi: int = 300,
        confidence: float = 0.5,
        fix_rotation: bool = False,
        enforce_rotation: bool = False,
        rotate_pages: Optional[List[Tuple[int, int]]] = None,
        max_chars_per_chunk: int = 280000,
        pages_per_split: int = 25,
        skip_processed: bool = True
    ):
        self.engine = engine
        self.language = language
        self.dpi = dpi
        self.confidence = confidence
        self.fix_rotation = fix_rotation
        self.enforce_rotation = enforce_rotation
        self.rotate_pages = rotate_pages
        self.max_chars_per_chunk = max_chars_per_chunk
        self.pages_per_split = pages_per_split
        self.skip_processed = skip_processed

    def apply_environment(self) -> None:
        """Apply OCR configuration to environment variables."""
        os.environ["DOCLING_OCR_ENGINE"] = self.engine
        os.environ["DOCLING_OCR_LANG"] = self.language
        os.environ["DOCLING_OCR_CONFIDENCE"] = str(self.confidence)
        os.environ["DOCLING_OCR_DPI"] = str(self.dpi)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'OCRConfig':
        """Create OCRConfig from a dictionary."""
        ocr_config = config_dict.get('ocr', {})
        processing_config = config_dict.get('processing', {})
        
        return cls(
            engine=ocr_config.get('engine', 'paddle'),
            language=ocr_config.get('language', 'en'),
            dpi=ocr_config.get('dpi', 300),
            confidence=ocr_config.get('confidence', 0.5),
            fix_rotation=processing_config.get('fix_rotation', False),
            enforce_rotation=processing_config.get('enforce_rotation', False),
            rotate_pages=parse_rotate_pages(processing_config.get('rotate_pages', [])),
            max_chars_per_chunk=processing_config.get('max_chars_per_chunk', 280000),
            pages_per_split=processing_config.get('pages_per_split', 25),
            skip_processed=processing_config.get('skip_processed', True)
        )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'OCRConfig':
        """Create OCRConfig from command line arguments."""
        return cls(
            engine=args.ocr,
            language=args.lang,
            dpi=args.dpi,
            confidence=args.confidence,
            fix_rotation=args.fix_rotation,
            enforce_rotation=args.enforce_rotation,
            rotate_pages=parse_rotate_pages(args.rotate_pages) if args.rotate_pages else None,
            max_chars_per_chunk=args.max_chars_per_chunk,
            pages_per_split=args.pages_per_split,
            skip_processed=not args.force
        )

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Error loading config file '{config_path}': {e}")
        return {}

def should_process_file(file_path: str, include_patterns: List[str], exclude_patterns: List[str]) -> bool:
    """Check if a file should be processed based on include/exclude patterns."""
    filename = os.path.basename(file_path)
    
    # Check exclude patterns first
    for pattern in exclude_patterns:
        if fnmatch.fnmatch(filename, pattern):
            return False
    
    # If no include patterns, process all non-excluded files
    if not include_patterns:
        return True
    
    # Check include patterns
    for pattern in include_patterns:
        if fnmatch.fnmatch(filename, pattern):
            return True
    
    return False

def get_pdf_files(input_path: str, include_patterns: List[str], exclude_patterns: List[str]) -> List[str]:
    """Get list of PDF files to process based on patterns."""
    pdf_files = []
    
    if os.path.isfile(input_path):
        if should_process_file(input_path, include_patterns, exclude_patterns):
            pdf_files.append(input_path)
    elif os.path.isdir(input_path):
        for root, _, files in os.walk(input_path):
            for file in files:
                file_path = os.path.join(root, file)
                if should_process_file(file_path, include_patterns, exclude_patterns):
                    pdf_files.append(file_path)
    
    return sorted(pdf_files)

def is_already_processed(pdf_path: str, output_dir: str, config: OCRConfig) -> bool:
    """Check if a PDF file has already been processed."""
    if not config.skip_processed:
        return False
        
    basename = os.path.splitext(os.path.basename(pdf_path))[0]
    json_output = os.path.join(output_dir, f"{basename}_chunks.json")
    return os.path.exists(json_output)

def parse_rotate_pages(rotate_str: str) -> Optional[List[Tuple[int, int]]]:
    """Parse a string of page rotations into a list of (page, degrees) tuples."""
    if not rotate_str:
        return None
        
    rotate_pages = []
    for part in rotate_str.split(','):
        try:
            if ':' in part:
                page, degrees = part.split(':')
                rotate_pages.append((int(page), int(degrees)))
            else:
                rotate_pages.append((int(part), 90))  # Default to 90 degrees
        except ValueError:
            logging.warning(f"Invalid rotation specification: {part}, should be 'page:degrees'")
            
    return rotate_pages if rotate_pages else None

def fix_page_rotation(src: str, dst: str, config: OCRConfig) -> str:
    """
    Enhanced function to correct page rotation issues before OCR.
    Uses multiple detection methods to improve auto-detection reliability.
    """
    if not config.fix_rotation:
        return src

    if shutil.which("ocrmypdf") is None:
        logging.info("OCRmyPDF not available for rotation fix - using original file")
        return src
        
    logging.info("Fixing page rotation...")
    try:
        # Handle manual page rotation
        if config.rotate_pages and PYPDF_AVAILABLE:
            reader = PdfReader(src)
            writer = PdfWriter()
            
            for i, page in enumerate(reader.pages):
                page_num = i + 1
                for p, degrees in config.rotate_pages:
                    if page_num == p:
                        logging.info(f"Rotating page {page_num} by {degrees}°")
                        page.rotate(int(degrees))
                        break
                writer.add_page(page)
            
            with open(dst, "wb") as f:
                writer.write(f)
            return dst

        # Auto-rotation using OCRmyPDF
        cmd = [
            "ocrmypdf",
            "--rotate-pages",
            "--rotate-pages-threshold", "0.6",
            "--skip-text",
            "--output-type", "pdf",
        ]
        
        if config.enforce_rotation:
            cmd.extend(["--redo-ocr"])
        
        cmd.extend([src, dst])
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        if "Rotating" in result.stderr:
            logging.info("✅ OCRmyPDF rotation applied successfully")
            return dst
        else:
            logging.info("ℹ️ OCRmyPDF didn't detect any pages that need rotation")
            return src
            
    except Exception as e:
        logging.error(f"⚠️ Error fixing rotation: {e}")
        return src

def create_text_chunks(markdown_content: str, max_chunk_chars: int, base_name: str) -> List[str]:
    """Split markdown content into chunks with titles."""
    if not markdown_content.strip():
        return []

    chunks = []
    offset = 0
    chunk_index = 0

    while offset < len(markdown_content):
        chunk_index += 1
        prefix = f"Title: {base_name}_chunk_{chunk_index}\n\n"
        content_limit = max(0, max_chunk_chars - len(prefix))
        
        if offset + content_limit >= len(markdown_content):
            chunks.append(prefix + markdown_content[offset:])
            break

        # Try to find a good split point
        text_slice = markdown_content[offset:offset + content_limit]
        
        # Try newline first
        split_pos = text_slice.rfind('\n')
        if split_pos == -1:
            # Try sentence enders
            for punc in ['.', '!', '?']:
                pos = text_slice.rfind(punc)
                if pos != -1 and (pos + 1 == len(text_slice) or text_slice[pos + 1].isspace()):
                    split_pos = pos + 1
                    break
        
        if split_pos == -1:
            # Try word boundary
            split_pos = text_slice.rfind(' ')
            if split_pos > 0:
                split_pos += 1

        if split_pos == -1:
            split_pos = content_limit

        chunks.append(prefix + markdown_content[offset:offset + split_pos])
        offset += split_pos

    return chunks

def process_pdf(
    pdf_path: str,
    output_dir: str,
    metadata_base_path: str,
    workspace_root: str,
    config: OCRConfig,
    original_filename: str,
    output_basename: str,
    metadata_basename: str,
    split_info: Optional[Tuple[int, int]] = None
) -> bool:
    """Process a single PDF file with the given configuration."""
    # Initialize processed_path to ensure it's defined even if an exception occurs early
    processed_path = pdf_path
    
    try:
        md_output = os.path.join(output_dir, f"{output_basename}.md")
        json_output = os.path.join(output_dir, f"{output_basename}_chunks.json")
        
        if os.path.exists(json_output):
            logging.info(f"Output JSON '{json_output}' already exists. Skipping.")
            return True

        # Apply rotation if needed
        if config.fix_rotation and pdf_path.lower().endswith(".pdf"):
            rotated_path = os.path.join(os.path.dirname(pdf_path), f".{os.path.basename(pdf_path)}.rotated.pdf")
            processed_path = fix_page_rotation(pdf_path, rotated_path, config)
            if processed_path != pdf_path:
                logging.info(f"Using rotation-fixed PDF: {processed_path}")

        # Load M-Files metadata if available
        metadata = {}
        metadata_path = os.path.join(metadata_base_path, f"{metadata_basename}.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                logging.info(f"Successfully loaded M-Files metadata from '{metadata_path}'")
            except Exception as e:
                logging.warning(f"Could not load metadata from '{metadata_path}': {e}")

        # Configure Docling
        pdf_options = PdfPipelineOptions(do_ocr=True, do_table_structure=True)
        pdf_options.ocr_options = EasyOcrOptions(force_full_page_ocr=True)
        pdf_options.table_structure_options.mode = TableFormerMode.ACCURATE
        
        converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options)})
        
        # Convert document
        result = converter.convert(processed_path)
        if not result or not result.document:
            logging.error(f"Docling conversion failed for {processed_path}")
            return False

        # Save markdown
        markdown_text = result.document.export_to_markdown()
        with open(md_output, "w", encoding="utf-8") as f:
            f.write(markdown_text)

        # Create chunks
        chunks = create_text_chunks(markdown_text, config.max_chars_per_chunk, output_basename)
        
        # Prepare JSON output
        output_json = []
        current_time = datetime.now(timezone.utc).isoformat()
        
        # Determine languages list based on config or use default multilingual list
        languages = [config.language]
        if config.language == "deu":  # For German documents, include common Swiss languages
            languages = ["deu", "fra", "ita", "eng"]
        
        for chunk_text in chunks:
            chunk_metadata = {
                "filename": original_filename,
                "file_directory": os.path.relpath(os.path.dirname(pdf_path), workspace_root),
                "file_metadata": metadata,
                "filetype": "application/pdf",
                "languages": languages,
                "last_modified": current_time,
                "title": metadata_basename
            }
            
            if split_info:
                chunk_metadata["source_document_part"] = {
                    "part_num": split_info[0],
                    "total_parts": split_info[1]
                }
            
            output_json.append({
                "element_id": str(uuid.uuid4()),
                "metadata": chunk_metadata,
                "text": chunk_text
            })

        # Save JSON
        with open(json_output, "w", encoding="utf-8") as f:
            json.dump(output_json, f, ensure_ascii=False, indent=4)

        logging.info(f"Successfully processed '{pdf_path}' into {len(chunks)} chunks")
        return True

    except Exception as e:
        logging.error(f"Error processing '{pdf_path}': {e}")
        return False

    finally:
        # Clean up rotated file if it exists
        if processed_path != pdf_path and os.path.exists(processed_path):
            try:
                os.remove(processed_path)
            except OSError as e:
                logging.warning(f"Could not remove temporary file '{processed_path}': {e}")

def main():
    parser = argparse.ArgumentParser(description="Convert PDFs to Markdown and JSON using Docling with OCR.")
    parser.add_argument("--input", help="Override input path from default config")
    parser.add_argument("--output", "-o", help="Override output directory from default config")
    parser.add_argument("--force", action="store_true", help="Force reprocessing of already processed files")
    parser.add_argument("--simple", action="store_true", help="Use simple mode for a single file")
    parser.add_argument("--metadata_dir", default="metadata", help="Directory name or path for M-Files metadata")
    
    args = parser.parse_args()
    
    # Create OCR configuration from default config
    config = OCRConfig.from_dict(DEFAULT_CONFIG)
    
    # Override config with command line arguments
    if args.force:
        config.skip_processed = False
    
    # Apply OCR configuration
    config.apply_environment()
    
    # Setup paths
    input_path = os.path.abspath(args.input or DEFAULT_CONFIG['paths']['input'])
    output_dir = os.path.abspath(args.output or DEFAULT_CONFIG['paths']['output'])
    metadata_dir = args.metadata_dir or DEFAULT_CONFIG['paths']['metadata_dir']
    workspace_root = os.path.abspath(DEFAULT_CONFIG['paths']['workspace_root'])
    
    # Handle absolute or relative metadata path
    if os.path.isabs(metadata_dir):
        metadata_base_path = metadata_dir
    else:
        # Check if the metadata directory exists within the input path's parent directory
        potential_metadata_dir = os.path.join(os.path.dirname(input_path), metadata_dir)
        if os.path.isdir(potential_metadata_dir):
            metadata_base_path = potential_metadata_dir
        else:
            # Try to locate the metadata directory in a standard location
            standard_metadata_path = os.path.join(workspace_root, 'downloads/bbConcept/Prologist AG/metadata')
            if os.path.isdir(standard_metadata_path):
                metadata_base_path = standard_metadata_path
                logging.info(f"Using standard metadata path: {metadata_base_path}")
            else:
                # Default fallback
                metadata_base_path = os.path.join(os.path.dirname(input_path), metadata_dir)
                logging.warning(f"Metadata directory not found, using default: {metadata_base_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get file patterns
    patterns = DEFAULT_CONFIG['patterns']
    include_patterns = patterns['include']
    exclude_patterns = patterns['exclude']
    
    # Get PDF files to process
    pdf_files = get_pdf_files(input_path, include_patterns, exclude_patterns)
    
    if not pdf_files:
        logging.warning(f"No PDF files found in: {input_path}")
        return 0
    
    # Filter out already processed files
    if config.skip_processed:
        pdf_files = [f for f in pdf_files if not is_already_processed(f, output_dir, config)]
        if not pdf_files:
            logging.info("All files have already been processed. Use --force to reprocess.")
            return 0
    
    # Simple mode for single file
    if args.simple and len(pdf_files) == 1:
        basename = os.path.splitext(os.path.basename(pdf_files[0]))[0]
        success = process_pdf(
            pdf_path=pdf_files[0],
            output_dir=output_dir,
            metadata_base_path=metadata_base_path,
            workspace_root=workspace_root,
            config=config,
            original_filename=os.path.basename(pdf_files[0]),
            output_basename=basename,
            metadata_basename=basename
        )
        return 0 if success else 1
    
    # Prepare tasks for parallel processing
    tasks = []
    temp_dir = os.path.join(output_dir, ".temp_split_pdfs")
    os.makedirs(temp_dir, exist_ok=True)
    
    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        basename = os.path.splitext(filename)[0]
        
        # Check if splitting is needed
        if PYPDF_AVAILABLE:
            try:
                reader = PdfReader(pdf_path)
                total_pages = len(reader.pages)
                
                if total_pages >= config.pages_per_split + 1:
                    num_parts = (total_pages + config.pages_per_split - 1) // config.pages_per_split
                    
                    if num_parts > 1:
                        logging.info(f"Splitting '{filename}' ({total_pages} pages) into {num_parts} parts")
                        
                        for i in range(num_parts):
                            writer = PdfWriter()
                            start = i * config.pages_per_split
                            end = min((i + 1) * config.pages_per_split, total_pages)
                            
                            for page_num in range(start, end):
                                writer.add_page(reader.pages[page_num])
                            
                            part_num = i + 1
                            part_name = f"{basename}_part_{part_num}_of_{num_parts}"
                            part_path = os.path.join(temp_dir, f"{part_name}_temp.pdf")
                            
                            with open(part_path, "wb") as f:
                                writer.write(f)
                            
                            tasks.append((part_path, output_dir, 
                                        metadata_base_path,
                                        workspace_root, config, filename, part_name, basename,
                                        (part_num, num_parts)))
                        continue
            except Exception as e:
                logging.error(f"Error checking/splitting '{filename}': {e}")
        
        # Add task for whole file
        tasks.append((pdf_path, output_dir,
                     metadata_base_path,
                     workspace_root, config, filename, basename, basename, None))
    
    # Process files in parallel
    num_processes = max(1, os.cpu_count() // 4 if os.cpu_count() else 1)
    logging.info(f"Processing {len(tasks)} tasks with {num_processes} workers")
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.starmap(process_pdf, tasks)
    
    # Cleanup
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    
    return 0 if all(results) else 1

if __name__ == "__main__":
    main()
