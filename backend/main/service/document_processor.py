"""
Document Processor Module
Handles multimodal content extraction from various document types.
Extracts text, images, and tables from PDFs, DOCX, TXT, and XLSX files.
"""

import os
import logging
import tempfile
import uuid
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime

from llama_index.core.schema import Document, TextNode

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Process documents and extract multimodal content (text, images, tables)."""
    
    def __init__(self, images_folder: str = "stored_images"):
        """Initialize document processor.
        
        Args:
            images_folder: Folder to store extracted images
        """
        self.images_folder = images_folder
        self._create_images_folder()
    
    def _create_images_folder(self):
        """Create images folder if it doesn't exist."""
        if not os.path.exists(self.images_folder):
            os.makedirs(self.images_folder, exist_ok=True)
            logger.info(f"Created images folder at {self.images_folder}")
    
    def process_document(self, file_path: str, filename: str) -> Tuple[List[TextNode], Dict]:
        """Process a single document and extract all content.
        
        Args:
            file_path: Path to the uploaded file
            filename: Original filename
            
        Returns:
            Tuple of (list of TextNodes, metadata dict)
        """
        file_ext = os.path.splitext(filename)[1].lower()
        
        logger.info(f"Processing document: {filename} ({file_ext})")
        
        if file_ext == '.pdf':
            return self._process_pdf(file_path, filename)
        elif file_ext in ['.docx', '.doc']:
            return self._process_docx(file_path, filename)
        elif file_ext == '.txt':
            return self._process_text(file_path, filename)
        elif file_ext == '.xlsx':
            return self._process_excel(file_path, filename)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
    
    def _process_pdf(self, file_path: str, filename: str) -> Tuple[List[TextNode], Dict]:
        """Extract text, images, and tables from PDF."""
        try:
            import PyPDF2
            from pdf2image import convert_from_path
        except ImportError:
            raise ImportError("PyPDF2 and pdf2image required for PDF processing. Install with: pip install PyPDF2 pdf2image pillow")
        
        text_nodes = []
        extracted_images = []
        tables = []
        
        # Extract text from PDF
        with open(file_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            total_pages = len(pdf_reader.pages)
            
            for page_num, page in enumerate(pdf_reader.pages, 1):
                page_text = page.extract_text()
                
                if page_text.strip():
                    node = TextNode(
                        text=page_text,
                        metadata={
                            "source": filename,
                            "source_type": "pdf",
                            "page": page_num,
                            "total_pages": total_pages,
                            "extracted_at": datetime.now().isoformat(),
                            "content_type": "text",
                            "images": [],
                            "tables": []
                        }
                    )
                    text_nodes.append(node)
                    logger.debug(f"Extracted text from page {page_num}")
        
        # Extract images from PDF
        try:
            images = convert_from_path(file_path)
            for page_num, image in enumerate(images, 1):
                image_filename = self._save_image(image, f"{filename}_page_{page_num}")
                extracted_images.append({
                    "filename": image_filename,
                    "page": page_num,
                    "content_type": "image"
                })
                logger.debug(f"Extracted image from page {page_num}")
        except Exception as e:
            logger.warning(f"Could not extract images from PDF: {e}")
        
        # Update metadata with images
        for node in text_nodes:
            node.metadata["images"] = extracted_images
        
        metadata = {
            "source": filename,
            "source_type": "pdf",
            "total_pages": total_pages,
            "total_text_chunks": len(text_nodes),
            "extracted_images": len(extracted_images),
            "image_references": extracted_images,
            "tables": tables,
            "extracted_at": datetime.now().isoformat()
        }
        
        return text_nodes, metadata
    
    def _process_docx(self, file_path: str, filename: str) -> Tuple[List[TextNode], Dict]:
        """Extract text, images, and tables from DOCX."""
        try:
            from docx import Document as DocxDocument
            from docx.table import _Cell
        except ImportError:
            raise ImportError("python-docx required for DOCX processing. Install with: pip install python-docx")
        
        text_nodes = []
        extracted_images = []
        tables = []
        
        docx_doc = DocxDocument(file_path)
        
        # Extract text and handle structure
        full_text = []
        for para in docx_doc.paragraphs:
            if para.text.strip():
                full_text.append(para.text)
        
        # Extract tables
        for table_idx, table in enumerate(docx_doc.tables, 1):
            table_data = []
            for row in table.rows:
                row_data = [cell.text for cell in row.cells]
                table_data.append(row_data)
            
            table_text = self._format_table(table_data)
            tables.append({
                "table_id": table_idx,
                "content": table_text,
                "content_type": "table"
            })
            full_text.append(f"\n[TABLE {table_idx}]\n{table_text}")
        
        # Extract images from document
        for rel in docx_doc.part.rels.values():
            if "image" in rel.target_ref:
                try:
                    image_part = rel.target_part
                    image_filename = self._save_image_from_bytes(
                        image_part.blob,
                        f"{filename}_image_{len(extracted_images)+1}"
                    )
                    extracted_images.append({
                        "filename": image_filename,
                        "content_type": "image"
                    })
                    logger.debug(f"Extracted image from DOCX")
                except Exception as e:
                    logger.warning(f"Could not extract image from DOCX: {e}")
        
        # Create text node
        if full_text:
            node = TextNode(
                text="\n".join(full_text),
                metadata={
                    "source": filename,
                    "source_type": "docx",
                    "extracted_at": datetime.now().isoformat(),
                    "content_type": "text",
                    "images": extracted_images,
                    "tables": tables
                }
            )
            text_nodes.append(node)
        
        metadata = {
            "source": filename,
            "source_type": "docx",
            "total_text_chunks": len(text_nodes),
            "extracted_images": len(extracted_images),
            "image_references": extracted_images,
            "total_tables": len(tables),
            "tables": tables,
            "extracted_at": datetime.now().isoformat()
        }
        
        return text_nodes, metadata
    
    def _process_text(self, file_path: str, filename: str) -> Tuple[List[TextNode], Dict]:
        """Extract text from TXT file."""
        text_nodes = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if content.strip():
            node = TextNode(
                text=content,
                metadata={
                    "source": filename,
                    "source_type": "text",
                    "extracted_at": datetime.now().isoformat(),
                    "content_type": "text",
                    "images": [],
                    "tables": []
                }
            )
            text_nodes.append(node)
        
        metadata = {
            "source": filename,
            "source_type": "text",
            "total_text_chunks": len(text_nodes),
            "extracted_images": 0,
            "image_references": [],
            "tables": [],
            "extracted_at": datetime.now().isoformat()
        }
        
        return text_nodes, metadata
    
    def _process_excel(self, file_path: str, filename: str) -> Tuple[List[TextNode], Dict]:
        """Extract text and tables from XLSX."""
        try:
            import openpyxl
            import pandas as pd
        except ImportError:
            raise ImportError("openpyxl and pandas required for XLSX processing. Install with: pip install openpyxl pandas")
        
        text_nodes = []
        tables = []
        
        # Read all sheets
        xls = pd.ExcelFile(file_path)
        sheet_names = xls.sheet_names
        
        for sheet_name in sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            
            table_text = self._format_dataframe_table(df)
            tables.append({
                "sheet_name": sheet_name,
                "content": table_text,
                "content_type": "table",
                "rows": len(df),
                "columns": len(df.columns)
            })
            
            node = TextNode(
                text=table_text,
                metadata={
                    "source": filename,
                    "source_type": "excel",
                    "sheet_name": sheet_name,
                    "extracted_at": datetime.now().isoformat(),
                    "content_type": "table",
                    "images": [],
                    "tables": [{"sheet_name": sheet_name}]
                }
            )
            text_nodes.append(node)
            logger.debug(f"Extracted table from sheet: {sheet_name}")
        
        metadata = {
            "source": filename,
            "source_type": "excel",
            "total_sheets": len(sheet_names),
            "total_text_chunks": len(text_nodes),
            "extracted_images": 0,
            "image_references": [],
            "total_tables": len(tables),
            "tables": tables,
            "extracted_at": datetime.now().isoformat()
        }
        
        return text_nodes, metadata
    
    def _save_image(self, image, base_name: str) -> str:
        """Save PIL image to folder and return filename.
        
        Args:
            image: PIL Image object
            base_name: Base name for the image file
            
        Returns:
            Filename of saved image
        """
        image_id = str(uuid.uuid4())[:8]
        image_filename = f"{base_name}_{image_id}.png"
        image_path = os.path.join(self.images_folder, image_filename)
        
        image.save(image_path, 'PNG')
        logger.info(f"Saved image to {image_path}")
        
        return image_filename
    
    def _save_image_from_bytes(self, image_bytes: bytes, base_name: str) -> str:
        """Save image from bytes to folder and return filename.
        
        Args:
            image_bytes: Image bytes
            base_name: Base name for the image file
            
        Returns:
            Filename of saved image
        """
        try:
            from PIL import Image
            import io
            
            image = Image.open(io.BytesIO(image_bytes))
            return self._save_image(image, base_name)
        except Exception as e:
            logger.error(f"Could not save image from bytes: {e}")
            raise
    
    def _format_table(self, table_data: List[List[str]]) -> str:
        """Format table data as text.
        
        Args:
            table_data: 2D list of table cells
            
        Returns:
            Formatted table text
        """
        if not table_data:
            return ""
        
        lines = []
        for row in table_data:
            lines.append(" | ".join(str(cell) for cell in row))
        
        return "\n".join(lines)
    
    def _format_dataframe_table(self, df) -> str:
        """Format DataFrame as text.
        
        Args:
            df: pandas DataFrame
            
        Returns:
            Formatted table text
        """
        return df.to_string()
