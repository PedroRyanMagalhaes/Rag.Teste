import PyPDF2
from pathlib import Path
from typing import List, Dict, Optional
import Logging

Logging.basicConfig(level=Logging.INFO)
logger = Logging.getLogger(__name__)

def extract_text_from_pdf(pdf_path: str) -> str:
    
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF não encontrado: {pdf_path}")
    
    if pdf_path.suffix.lower() != '.pdf':
        raise ValueError(f"O arquivo não é um PDF: {pdf_path}")
    
    logger.info(f"Extraindo texto do PDF: {pdf_path.name}")

    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            num_pages = len(pdf_reader.pages)
            logger.info(f"Número de páginas no PDF: {num_pages}")

            text = ""
            for page_num in enumerate(pdf_reader.pages, 1):
                try:
                    page_text = page.extract_text()
                    text += page_text + "\n"

                    if page_num % 10 == 0:
                        logger.info(f"Processadas {page_num}/{num_pages} páginas...")
                except Exception as e:
                    logger.error(f"Erro ao extrair texto da página {page_num}: {e}")
                    continue
            if not text.strip():
                raise ValueError("Nenhum texto extraído do PDF.")
            logger.info(f"Extração de texto concluída: {len(text)} caracteres extraídos.")
            return text.strip()
    except Exception as e:
        logger.error(f"Erro ao extrair texto do PDF {pdf_path}: {e}")
        raise

def get_pdf_metadata(pdf_path: str) -> Dict[str, any]:

    pdf_path = Path(pdf_path)

    try: 
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)

            metadata = {
                'filename': pdf_path.name,
                'file_size': pdf_path.stat().st_size,
                'num_pages': len(pdf_reader.pages),
                'title': getattr(pdf_reader.metadata, 'title', 'N/A') if pdf_reader.metadata else 'N/A',
                'author': getattr(pdf_reader.metadata, 'author', 'N/A') if pdf_reader.metadata else 'N/A',
                'creation_date': getattr(pdf_reader.metadata, 'creation_date', 'N/A') if pdf_reader.metadata else 'N/A',
            }

            return metadata
        
    except Exception as e:
        logger.error(f"Erro ao obter metadados do PDF {pdf_path}: {e}")
        raise

def find_pdf_files(directory: str) -> List[str]:
    
    directory = Path(directory)
    if not directory.existis():
        logger.warning(f"O diretório não existe: {directory}")
        return []
    
    pdf_files = []
    for pattern in ['**/*.pdf', '**/*.PDF']:
        pdf_files.extend([str(p) for p in directory.glob(pattern)])

    logger.info(f"Encontrados {len(pdf_files)} arquivos PDF em {directory}")
    return sorted(pdf_files)

def validate_pdf(pdf_path: str) -> bool:
    
    try: 
        extract_text_from_pdf(pdf_path)
        return True
    except Exception as e:
        return False

    