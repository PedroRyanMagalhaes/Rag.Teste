import PyPDF2
from pathlib import Path
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extrai todo o texto de um arquivo PDF
    
    Args:
        pdf_path (str): Caminho para o arquivo PDF
        
    Returns:
        str: Texto extraído do PDF
        
    Raises:
        FileNotFoundError: Se o arquivo não existir
        Exception: Se houver erro na leitura do PDF
    """
    
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
            for page_num, page in enumerate(pdf_reader.pages, 1):
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
    if not directory.exists():
        logger.warning(f"O diretório não existe: {directory}")
        return []
    
    pdf_files = set()  
    for pattern in ['*.pdf', '*.PDF']:
        for p in directory.glob(pattern):
            pdf_files.add(str(p))  

    pdf_files = sorted(list(pdf_files))  
    logger.info(f"Encontrados {len(pdf_files)} arquivos PDF em {directory}")
    return pdf_files

def validate_pdf(pdf_path: str) -> bool:
    
    try: 
        extract_text_from_pdf(pdf_path)
        return True
    except Exception as e:
        return False
    
def extract_text_from_directory(directory: str) -> Dict[str, str]:
    """
    Extrai texto de todos os PDFs em um diretório
    
    Args:
        directory (str): Caminho do diretório
        
    Returns:
        Dict[str, str]: Dicionário com nome do arquivo → texto extraído
    """
    pdf_files = find_pdf_files(directory)
    
    if not pdf_files:
        logger.warning("⚠️ Nenhum PDF encontrado no diretório")
        return {}
    
    results = {}
    for pdf_file in pdf_files:
        try:
            text = extract_text_from_pdf(pdf_file)
            results[Path(pdf_file).name] = text 
            logger.info(f"✅ {Path(pdf_file).name}: {len(text)} caracteres")
        except Exception as e:
            logger.error(f"❌ Erro em {Path(pdf_file).name}: {e}")
            results[Path(pdf_file).name] = f"ERRO: {e}"

    return results

# Teste rápido da função
if __name__ == "__main__":
    
    from src.utils.config import DOCUMENTS_DIR
    
    print("🧪 Testando extração de PDF...")
    
    # Lista PDFs disponíveis
    pdfs = find_pdf_files(DOCUMENTS_DIR)
    if pdfs:
        print(f"📄 PDFs encontrados: {[Path(p).name for p in pdfs]}")        
        # Testa o primeiro PDF
        try:
            text = extract_text_from_pdf(pdfs[0])
            print(f"✅ Teste OK! Extraídos {len(text)} caracteres")
            print(f"📝 Primeiros 200 caracteres:\n{text[:200]}...")
        except Exception as e:
            print(f"❌ Erro no teste: {e}")
    else:
        print("📁 Coloque alguns PDFs na pasta data/documents/ para testar")

    