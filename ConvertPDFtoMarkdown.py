import json
from pathlib import Path
from docling.document_converter import DocumentConverter

def pdf_to_json(pdf_path: str, output_path: str):
    
    converter = DocumentConverter()

    result = converter.convert(pdf_path)

    document_dict = result.document.export_to_dict()

    with open(output_path,"w",encoding="utf-8") as json_file:
        json.dump(document_dict,json_file,indent=4,ensure_ascii= False)


def pdf_to_markdown(pdf_path:str, output_path:str):
    converter = DocumentConverter()

    result = converter.convert(pdf_path)

    markdown = result.document.export_to_markdown()

    with open(output_path, "w", encoding="utf-8") as md_file:
        md_file.write(markdown)
    print(f"Successfully saved Markdown to {output_path}")

if __name__ == "__main__":
    # FIXED: Wrapped the Windows path in Path() to automatically handle backslashes safely
    input_contract = Path(r"D:\Internship\AStar\CuadNew\cuad\dataset\full_contract_pdf\Part_I\Affiliate_Agreements\1.pdf")
    
    output_json = Path("parsed_contract.json")
    output_markdown = Path("parsed_contract.markdown")
    
    # Run the converter
    try:
        pdf_to_json(input_contract, output_json)
        pdf_to_markdown(input_contract, output_markdown)
    except Exception as e:
        print(f"An error occurred during conversion: {e}")