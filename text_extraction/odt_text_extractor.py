import os, sys, json
from typing import Iterable, Iterator, Callable
import zipfile
import xml.etree.ElementTree as ET
import re
from lxml import etree
def extract_text_from_odt(odt_file):
    # Open the ODT file as a ZIP archive
    with zipfile.ZipFile(odt_file, 'r') as zip_ref:
        # Extract the content.xml file
        content = zip_ref.read('content.xml')
        
    # Parse the XML
    root = ET.fromstring(content)
    
    # Define the namespace
    namespace = {'text': 'urn:oasis:names:tc:opendocument:xmlns:text:1.0'}
    
    # Find all text elements
    paragraphs = root.findall('.//text:p', namespace)
    # Extract text from each paragraph
    text_content = []
    for paragraph in paragraphs:
        # Get all text content from this paragraph
        para_text = ''.join(paragraph.itertext())
        text_content.append(para_text)
    
    # Join paragraphs with newlines
    full_text = '\n'.join(text_content)
    
    return full_text
def g_body(odt_file_path):
    """
    Extracts and returns the office:body element from content.xml
    in an ODT file as an lxml Element object.
    
    Args:
        odt_file_path: Path to the ODT file
        
    Returns:
        lxml Element object representing the office:body tag
    """
    # Extract content.xml from the ODT file
    with zipfile.ZipFile(odt_file_path, 'r') as zip_ref:
        content_xml = zip_ref.read('content.xml')
    
    # Parse the XML
    parser = etree.XMLParser(remove_blank_text=True)
    root = etree.fromstring(content_xml, parser)
    
    # Find and return the office:body element
    namespace = {'office': 'urn:oasis:names:tc:opendocument:xmlns:office:1.0'}
    body = root.find('.//office:body', namespace)
    
    return body

def g_body_content_for_model(odt_file_path, indent=True):
    """
    Processes the office:body element by:
    1. Removing any text:sequence-decls elements
    2. Converting the entire office:body element to a string
    3. Removing namespace declarations from the body tag
    
    Args:
        odt_file_path: Path to the ODT file
        indent: Boolean to determine if output should be indented (default: True)
        
    Returns:
        Clean string representation of the office:body element
    """
    # Get the body element
    body = g_body(odt_file_path)
    
    # Find and remove any sequence-decls
    namespace = {'text': 'urn:oasis:names:tc:opendocument:xmlns:text:1.0'}
    for seq_decl in body.findall('.//text:sequence-decls', namespace):
        parent = seq_decl.getparent()
        if parent is not None:
            parent.remove(seq_decl)
    
    # Convert to string, preserving all tags, with indentation if requested
    body_str = etree.tostring(body, encoding='unicode', method='xml', pretty_print=indent)
    
    # Remove all the namespace declarations from the body tag
    # Replace the long xmlns declarations with just a simple open tag
    body_str = re.sub(r'<office:body[^>]+>', '<office:body>', body_str)
    
    return body_str
if __name__ == "__main__":
    todays_notes_path = "D:\DocumentsHDD\ToCards\ToCards_04_01_25.odt"
    text = extract_text_from_odt(todays_notes_path)
    print(text)