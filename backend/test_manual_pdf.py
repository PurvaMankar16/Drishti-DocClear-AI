#!/usr/bin/env python3
"""
Manual PDF Enhancement Testing Script

Usage:
    python test_manual_pdf.py path/to/your.pdf

This script will:
1. Upload the PDF to the API
2. Enhance it
3. Save diagnostic outputs
4. Report validation results
"""

import sys
import json
from pathlib import Path
import requests

API_BASE = "http://127.0.0.1:8000"


def upload_pdf(pdf_path: Path):
    """Upload PDF and get analysis."""
    print(f"📤 Uploading {pdf_path.name}...")

    with open(pdf_path, 'rb') as f:
        files = {'file': (pdf_path.name, f, 'application/pdf')}
        response = requests.post(f"{API_BASE}/api/documents/analyze", files=files)

    if response.status_code != 200:
        print(f"❌ Upload failed: {response.status_code}")
        print(response.text)
        sys.exit(1)

    data = response.json()
    print(f"✅ Uploaded successfully")
    print(f"   Document ID: {data['document_id']}")
    print(f"   Quality Score: {data['overall_score']}/100")
    print(f"   Classification: {data['classification']}")
    return data


def enhance_document(doc_id: str):
    """Enhance the document."""
    print(f"\n🔧 Enhancing document {doc_id}...")

    response = requests.post(f"{API_BASE}/api/documents/{doc_id}/enhance")

    if response.status_code != 200:
        print(f"❌ Enhancement failed: {response.status_code}")
        print(response.text)
        sys.exit(1)

    data = response.json()
    print(f"✅ Enhancement completed")
    print(f"   Status: {data['status']}")
    print(f"   Original Score: {data['original_quality']['overall_score']}/100")
    print(f"   Enhanced Score: {data['enhanced_quality']['overall_score']}/100")
    print(f"   Improvement: {data['improvement']['absolute_score_change']:+.2f} points")

    # Show operations
    print(f"\n   Operations Applied:")
    for op in data['operations']:
        if op['applied'] and op['name'] not in ['validation', 'pdf_processing']:
            print(f"   - {op['name']}")

    # Show validation result
    validation_op = next((op for op in data['operations'] if op['name'] == 'validation'), None)
    if validation_op and 'result' in validation_op:
        result = validation_op['result']
        print(f"\n   Validation: {'✅ PASSED' if result['is_valid'] else '❌ REJECTED'}")
        print(f"   Message: {result['message']}")

    return data


def download_files(doc_id: str, enhancement_data: dict, output_dir: Path):
    """Download enhanced PDF and preview."""
    output_dir.mkdir(exist_ok=True)

    if not enhancement_data['enhanced_file']['available']:
        print("\n⚠️  Enhanced file not available (rejected by validation)")
        return

    print(f"\n💾 Downloading files to {output_dir}/...")

    # Download enhanced PDF
    download_url = enhancement_data['enhanced_file']['download_url']
    response = requests.get(f"{API_BASE}{download_url}")

    if response.status_code == 200:
        output_path = output_dir / f"{doc_id}_enhanced.pdf"
        output_path.write_bytes(response.content)
        print(f"✅ Saved enhanced PDF: {output_path}")
    else:
        print(f"❌ Failed to download enhanced PDF: {response.status_code}")

    # Check for debug files (if running locally and have access to backend storage)
    print(f"\n📊 Check backend storage for diagnostic files:")
    print(f"   - {doc_id}_page_0_original_debug.png")
    print(f"   - {doc_id}_page_0_enhanced_debug.png")
    print(f"   - {doc_id}_page_0_validation.png")


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_manual_pdf.py path/to/your.pdf")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])

    if not pdf_path.exists():
        print(f"❌ File not found: {pdf_path}")
        sys.exit(1)

    if pdf_path.suffix.lower() != '.pdf':
        print(f"❌ Not a PDF file: {pdf_path}")
        sys.exit(1)

    print("="*60)
    print("DocClear AI - Manual PDF Enhancement Test")
    print("="*60)

    # Upload
    analysis = upload_pdf(pdf_path)

    # Enhance
    enhancement = enhance_document(analysis['document_id'])

    # Download
    output_dir = Path("test_output")
    download_files(analysis['document_id'], enhancement, output_dir)

    print("\n" + "="*60)
    print("✅ Test completed!")
    print("="*60)

    # Final summary
    if enhancement['enhanced_file']['available']:
        preview_url = f"{API_BASE}{enhancement['enhanced_file']['preview_url']}"
        print(f"\n🔗 Preview URL: {preview_url}")
        print(f"   Open this URL in your browser to see the enhanced PDF preview")
    else:
        print(f"\n⚠️  Enhancement was rejected - original PDF is better")


if __name__ == "__main__":
    main()
