# Drishti — DocClear AI

**Printed Document Quality Assessment and Image Enhancement**

Drishti (DocClear AI) is a document image enhancement application designed to assess and improve the visual quality of printed and digitally generated documents. It combines computer vision, image-processing techniques, and deep learning-based quality assessment to help users identify document-quality issues and obtain enhanced versions of their images.

The application focuses on improving clarity while preserving the original document's content, structure, and layout.

## Overview

Images captured using mobile cameras or scanned under poor conditions may contain blur, low contrast, uneven illumination, noise, skew, or insufficient resolution. These issues can make documents difficult to read and reduce their usability. Drishti addresses these challenges through an image-processing pipeline that assesses document quality and applies targeted enhancement operations.

**Supported Document Types:**
- Scholarship application documents
- Mark sheets and academic certificates
- Income certificates
- Caste certificates
- Government application forms
- Identity documents
- Printed letters and official documents
- Scanned PDF documents

**Out of Scope for Version 1:**
- Handwritten documents
- OCR / text extraction
- Digitally generated PDFs with embedded text

## Key Features

- **Document Upload:** Upload document images and multi-page PDFs for quality assessment and enhancement.
- **Image Quality Assessment:** Identify document-quality issues using objective measurements and a ResNet-18-based multi-label classification model.
- **Image Deblurring:** Apply sharpening techniques to improve the apparent clarity of text and document details.
- **Contrast Enhancement:** Use Contrast Limited Adaptive Histogram Equalization (CLAHE) to improve local contrast.
- **Noise Reduction:** Apply image denoising techniques to reduce unwanted visual noise.
- **Illumination Correction:** Improve uneven brightness where appropriate.
- **Document Deskewing:** Correct document orientation when skew is detected and correction is appropriate.
- **Before-and-After Comparison:** Compare the original and processed images side-by-side.
- **Quality Metrics:** Display specific quality measurements: blur, contrast, brightness, noise, resolution, and skew.
- **Original Image Preservation:** Keep the original upload available separately from the enhanced output.
- **PDF Multi-Page Support:** Upload, assess, and enhance multi-page PDF documents.

## System Architecture

```text
              User
               |
               v
       React Frontend
               |
               v
        FastAPI Backend
               |
       +-------+--------+
       |                |
       v                v
 Quality Assessment   Image Enhancement
 (ResNet-18)          Pipeline
       |                |
       |          +-----+------+-------+
       |          |     |      |       |
       |        Deblur Contrast Denoise Deskew
       |          |     |      |       |
       |          +-----+------+-------+
       |                |
       +-------+--------+
               |
               v
     Results and Comparison
               |
               v
           User Review
```

*The diagram represents the high-level workflow; individual processing steps are applied adaptively based on detected quality issues.*

## Technology Stack

| Component | Technology |
|---|---|
| Frontend | React, JavaScript, Vite |
| Backend | Python, FastAPI |
| Deep Learning | PyTorch |
| Quality Assessment | ResNet-18-based multi-label classifier |
| Image Processing | OpenCV |
| PDF Processing | PyMuPDF (fitz) |
| Data Validation | Pydantic |
| Testing | pytest |
| Package Management | npm for frontend; pip/virtual environments for Python |

## Image Enhancement Workflow

The application follows a quality-assessment and enhancement workflow.

1. **Upload:** The user uploads a document image or multi-page PDF.
2. **Assess:** The application evaluates the document and identifies potential quality issues using objective metrics and optional ML-based predictions.
3. **Preprocess:** The backend prepares the image for the selected processing operations.
4. **Enhance:** Targeted image-processing techniques are applied adaptively, such as contrast enhancement, sharpening, denoising, illumination correction, and deskewing.
5. **Evaluate:** Quality metrics are recalculated to measure improvement.
6. **Present:** The frontend presents the original document, enhancement results, and quality metrics.
7. **Review:** The user can inspect the result and compare metrics before downloading the enhanced document.

The enhancement process is designed to improve the existing image rather than generate a replacement document using generative AI.

## Project Structure

The repository is organized into frontend and backend components.

```text
Drishti/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── routes/
│   │   ├── core/
│   │   ├── schemas/
│   │   ├── services/
│   │   │   ├── enhancement/
│   │   │   ├── ml/
│   │   │   └── ...
│   │   └── utils/
│   ├── tests/
│   ├── requirements.txt
│   └── models/
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   └── services/
│   ├── package.json
│   └── package-lock.json
├── .gitignore
└── README.md
```

## Prerequisites

Install the following before running the application:

- Python 3.10 or a compatible version supported by the project's dependencies.
- Node.js and npm.
- Git.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/PurvaMankar16/Drishti.git
cd Drishti
```

### 2. Set up the backend

Open a terminal in the project root:

```bash
cd backend
python -m venv .venv
```

Activate the environment.

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux:**

```bash
source .venv/bin/activate
```

Install the backend dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure environment variables

Review the provided environment example, if available:

```text
backend/.env.example
```

Create a local `.env` file from the example and configure the required values.

Do not commit secrets, API keys, credentials, or machine-specific paths to GitHub.

### 4. Start the backend

From the `backend` directory, run:

```bash
uvicorn app.main:app --reload
```

The backend should start on its configured local address. Check the terminal output for the actual host and port.

### 5. Set up the frontend

Open another terminal in the project root:

```bash
cd frontend
npm install
```

Start the Vite development server:

```bash
npm run dev
```

Open the local URL printed by Vite in your browser.

**Configuration note:** Ensure that the frontend's API base URL points to the running backend. Use the environment-variable name and configuration method defined in the actual frontend code.

## Model Information

Drishti uses a ResNet-18-based multi-label classifier to assess potential document-quality issues.

The supported quality categories are:

- Blur
- Low contrast
- Poor illumination or dark images
- Noise
- Skew
- Low resolution

The classifier provides supplementary evidence and is separate from the objective image-processing metrics. Model performance depends on the training data, preprocessing, and real-world input distribution. Results from synthetic training data should not be treated as proof of equivalent performance on real printed documents.

### Model Weights

Trained model checkpoints are not necessarily included in the Git repository because binary model files can be large.

If a checkpoint is required to run the classifier, obtain the corresponding file from the project's authorized distribution location and configure the application to use it. Ensure that the architecture, preprocessing, label order, and checkpoint are compatible.

## Testing

The backend uses pytest for automated testing.

From the project root, run:

```bash
python -m pytest
```

If the project's tests require a specific Python environment or configuration, activate that environment first and follow the repository's test configuration.

For frontend build verification, run:

```bash
cd frontend
npm run build
```

Testing covers image enhancement, quality assessment, error handling, and result presentation.

## Limitations

- Enhancement quality varies depending on the original image and the specific quality issues present.
- Severe blur, missing image details, or very low resolution cannot always be corrected reliably.
- The quality classifier's real-world reliability depends on representative training and evaluation data.
- Results should be visually verified before using enhanced documents for official, academic, or other important purposes.
- Handwritten documents are not supported in Version 1.
- Text extraction and OCR are not supported.

## Future Enhancements

Potential areas for further development include:

- Evaluation on larger and more diverse real-world document datasets.
- Adaptive enhancement parameters based on detected document characteristics.
- Quantitative benchmarking using image-quality metrics.
- Multi-page PDF processing with synchronized quality metrics and batch enhancement.
- Improved confidence reporting and user-facing explanations.
- Support for additional document types and formats.
- Handwriting-aware quality assessment and enhancement for Version 2+.

## Contributing

Contributions, bug reports, and suggestions are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Make your changes and add relevant tests.
4. Verify that existing tests and builds pass.
5. Submit a pull request describing the changes.

## License

No license is specified here. Add an appropriate `LICENSE` file before distributing the project or defining reuse permissions.

## Acknowledgements

Drishti brings together open-source computer vision and deep learning technologies to explore practical document-quality assessment and image enhancement.
