# Drishti — DocClear AI

**AI-Assisted Document Quality Assessment and Image Enhancement**

Drishti (DocClear AI) is a document image enhancement application designed to improve the visual quality and readability of uploaded document images. It combines computer vision, image-processing techniques, deep learning-based quality assessment, and Optical Character Recognition (OCR) to help users identify document-quality issues and obtain enhanced versions of their images.

The application focuses on improving clarity while preserving the original document's text, handwriting, structure, and layout.

## Overview

Images captured using mobile cameras or scanned under poor conditions may contain blur, low contrast, uneven illumination, noise, skew, or insufficient resolution. These issues can make documents difficult to read and reduce the effectiveness of downstream text extraction.

Drishti addresses these challenges through an image-processing pipeline that assesses document quality, applies enhancement operations, and uses OCR-based validation to help evaluate whether the resulting image remains readable.

## Key Features

- **Document Upload:** Upload document images for quality assessment and enhancement.
- **Image Quality Assessment:** Identify potential document-quality issues using a ResNet-18-based multi-label classification model.
- **Image Deblurring:** Apply sharpening techniques to improve the apparent clarity of text and document details.
- **Contrast Enhancement:** Use Contrast Limited Adaptive Histogram Equalization (CLAHE) to improve local contrast.
- **Noise Reduction:** Apply image denoising techniques to reduce unwanted visual noise.
- **Illumination Correction:** Improve uneven brightness where appropriate.
- **Document Deskewing:** Correct document orientation when skew is detected and correction is appropriate.
- **Before-and-After Comparison:** Compare the original and processed images.
- **OCR-Based Validation:** Use PaddleOCR to evaluate text recognition before and after enhancement.
- **Content-Preservation Checks:** Compare OCR results to identify potential changes in recognized text.
- **Review Recommendations:** Flag cases that require manual review when readability or content preservation cannot be adequately validated.
- **Original Image Preservation:** Keep the original upload available separately from the enhanced output.

> **Important:** Image enhancement does not guarantee perfect text recovery. OCR comparisons can produce differences even when the underlying content is preserved. Results should be reviewed carefully, especially for handwritten documents and low-quality images.

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
       |          +-----+------+
       |          |     |      |
       |        Deblur Contrast Denoise
       |          |     |      |
       |          +-----+------+
       |                |
       +-------+--------+
               |
               v
          PaddleOCR
               |
               v
     Readability and Content
       Preservation Checks
               |
               v
      Results and Comparison
               |
               v
           User Review
```

*The diagram represents the intended high-level workflow; individual processing steps may be conditional depending on the implementation.*

## Technology Stack

| Component | Technology |
|---|---|
| Frontend | React, JavaScript, Vite |
| Backend | Python, FastAPI |
| Deep Learning | PyTorch |
| Quality Assessment | ResNet-18-based multi-label classifier |
| Image Processing | OpenCV |
| OCR | PaddleOCR, PaddlePaddle |
| Data Validation | OCR output and readability comparisons |
| Testing | pytest |
| Package Management | npm for frontend; pip/virtual environments for Python |

## Image Enhancement Workflow

The application follows a quality-assessment and enhancement workflow.

1. **Upload:** The user uploads a document image.
2. **Assess:** The application evaluates the image and identifies potential quality issues.
3. **Preprocess:** The backend prepares the image for the selected processing operations.
4. **Enhance:** Appropriate image-processing techniques are applied, such as contrast enhancement, sharpening, denoising, illumination correction, and deskewing.
5. **Evaluate:** OCR is used to compare text recognition and assess whether enhancement may have affected readability or content.
6. **Present:** The frontend presents the original image, enhancement results, and any relevant review warnings.
7. **Review:** The user can inspect the result before relying on or sharing the enhanced document.

The enhancement process is intended to improve the existing image rather than generate a replacement document using generative AI.

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
│   │   └── services/
│   │       ├── enhancement/
│   │       └── ocr/
│   ├── tests/
│   ├── requirements.txt
│   └── ocr-requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   └── pages/
│   ├── package.json
│   └── package-lock.json
├── .gitignore
└── README.md
```

*This is a representative structure. Additional files and directories may exist in the repository.*

## Prerequisites

Install the following before running the application:

- Python 3.10 or a compatible version supported by the project's dependencies.
- Node.js and npm.
- Git.
- A compatible Python environment for the backend and OCR dependencies.

PaddleOCR and PaddlePaddle may require additional platform-specific setup and model downloads.

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

Install the OCR dependencies separately if required by the project's configured OCR runtime:

```bash
pip install -r ocr-requirements.txt
```

**Note:** If OCR is configured to run in a separate virtual environment, create and configure that environment according to the project's OCR runtime settings. Do not install incompatible dependencies into the same environment without checking their version requirements.

### 3. Configure environment variables

Review the provided environment example, if available:

```text
backend/.env.example
```

Create a local `.env` file from the example and configure the required values.

Do not commit secrets, API keys, credentials, or machine-specific paths to GitHub.

If the application uses a dedicated OCR Python interpreter, configure the relevant runtime variable, such as `OCR_RUNTIME_PYTHON`, to point to the correct interpreter.

### 4. Start the backend

From the `backend` directory, run the command matching the application's actual FastAPI entry point. For example, if the application instance is defined in `app/main.py` as `app`:

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

The classifier supports quality assessment; it is separate from the image-processing operations used to enhance the document.

Model performance depends on the training data, preprocessing, and real-world input distribution. Results from synthetic training data should not be treated as proof of equivalent performance on real scanned or handwritten documents.

### Model Weights

Trained model checkpoints are not necessarily included in the Git repository because binary model files can be large.

If a checkpoint is required to run the classifier, obtain the corresponding file from the project's authorized distribution location and configure the application to use it. Ensure that the architecture, preprocessing, label order, and checkpoint are compatible.

Do not assume that a missing checkpoint can be reconstructed from the model's Python source alone.

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

Testing should cover image enhancement, OCR execution, error handling, content-preservation checks, and frontend result presentation.

## Limitations

- Enhancement quality varies depending on the original image.
- Severe blur, missing image details, or very low resolution cannot always be corrected reliably.
- OCR performance may be lower for handwriting, unusual fonts, complex layouts, and poor-quality images.
- Differences in OCR output do not automatically establish that text has been lost.
- Some images may require manual review rather than automatic acceptance.
- The quality classifier's real-world reliability depends on representative training and evaluation data.
- Results should be visually verified before using enhanced documents for official, academic, or other important purposes.

## Future Enhancements

Potential areas for further development include:

- Evaluation on larger and more diverse real-world document datasets.
- Improved handwriting-aware text preservation checks.
- Adaptive enhancement parameters based on document characteristics.
- Quantitative benchmarking using image-quality and OCR metrics.
- Improved confidence reporting and user-facing explanations.
- More comprehensive testing across different document types and capture conditions.

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

Drishti brings together open-source computer vision, deep learning, and OCR technologies to explore practical document-quality improvement and readability assessment.
