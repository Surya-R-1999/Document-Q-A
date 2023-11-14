# Methods to Extract Table's from Documents:

1. Table Transformer (TATR): 
- source : https://github.com/microsoft/table-transformer#table-transformer-tatr
-  A deep learning model based on object detection for extracting tables from PDFs and images.
-  First proposed in "PubTables-1M: Towards comprehensive table extraction from unstructured documents".

![image](https://github.com/Surya-R-1999/Document-Q-A/assets/121089254/66c9dacc-04af-4230-94da-2290860fe095)

This repository also contains the official code for these papers:

"GriTS: Grid table similarity metric for table structure recognition"
"Aligning benchmark datasets for table structure recognition"
Note: If you are looking to use Table Transformer to extract your own tables, here are some helpful things to know:

TATR can be trained to work well across many document domains and everything needed to train your own model is included here. But at the moment pre-trained model weights are only available for TATR trained on the PubTables-1M dataset. (See the additional documentation for how to train your own multi-domain model.)
TATR is an object detection model that recognizes tables from image input. The inference code built on TATR needs text extraction (from OCR or directly from PDF) as a separate input in order to include text in its HTML or CSV output.


2. pdfplumber
- source : https://github.com/jsvine/pdfplumber
- Plumb a PDF for detailed information about each text character, rectangle, and line. Plus: Table extraction and visual debugging.
-  Works best on machine-generated, rather than scanned, PDFs. Built on pdfminer.six.
- Currently tested on Python 3.8, 3.9, 3.10, 3.11.
- ![image](https://github.com/Surya-R-1999/Document-Q-A/assets/121089254/d269fd71-ff28-456a-9347-aa5c84ed4dbe)

3. Multi-Type-TD-TSR
- source : https://github.com/Psarpei/Multi-Type-TD-TSR
- Extracting Tables from Document Images using a Multi-stage Pipeline for Table Detection and Table Structure Recognition
- ![image](https://github.com/Surya-R-1999/Document-Q-A/assets/121089254/7e3f4288-38e8-4c3c-9a51-3c7404ebdb99)

4. img2table (Multi Page Images Not Supported)
- source :  https://github.com/xavctn/img2table
- img2table is a simple, easy to use, table identification and extraction Python Library based on OpenCV image processing that supports most common image file formats as well as PDF files.

5. CascadeTabNet
- CascadeTabNet: An approach for end to end table detection and structure recognition from image-based documents
- The paper was presented (Orals) at **CVPR 2020 Workshop on Text and Documents in the Deep Learning Era**
- Table Detection:
- ![image](https://github.com/Surya-R-1999/Document-Q-A/assets/121089254/ac19c64c-aa3f-407b-80ce-451a89ca4d99)
- Table Structure Recognition:(Bordered Table and Borderless table)
- ![image](https://github.com/Surya-R-1999/Document-Q-A/assets/121089254/379edc57-7172-4fec-ba89-610a9618bce6)

