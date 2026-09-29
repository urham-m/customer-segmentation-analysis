Customer Segmentation Analysis
How to Run
1. Clone the repository
git clone https://github.com/urham-m/customer-segmentation-analysis.git

2. Go into the project folder
cd customer-segmentation-analysis

3. Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

4. Install dependencies
pip install -r requirements.txt

5. Run preprocessing
python src\preprocess.py

6. Run clustering
python src\clustering.py

7. Generate visualizations
python src\visualize.py

Generated datasets and metrics are saved in outputs/, and visualizations are saved in plots/.