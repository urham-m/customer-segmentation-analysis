Customer Segmentation Analysis
How to Run
1. Clone the repository
git clone https://github.com/urham-m/customer-segmentation-analysis.git
cd customer-segmentation-analysis
2. Create and activate a virtual environment

Windows PowerShell:

python -m venv .venv
.\.venv\Scripts\Activate.ps1
3. Install dependencies
pip install -r requirements.txt
4. Add the datasets

Place the following files in:

data/raw/
├── users.csv
└── orders.csv
5. Run preprocessing
python src\preprocess.py
6. Run clustering
python src\clustering.py
7. Generate visualizations
python src\visualize.py

Generated datasets and metrics are saved in outputs/, and visualizations are saved in plots/.