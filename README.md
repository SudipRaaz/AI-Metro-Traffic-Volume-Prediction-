# 1. Open the project folder
cd DATA_PROJECT

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate       # macOS / Linux
.\venv\Scripts\Activate.ps1    # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run main.py
python main.py

Note: Python verion 3.14.4 or higher is recommended or you could get dependency error with third party libraries eg: pandas. 