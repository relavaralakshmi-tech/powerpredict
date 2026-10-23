# PowerPredict — AI Electricity Consumption Prediction

Responsive dashboard inspired by the supplied reference image. Includes Dashboard, Predict, Analytics, History, Insights and Settings pages; FastAPI APIs; configurable MySQL database (SQLite fallback for first run); and a model-training script for `household_power_consumption.txt`.

## Requirements
- Python 3.10+
- Optional: MySQL 8+ (SQLite is used automatically if `DATABASE_URL` is not set)
- Internet connection for the Chart.js CDN on the frontend

## 1. Put the dataset in place
Copy the supplied `household_power_consumption.txt` into `data/household_power_consumption.txt`. The training script handles the dataset's semicolon separator and `?` missing values. It samples rows while reading so it doesn't need to load the entire 127 MB file into memory.

## 2. Install and run on Windows (PowerShell)
```powershell
cd PowerPredict_Full_Project
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env
```
If PowerShell blocks activation, use `.venv\Scripts\python.exe -m pip install -r requirements.txt` and `.venv\Scripts\python.exe -m uvicorn backend.main:app --reload`.

### Configure MySQL (optional but recommended for persistent MySQL storage)
Create a database in MySQL:
```sql
CREATE DATABASE powerpredict CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```
Edit `.env` and set e.g.:
`DATABASE_URL=mysql+pymysql://powerpredict_user:YOUR_PASSWORD@localhost:3306/powerpredict`
Use a dedicated database user and URL-encode special characters in passwords. If no `DATABASE_URL` is configured, the app uses `sqlite:///./powerpredict.db` so it can run immediately.

## 3. Train the AI model
```powershell
python -m backend.train_model
```
Training creates `models/power_model.joblib` and `models/model_metrics.json`. It uses time-derived features and lagged hourly consumption; this is a forecasting baseline, not a guarantee of future accuracy. If training has not been run, the API still works with a transparent historical-average fallback.

## 4. Start the app
```powershell
python -m uvicorn backend.main:app --reload
```
Open http://127.0.0.1:8000. API docs: http://127.0.0.1:8000/docs

## 5. GitHub using VS Code
1. Open the project folder in VS Code.
2. Make sure `.env`, `.venv/`, `data/household_power_consumption.txt`, database files, and trained model artifacts are ignored by `.gitignore` (dataset is large; don't push it).
3. Open Terminal → New Terminal:
```powershell
git init
git add .
git commit -m "Build PowerPredict full-stack app"
```
4. Create an empty **public** repository on GitHub. Copy its HTTPS URL, then:
```powershell
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/PowerPredict.git
git push -u origin main
```
Replace the URL with your repository URL. GitHub authentication may open in a browser.

## 6. Deploy
- **Frontend + API together:** easiest deployment is one Python web service (Render, Railway, or another host supporting Python). Push code to GitHub, create a Web Service, build command `pip install -r requirements.txt`, start command `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`. Add environment variable `DATABASE_URL` pointing to a managed MySQL database. Add `CORS_ORIGINS` if hosting the frontend separately.
- **Separate frontend:** the current frontend is served by FastAPI at `/`, so no separate frontend host is required. If you move it to static hosting, set `API_BASE` in `frontend/app.js` to the deployed API URL and configure `CORS_ORIGINS` on the API.
- **Dataset/model:** do not commit the raw dataset. Train locally and deploy the generated model artifact through your host's supported artifact mechanism, or train in a build/job environment. Persistent disk or managed DB may cost money; check current provider limits/pricing.
- Never commit `.env` or database credentials.

## Important
This is a runnable starter project, not a claim that every deployment provider's free tier supports persistent MySQL or model artifacts. Validate prediction quality with a time-based holdout before presenting model accuracy as a scientific result.
