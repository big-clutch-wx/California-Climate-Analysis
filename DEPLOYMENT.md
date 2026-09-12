# Deployment Guide - California Climate Analysis Web App

## Quick Start (5 minutes to live)

### Option 1: Streamlit Cloud (Recommended - FREE)

**Step 1: Create GitHub Repository**
```bash
# Initialize git in your project directory
cd E:\ACIS
git init
git add .
git commit -m "Initial commit"

# Create new repo on GitHub.com
# Then push:
git remote add origin https://github.com/YOUR_USERNAME/california-climate.git
git branch -M main
git push -u origin main
```

**Step 2: Deploy to Streamlit Cloud**
1. Go to https://share.streamlit.io/
2. Click "New app"
3. Connect your GitHub account
4. Select your repository
5. Select branch: `main`
6. File path: `app_streamlit.py`
7. Click "Deploy"

✅ Your app is now live! Share the URL with anyone.

---

### Option 2: Self-Hosted (Heroku or Railway)

**Heroku (Free tier ended, but still available for $5/month):**

```bash
# Install Heroku CLI
# Then:
heroku login
heroku create your-app-name
git push heroku main
heroku open
```

**Railway (Easier, $5/month minimum):**

1. Go to https://railway.app
2. Click "New Project"
3. Select "Deploy from GitHub"
4. Connect your repo
5. Set environment: Python 3.11
6. Railway auto-detects and deploys

---

## File Structure Required

```
california-climate/
├── app_streamlit.py          # Main web app
├── compare.py                # Precipitation comparison logic
├── compare_snowfall.py       # Snowfall comparison logic
├── requirements.txt          # Python dependencies
├── stations_meta.jsonl       # Station metadata
├── stations_meta_watersheds.jsonl
├── snowfall_*.parquet        # Snowfall data files (all decades)
├── daily_*.parquet           # Precipitation data files (all decades)
└── README.md
```

---

## Local Testing (Before Deploying)

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run app_streamlit.py

# Opens at http://localhost:8501
```

---

## What Gets Deployed

✅ `app_streamlit.py` - Web UI
✅ `requirements.txt` - Dependencies  
✅ `compare.py` & `compare_snowfall.py` - Analysis logic
✅ `*.parquet` files - Data (keep in repo)
✅ `*.jsonl` files - Metadata

---

## Important Notes

### File Size
- Parquet files are ~500MB-1GB total
- Streamlit Cloud has limits, may need to optimize

### Data Loading
- First load might be slow (parquets are big)
- DuckDB queries run server-side
- Consider caching for performance

### GitHub Limits
- Max 2GB repo size (may exceed with all parquets)
- Solution: Use Git LFS (Large File Storage)

```bash
# Install Git LFS
git lfs install

# Track parquet files
git lfs track "*.parquet"
git add .gitattributes
git commit -m "Add git lfs"
git push
```

---

## Custom Domain (Optional)

After deployment, map a custom domain:
- Streamlit Cloud → Settings → Custom domain
- Railway → Settings → Public Networking → Custom domain
- Point your domain's DNS to the service

Example: `california-climate.com` → your-streamlit-app.streamlit.app

---

## Monitoring & Updates

**Make changes locally:**
```bash
# Edit code, test locally
streamlit run app_streamlit.py

# Commit and push
git add .
git commit -m "Update analysis logic"
git push
```

Streamlit Cloud auto-redeploys on every push!

---

## Troubleshooting

**"Module not found"**
- Check requirements.txt has all imports

**"Data files not found"**
- Ensure *.parquet and *.jsonl files are in repo
- Check file paths in code

**"Out of memory"**
- Streamlit Cloud has 1GB RAM limit
- May need to optimize queries or use Railway

**"Too slow"**
- Add caching: `@st.cache_data`
- Reduce default data range

---

## Next Steps

1. ✅ Test `app_streamlit.py` locally
2. ✅ Push to GitHub
3. ✅ Deploy to Streamlit Cloud
4. ✅ Share the link!

Questions? Let me know!
