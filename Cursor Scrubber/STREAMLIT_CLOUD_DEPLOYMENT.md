# Deploying PDF Redactor Tool on Streamlit Cloud

## 🚀 **Streamlit Cloud Deployment (Recommended)**

### **Why Streamlit Cloud?**
- ✅ **Official Streamlit hosting**
- ✅ **Zero configuration needed**
- ✅ **Always free for public apps**
- ✅ **Custom domains supported**
- ✅ **Automatic HTTPS**
- ✅ **No dependency conflicts**

---

## 📋 **Step-by-Step Deployment**

### **Step 1: Prepare Your Repository**
Your repository is already ready! It contains:
- ✅ `app.py` (main Streamlit app)
- ✅ `requirements-streamlit-cloud.txt` (dependencies)
- ✅ `.streamlit/config.toml` (configuration)
- ✅ All supporting files

### **Step 2: Deploy to Streamlit Cloud**

1. **Go to Streamlit Cloud**
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub

2. **Connect Your Repository**
   - Click "New app"
   - Select your repository: `langebuilds/pdf_scrubber_tool`
   - Set **Main file path**: `Cursor Scrubber/app.py`
   - Set **Python version**: `3.11`

3. **Deploy**
   - Click "Deploy!"
   - Wait 2-3 minutes for deployment

### **Step 3: Access Your App**
- **URL**: `https://your-app-name.streamlit.app`
- **Share this URL** with your team

---

## 🌐 **Adding Custom Domain**

### **Option 1: Streamlit Cloud Custom Domain**
1. Go to your app settings in Streamlit Cloud
2. Click "Custom domain"
3. Add your domain (e.g., `pdf-redactor.yourdomain.com`)
4. Update DNS records as instructed

### **Option 2: Cloudflare Proxy (Recommended)**
1. **Set up Cloudflare** for your domain
2. **Create CNAME record**:
   - Name: `pdf-redactor` (or whatever you want)
   - Target: `your-app-name.streamlit.app`
   - Proxy: ✅ Enabled (orange cloud)
3. **SSL/TLS**: Set to "Full (strict)"
4. **Result**: `https://pdf-redactor.yourdomain.com`

---

## 🔧 **Alternative: Railway Deployment**

If you prefer Railway:

### **Step 1: Deploy to Railway**
1. Go to [railway.app](https://railway.app)
2. Connect GitHub repository
3. Railway will auto-detect it's a Python app
4. Set environment variables:
   ```
   PYTHON_VERSION=3.11
   STREAMLIT_SERVER_PORT=$PORT
   STREAMLIT_SERVER_ADDRESS=0.0.0.0
   ```

### **Step 2: Add Custom Domain**
1. Go to your Railway project
2. Click "Settings" → "Domains"
3. Add your custom domain
4. Update DNS records

---

## 🔧 **Alternative: Heroku Deployment**

### **Step 1: Deploy to Heroku**
```bash
# Install Heroku CLI
brew install heroku/brew/heroku

# Login to Heroku
heroku login

# Create Heroku app
heroku create your-app-name

# Deploy
git push heroku main

# Open app
heroku open
```

### **Step 2: Add Custom Domain**
```bash
# Add custom domain
heroku domains:add pdf-redactor.yourdomain.com

# Update DNS records as shown
```

---

## 📊 **Comparison of Hosting Options**

| Platform | Ease | Cost | Custom Domain | Reliability |
|----------|------|------|---------------|-------------|
| **Streamlit Cloud** | ⭐⭐⭐⭐⭐ | Free | ✅ | ⭐⭐⭐⭐⭐ |
| **Railway** | ⭐⭐⭐⭐ | $5/month | ✅ | ⭐⭐⭐⭐⭐ |
| **Heroku** | ⭐⭐⭐ | $7/month | ✅ | ⭐⭐⭐⭐ |
| **DigitalOcean** | ⭐⭐⭐ | $5/month | ✅ | ⭐⭐⭐⭐ |
| **Render** | ⭐⭐ | Free | ✅ | ⭐⭐ |

---

## 🎯 **My Recommendation**

**Start with Streamlit Cloud** because:
1. **Easiest setup** - just connect GitHub
2. **Always free** for public apps
3. **Perfect for Streamlit apps**
4. **No dependency issues**
5. **Custom domains supported**

**If you need more control**, then try Railway.

---

## 🚀 **Quick Start Commands**

### **For Streamlit Cloud:**
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Connect repository: `langebuilds/pdf_scrubber_tool`
3. Set main file: `Cursor Scrubber/app.py`
4. Deploy!

### **For Railway:**
1. Go to [railway.app](https://railway.app)
2. Connect repository
3. Deploy automatically

---

## 🔒 **Security & Performance**

### **Security Features:**
- ✅ HTTPS automatically enabled
- ✅ No sensitive data in code
- ✅ File uploads are temporary
- ✅ Database is local to deployment

### **Performance Tips:**
- ✅ Use headless OpenCV (already configured)
- ✅ Optimized requirements file
- ✅ Efficient file handling
- ✅ Database for file tracking

---

## 📞 **Support**

- **Streamlit Cloud**: [docs.streamlit.io](https://docs.streamlit.io)
- **Railway**: [docs.railway.app](https://docs.railway.app)
- **Heroku**: [devcenter.heroku.com](https://devcenter.heroku.com)

---

## 🎉 **Expected Result**

After deployment, you'll have:
- **Public URL**: `https://your-app-name.streamlit.app`
- **Custom Domain**: `https://pdf-redactor.yourdomain.com`
- **Global Access**: Available from anywhere
- **All Features**: Database, batch download, file browser
- **Professional**: Clean, fast, reliable 