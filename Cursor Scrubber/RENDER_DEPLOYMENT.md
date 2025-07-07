# Deploying PDF Redactor Tool on Render

## 🚀 Quick Deploy on Render

### Option 1: Automatic Deployment (Recommended)

1. **Fork or push your code to GitHub**
   - Make sure all files are in your repository
   - Include the `render.yaml` configuration file

2. **Connect to Render**
   - Go to [render.com](https://render.com)
   - Sign up/login with your GitHub account
   - Click "New +" and select "Blueprint"

3. **Deploy from Blueprint**
   - Render will automatically detect the `render.yaml` file
   - Click "Connect" to your GitHub repository
   - Render will automatically configure everything
   - Click "Apply" to start deployment

4. **Access your application**
   - Render will provide a URL like: `https://your-app-name.onrender.com`
   - The app will be accessible from anywhere on the internet

### Option 2: Manual Deployment

1. **Create a new Web Service**
   - Go to Render Dashboard
   - Click "New +" → "Web Service"
   - Connect your GitHub repository

2. **Configure the service**
   - **Name**: `pdf-redactor-tool`
   - **Environment**: `Python`
   - **Build Command**: `pip install -r requirements-render.txt`
   - **Start Command**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

3. **Environment Variables**
   - `PYTHON_VERSION`: `3.11.0`
   - `STREAMLIT_SERVER_PORT`: `$PORT`
   - `STREAMLIT_SERVER_ADDRESS`: `0.0.0.0`
   - `STREAMLIT_SERVER_HEADLESS`: `true`
   - `STREAMLIT_BROWSER_GATHER_USAGE_STATS`: `false`

4. **Deploy**
   - Click "Create Web Service"
   - Render will build and deploy your application

## 🔧 Render-Specific Optimizations

### 1. **Headless OpenCV**
- Using `opencv-python-headless` instead of `opencv-python`
- No GUI dependencies required
- Smaller image size and faster builds

### 2. **Environment Variables**
- `$PORT`: Automatically provided by Render
- `0.0.0.0`: Allows external access
- `headless=true`: No browser requirements

### 3. **File Storage Considerations**
- **Important**: Render has ephemeral file storage
- Files uploaded and processed will be lost when the service restarts
- Consider using external storage for production use

## 📁 File Structure for Render

```
your-repository/
├── app.py                    # Main Streamlit application
├── database.py               # Database management
├── pdf_redactor.py           # Core redaction logic
├── logo_detector.py          # Logo detection
├── config.py                 # Configuration
├── requirements-render.txt   # Render-optimized dependencies
├── render.yaml               # Render configuration
├── templates/                # Logo templates
└── README.md                 # Documentation
```

## 🌐 Accessing Your Application

### After Deployment
- **URL**: `https://your-app-name.onrender.com`
- **Access**: Available from anywhere on the internet
- **HTTPS**: Automatically provided by Render

### Sharing Access
- Share the Render URL with your team
- No need to worry about IP addresses or network configuration
- Works from any device with internet access

## 🔒 Security & Limitations

### Render Limitations
1. **File Persistence**: Files are not persistent across restarts
2. **Memory Limits**: Starter plan has memory limitations
3. **Cold Starts**: Service may sleep after inactivity

### Recommendations
1. **For Production**: Consider upgrading to paid plans
2. **File Storage**: Use external storage (AWS S3, Google Cloud Storage)
3. **Database**: Consider external database for file metadata
4. **Backup**: Implement backup strategies for important data

## 💰 Pricing

### Free Tier
- **Cost**: $0/month
- **Limitations**: 
  - Service sleeps after 15 minutes of inactivity
  - 750 hours/month
  - Limited memory and storage

### Paid Plans
- **Starter**: $7/month
- **Standard**: $25/month
- **Pro**: $85/month

## 🚀 Production Deployment

### For Production Use:
1. **Upgrade Plan**: Use paid plan for better performance
2. **External Storage**: Implement cloud storage for files
3. **Database**: Use external database (PostgreSQL, MongoDB)
4. **Monitoring**: Set up health checks and monitoring
5. **Backup**: Implement automated backups

### Environment Variables for Production:
```bash
# Add these to your Render environment variables
DATABASE_URL=your_external_database_url
STORAGE_BUCKET=your_cloud_storage_bucket
SECRET_KEY=your_secret_key
```

## 🐛 Troubleshooting

### Common Issues:
1. **Build Failures**: Check requirements-render.txt
2. **Port Issues**: Ensure using `$PORT` environment variable
3. **File Access**: Remember files are ephemeral
4. **Memory Issues**: Upgrade plan if needed

### Debug Commands:
```bash
# Check logs in Render dashboard
# Monitor resource usage
# Verify environment variables
```

## 📞 Support

- **Render Support**: [help.render.com](https://help.render.com)
- **Documentation**: [render.com/docs](https://render.com/docs)
- **Community**: Render Discord and forums

## 🎉 Benefits of Render Deployment

✅ **Easy Setup**: One-click deployment from GitHub  
✅ **Automatic HTTPS**: SSL certificates included  
✅ **Global Access**: Available from anywhere  
✅ **Scalable**: Easy to upgrade plans  
✅ **Reliable**: 99.9% uptime guarantee  
✅ **Cost-Effective**: Free tier available  
✅ **No Server Management**: Fully managed platform 