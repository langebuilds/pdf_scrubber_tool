# PDF Redactor Tool - Deployment Guide

## 🚀 Quick Start

### Option 1: Docker Deployment (Recommended)

1. **Build and run with Docker Compose:**
   ```bash
   docker-compose up -d
   ```

2. **Access the application:**
   - Open your browser to `http://localhost:8501`
   - For remote access: `http://YOUR_SERVER_IP:8501`

### Option 2: Direct Python Deployment

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Setup the application:**
   ```bash
   python deploy.py
   ```

3. **Run the application:**
   ```bash
   # Local development
   streamlit run app.py
   
   # Production (accessible from other computers)
   streamlit run app.py --server.port 8501 --server.address 0.0.0.0
   ```

## 🌐 Web Hosting Options

### 1. **Local Network Access**
- Run with `--server.address 0.0.0.0`
- Access from other computers on the same network
- URL: `http://YOUR_COMPUTER_IP:8501`

### 2. **Cloud Deployment (AWS, Google Cloud, Azure)**
- Deploy using Docker on cloud platforms
- Use load balancers for production traffic
- Configure SSL certificates for HTTPS

### 3. **Streamlit Cloud (Easiest)**
- Push code to GitHub
- Connect to Streamlit Cloud
- Automatic deployment and hosting

### 4. **Heroku/Railway**
- Use the provided Dockerfile
- Deploy as a containerized application
- Automatic scaling and SSL

## 🔧 Configuration

### Environment Variables
```bash
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_SERVER_HEADLESS=true
```

### Firewall Configuration
- Open port 8501 for web access
- Configure firewall rules for your deployment environment

## 📊 Database

The application uses SQLite for storing file metadata:
- **File**: `redaction_database.db`
- **Purpose**: Track processed files, enable file browsing
- **Backup**: Include this file in your backups

## 🔒 Security Considerations

1. **Access Control**: Consider adding authentication for production use
2. **File Storage**: Ensure secure storage of processed files
3. **Network Security**: Use HTTPS in production environments
4. **Data Privacy**: Implement proper data retention policies

## 📈 Performance Optimization

1. **File Storage**: Use cloud storage (S3, GCS) for large files
2. **Caching**: Implement Redis for session management
3. **Load Balancing**: Use multiple instances for high traffic
4. **CDN**: Serve static assets through CDN

## 🐛 Troubleshooting

### Common Issues:
1. **Port already in use**: Change port in configuration
2. **Permission errors**: Ensure proper file permissions
3. **Dependency issues**: Check requirements.txt and system dependencies
4. **Database errors**: Verify SQLite file permissions

### Logs:
- Check Streamlit logs for application errors
- Monitor system resources (CPU, memory, disk)
- Review database integrity

## 📞 Support

For deployment issues:
1. Check the logs for error messages
2. Verify all dependencies are installed
3. Ensure proper network configuration
4. Test with a simple file first 