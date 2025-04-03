# Deploying Backdoor AI Server to Koyeb

This guide explains how to deploy the Backdoor AI server to [Koyeb](https://www.koyeb.com), a developer-friendly serverless platform.

## Prerequisites

1. A [Koyeb account](https://app.koyeb.com/auth/signup)
2. The Backdoor AI repository (this repository)
3. Git installed on your local machine

## Deployment Options

There are two ways to deploy to Koyeb:

1. **Git-based deployment** (recommended): Koyeb builds the application directly from your GitHub repository
2. **Docker-based deployment**: Koyeb uses the provided Dockerfile to build and deploy a container

## Option 1: Git-based Deployment

### Step 1: Connect your GitHub account

1. Log in to the [Koyeb Control Panel](https://app.koyeb.com/)
2. Go to the "Settings" section, then "Integrations" tab
3. Connect your GitHub account by clicking "Connect with GitHub"
4. Grant access to the repository containing the Backdoor AI server

### Step 2: Create a new service

1. From the Koyeb dashboard, click "Create App"
2. Select "GitHub" as the deployment method
3. Choose your repository from the list
4. Select the branch you want to deploy (e.g., `main`)
5. Koyeb will automatically detect the `koyeb.yaml` configuration

### Step 3: Configure environment variables

Set the following environment variables:
- `KOYEB_STORAGE_PATH`: `/var/koyeb/storage` (this is set automatically)
- `PYTHONUNBUFFERED`: `1` (already set in koyeb.yaml)

### Step 4: Add persistent storage

1. Under "Advanced" settings, add a persistent storage volume
2. Set the mount path to `/var/koyeb/storage`
3. Choose an appropriate size (1GB should be sufficient)

### Step 5: Deploy

1. Click "Deploy" to start the deployment process
2. Koyeb will build and deploy your application
3. Once deployed, you can access the server at the provided URL

## Option 2: Docker-based Deployment

### Step 1: Build and push Docker image (optional)

This step is optional, as Koyeb can build the image for you from the Dockerfile.

```bash
# Build the Docker image
docker build -t backdoor-ai-server .

# Push to a registry if needed
docker tag backdoor-ai-server YOUR_REGISTRY/backdoor-ai-server
docker push YOUR_REGISTRY/backdoor-ai-server
```

### Step 2: Create a new service on Koyeb

1. From the Koyeb dashboard, click "Create App"
2. Select "Docker" as the deployment method
3. Choose either "Docker Hub registry" or "GitHub" (to use the Dockerfile in the repo)
4. Configure as in Option 1

## Verifying the Deployment

After deployment, you can verify that the server is running correctly:

1. Access the health check endpoint at `https://your-app-url.koyeb.app/health`
2. You should see a JSON response with system status information

For more detailed diagnostics, you can run the system check script in the Koyeb Web Terminal:

1. Go to your app in the Koyeb dashboard
2. Open the "Terminal" tab
3. Run the diagnostic script:
   ```
   python system_check.py
   ```

## Troubleshooting

### Storage Issues

If you encounter permission errors or missing files:

1. Verify that the persistent volume is correctly mounted at `/var/koyeb/storage`
2. Check that the required directories are created:
   ```
   ls -la /var/koyeb/storage
   ```
3. Run the system check script to diagnose issues

### Application Startup Issues

If the application fails to start:

1. Check the logs in the Koyeb dashboard
2. Verify that Gunicorn is correctly configured in the Procfile
3. Ensure the PORT environment variable is being correctly used

## Updating the Deployment

When you push changes to your GitHub repository, Koyeb will automatically rebuild and redeploy your application if you've configured git-based deployment.

For manual redeployment:

1. Go to your app in the Koyeb dashboard
2. Click "Redeploy" to trigger a new deployment

## Additional Resources

- [Koyeb Documentation](https://www.koyeb.com/docs)
- [Flask on Koyeb Guide](https://www.koyeb.com/tutorials/deploy-flask-on-koyeb)
- [Working with Persistent Storage on Koyeb](https://www.koyeb.com/docs/apps/persistent-storage)

For more help, contact the Koyeb support team or refer to the documentation.
