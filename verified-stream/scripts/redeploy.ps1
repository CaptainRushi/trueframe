# Helper script to trigger a Vercel redeployment (Reload the project)
# Requires Vercel CLI to be installed: npm install -g vercel

Write-Host "Triggering Vercel Redeployment..." -ForegroundColor Cyan

# Check if Vercel CLI is installed
if (!(Get-Command vercel -ErrorAction SilentlyContinue)) {
    Write-Host "Error: Vercel CLI not found." -ForegroundColor Red
    Write-Host "Please install it with: npm install -g vercel"
    exit
}

# Redeploy the current project
vercel redeploy

Write-Host "Done! Your Vercel project is being reloaded/repackaged." -ForegroundColor Green
