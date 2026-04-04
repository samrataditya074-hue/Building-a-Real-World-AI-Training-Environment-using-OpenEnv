# Deployment Script for Hugging Face Spaces

# Run this script using PowerShell

# 1. Login to huggingface (Will prompt for Read/Write Token)
Write-Host "Logging into Hugging Face..."
huggingface-cli login

# 2. Push the OpenEnv repository to Hugging Face
Write-Host "Pushing OpenEnv Architecture to Hugging Face Spaces..."
openenv push --space-name autonomous-ceo-simulator

Write-Host "Deployment Triggered! Follow the URL above to view your Space."
